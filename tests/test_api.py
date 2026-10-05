"""Integration, end-to-end and security tests against a real Postgres."""
import psycopg
import pytest

from core import db, scenario
from drc import engine, world

from .conftest import bearer

CMD, DISP, VIEW, OTHER = bearer("gulf", "commander"), bearer("gulf", "dispatcher"), bearer("gulf", "viewer"), bearer("other", "commander")
IN_COUNTY = {"lat": 27.745, "lon": -82.59}


def test_classifier_beats_keyword_rules(seeded):
    m = seeded["metrics"]
    assert m["model"]["accuracy"] > m["baseline"]["accuracy"] + 0.03


def test_nothing_works_before_a_scenario_is_loaded(client):
    assert client.get("/v1/map", headers=VIEW).json()["code"] == "no_scenario"
    assert client.post("/v1/incidents", headers=DISP, json={"text": "house on fire", **IN_COUNTY}).status_code == 409


@pytest.fixture(scope="module")
def demo(client):
    results, _ = scenario.run(client)
    return results


@pytest.fixture(scope="module")
def truth():
    return world.build(7)


def test_demo_journey_matches_the_simulated_storm(demo, truth):
    assert demo["load"]["population"] == sum(truth["pop"]) and demo["map0"]["hour"] == 0 and not any(e[3] for e in demo["map0"]["edges"])
    risk = demo["risk"]
    assert len(risk["p_isolated"]) == world.N and risk["likely_isolated"] >= risk["of_which_dry_but_cut_off"] >= 1
    evac = demo["evac"]
    assert evac["neighbourhoods"] == risk["likely_isolated"] and evac["placed"] + evac["unmet"] == evac["evacuees"] and evac["unmet"] > 0
    assert all(s["inflow"] <= s["capacity"] for s in evac["shelters"]) and all(r["path"][0] == r["from_node"] for r in evac["routes"])
    assert demo["alert"]["recipients"] == evac["population"] and demo["alert"]["shelter_places_reserved"] == evac["placed"]
    a4 = demo["adv4"]
    reps = [r for r in truth["reports"] if r["hour"] <= 4]
    assert a4["reports"] == len(reps) == a4["incidents_opened"] + a4["reports_merged_into_open_incidents"] + a4["filtered_as_noise"]
    assert a4["filtered_as_noise"] == sum(r["incident"] is None for r in reps) and a4["bridges_open"] == 2 and len(a4["events"]) == 2
    assert demo["adv6"]["bridges_open"] == 1 and any("West General" in e for e in demo["adv6"]["events"]) and all(s["occupancy"] == s["capacity"] for s in demo["adv6"]["shelters"])
    assert demo["adv6"]["units_arrived"] == len(demo["plan4"]["assignments"]) > 0
    for plan in (demo["plan4"], demo["plan6"], demo["plan6b"]):
        units = [a["unit_id"] for a in plan["assignments"]]
        assert len(units) == len(set(units)) and all(a["unit_type"] in engine.CAN_SERVE[a["type"]] for a in plan["assignments"])
        assert plan["optimized"]["weighted_wait"] <= plan["baseline"]["weighted_wait"] and len(plan["assignments"]) + len(plan["waiting"]) == plan["incidents"]
        assert all(a["hospital"] != "West General" for a in plan["assignments"]) or plan is demo["plan4"]
    assert demo["plan6"]["waiting"] and demo["plan6"]["weighted_wait_reduction"] > 0.05 and demo["plan6"]["held_for_verification"] > 0
    broken = demo["plan6"]["assignments"][0]
    assert demo["tele"]["released_incident"] == broken["incident_id"] and broken["unit_id"] not in [a["unit_id"] for a in demo["plan6b"]["assignments"]]
    assert demo["plan6b"]["units"] == demo["plan6"]["units"] - 1 and demo["plan6b"]["units_kept"] > 0
    v = demo["verify"]
    assert v["model"]["brier"] < v["baseline"]["brier"] and v["actually_isolated"] == v["model"]["hits"] + v["model"]["misses"] == sum(v["actual"])
    actions = {e["action"] for e in demo["audit"]["events"]}
    assert demo["audit"]["chain_valid"] and {"dispatch.optimized", "resource.status_changed", "scenario.advanced"} <= actions


def test_a_unit_never_has_two_live_dispatches(demo):
    with db.tx() as c:
        d = c.execute("SELECT tenant_id, incident_id, resource_id, plan_id FROM dispatches WHERE status = 'en_route' LIMIT 1").fetchone()
    with pytest.raises(psycopg.errors.UniqueViolation), db.tx() as c:
        c.execute("INSERT INTO dispatches (id, tenant_id, incident_id, resource_id, eta_seconds, status, plan_id) VALUES (gen_random_uuid(),%s,%s,%s,60,'en_route',%s)", list(d.values()))


def test_a_retried_report_is_not_counted_twice(client, demo):
    body = {"text": "Smoke pouring out of the store on the corner", "source": "social", **IN_COUNTY}
    key = {**DISP, "Idempotency-Key": "report-1"}
    first = client.post("/v1/incidents", headers=key, json=body)
    again = client.post("/v1/incidents", headers=key, json=body)
    assert first.status_code == 201 and again.headers["Idempotent-Replay"] == "true" and again.json() == first.json()
    out = first.json()
    assert out["predicted_type"] == "fire" and out["action"] == "created" and out["dispatchable"] is False        # one social-media report is not enough to send a unit
    assert client.post("/v1/incidents", headers=key, json={**body, "source": "app"}).json()["code"] == "idempotency_key_reused"
    second = client.post("/v1/incidents", headers=DISP, json={**body, "source": "911", "severity": 5}).json()
    assert second["action"] == "corroborated" and second["incident_id"] == out["incident_id"] and second["dispatchable"] and second["incident"]["severity"] == 5 and second["incident"]["reports"] == 2
    assert client.post("/v1/incidents", headers=DISP, json={"text": "thanks to all the first responders", **IN_COUNTY}).json()["action"] == "filtered_as_noise"


def test_incident_search_pages_without_repeats(client, demo):
    seen, cursor = [], None
    while True:
        page = client.get("/v1/incidents", headers=VIEW, params={"limit": 7, **({"cursor": cursor} if cursor else {})}).json()
        seen += [i["id"] for i in page["items"]]
        cursor = page["next_cursor"]
        if not cursor:
            break
    with db.tx() as c:
        total = c.execute("SELECT count(*) AS n FROM incidents WHERE tenant_id = %s", [seededid(c)]).fetchone()["n"]
    assert len(seen) == len(set(seen)) == total
    box = client.get("/v1/incidents", headers=VIEW, params={"bbox": "-82.65,27.69,-82.50,27.705", "severity": 4, "limit": 200}).json()["items"]
    assert box and all(i["y"] <= 0 and i["severity"] >= 4 for i in box)


def seededid(c):
    return c.execute("SELECT id FROM tenants WHERE name LIKE 'Gulf%'").fetchone()["id"]


@pytest.mark.parametrize("headers,method,path,body,expected", [
    ({}, "GET", "/v1/map", None, 401),
    (VIEW, "GET", "/v1/map", None, 200),
    (VIEW, "POST", "/v1/incidents", {"text": "house on fire", **IN_COUNTY}, 403),
    (VIEW, "POST", "/v1/optimize/dispatch", None, 403),
    (DISP, "POST", "/v1/scenarios/hurricane:advance", {"to_hour": 9}, 403),
    (DISP, "POST", "/v1/alerts/broadcast", {"message": "leave now please"}, 403),
    (DISP, "GET", "/v1/audit", None, 403),
    (CMD, "POST", "/v1/scenarios/hurricane:advance", {"to_hour": 2}, 409),
    (CMD, "POST", "/v1/scenarios/hurricane:advance", {"to_hour": 99}, 422),
    (VIEW, "GET", "/v1/incidents?bbox=1,2,3", None, 422),
    (VIEW, "GET", "/v1/incidents?bbox=a,b,c,d", None, 422),
    (DISP, "POST", "/v1/incidents", {"text": "house on fire", "lat": 40.0, "lon": -82.59}, 422),
    (VIEW, "GET", "/v1/risk/isolation?target_hour=2", None, 422),
])
def test_authorization_and_validation(client, demo, headers, method, path, body, expected):
    assert client.request(method, path, headers=headers, json=body).status_code == expected


def test_tenant_isolation(client, demo):
    unit = demo["plan6b"]["assignments"][0]["unit_id"]
    assert client.get("/v1/map", headers=OTHER).json()["code"] == "no_scenario"
    assert client.post(f"/v1/resources/{unit}/telemetry", headers=OTHER, json={**IN_COUNTY, "status": "out_of_service"}).status_code == 404
    before = client.get("/v1/map", headers=VIEW).json()
    assert client.post("/v1/scenarios/hurricane", headers=OTHER).status_code == 201                # their load deletes "everything": only their own rows
    assert client.post("/v1/scenarios/hurricane:advance", headers=OTHER, json={"to_hour": 5}).status_code == 200
    theirs = client.get("/v1/map", headers=OTHER).json()
    assert theirs["hour"] == 5 and theirs["elevation"] != before["elevation"]
    assert client.get("/v1/map", headers=VIEW).json() == before
    assert client.get("/v1/incidents", headers=OTHER, params={"limit": 200}).json()["items"] and not {i["id"] for i in client.get("/v1/incidents", headers=OTHER, params={"limit": 200}).json()["items"]} & {a["incident_id"] for a in demo["plan6b"]["assignments"]}
