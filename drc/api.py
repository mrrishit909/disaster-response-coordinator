"""Public API (modular monolith). Blueprint services map to: ingestion-service (reports -> incidents), incident-service,
resource-service (telemetry), geospatial-service (road state, map), ml-risk-service (isolation forecast), optimization-service
(dispatch, shelter allocation), notification-service (alerts, recorded only) and audit-service (core/audit.py).

    uvicorn drc.api:app
"""
import pickle
import uuid
from typing import Literal

import numpy as np
from fastapi import Depends, Header
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field
from scipy.sparse.csgraph import dijkstra

from core import audit, db
from core.app import Ctx, Problem, create_app, run

from . import engine, world
from .seed import TENANTS

READ = {"ops:read", "jobs:read"}
DISPATCH = READ | {"incidents:write", "resources:write", "dispatch:optimize"}
PERMISSIONS = {"viewer": READ, "dispatcher": DISPATCH, "commander": DISPATCH | {"scenario:control", "alerts:broadcast", "audit:read"}}
app = create_app("disaster-response-coordinator", PERMISSIONS)
auth = app.state.auth
IdemKey = Header(None, alias="Idempotency-Key")
EVACUATING_SHARE = 0.12            # share of an at-risk neighbourhood assumed to need a public shelter (an assumption, shown as such)
_cache = {}


def clf():
    if "clf" not in _cache:
        with db.tx() as c:
            _cache["clf"] = pickle.loads(c.execute("SELECT artifact FROM model_registry WHERE name = 'report-classifier' AND state = 'production'").fetchone()["artifact"])   # our own training output
    return _cache["clf"]


def latlon(n):
    return world.LAT0 + (n // world.W) * world.DLAT, world.LON0 + (n % world.W) * world.DLON


def to_node(lat, lon):
    x, y = round((lon - world.LON0) / world.DLON), round((lat - world.LAT0) / world.DLAT)
    if not (0 <= x < world.W and 0 <= y < world.H):
        raise Problem(422, "outside_county", "the point is outside the county this tenant covers")
    return world.node(x, y)


def state(c):
    row = c.execute("SELECT hour, seed FROM sim_clock").fetchone()
    if not row:
        raise Problem(409, "no_scenario", "load the scenario first: POST /v1/scenarios/hurricane")
    return world.build(row["seed"]), row["hour"]


def closed_now(c, w):
    idx = {int(r["road_segment_id"][1:]) for r in c.execute("SELECT road_segment_id FROM road_events WHERE ends_at IS NULL")}
    return [i in idx for i in range(len(w["edges"]))]


def open_hospitals(c):
    return [r["node"] for r in c.execute("SELECT node FROM hospitals WHERE status = 'open' ORDER BY node")]


def risk(w, hour, target, hospitals):
    key = ("risk", w["seed"], hour, target, tuple(hospitals))
    if key not in _cache:
        _cache[key] = engine.isolation_risk(w, hour, target, hospitals)
    return _cache[key]


# ---------------------------------------------------------------- scenario control (the simulator standing in for live feeds)

@app.post("/v1/scenarios/hurricane", status_code=201, tags=["scenario"], summary="(+) Load the synthetic county at hour 0, six hours before peak surge. Resets this tenant's operational data.")
def load_scenario(ctx: Ctx = Depends(auth("scenario:control")), idem: str | None = IdemKey):
    def work(c):
        name = c.execute("SELECT name FROM tenants WHERE id = %s", [ctx.tenant_id]).fetchone()["name"]
        w = world.build(next(seed for n, seed in TENANTS.values() if n == name))
        for t in ("dispatches", "reports", "incidents", "road_events", "alerts", "resources", "shelters", "hospitals", "sim_clock"):
            c.execute(f"DELETE FROM {t}")                         # row-level security limits this to the caller's tenant
        c.execute("INSERT INTO sim_clock VALUES (%s, 0, %s)", [ctx.tenant_id, w["seed"]])
        for i, u in enumerate(w["units"]):
            c.execute("INSERT INTO resources (id, tenant_id, resource_type, status, lat, lon, name, node) VALUES (%s,%s,%s,'available',%s,%s,%s,%s)",
                      [uuid.uuid5(ctx.tenant_id, f"unit-{i}"), ctx.tenant_id, u["type"], *latlon(u["node"]), u["name"], u["node"]])
        for i, s in enumerate(w["shelters"]):
            c.execute("INSERT INTO shelters (id, tenant_id, name, capacity, lat, lon, node) VALUES (%s,%s,%s,%s,%s,%s,%s)", [uuid.uuid5(ctx.tenant_id, f"shelter-{i}"), ctx.tenant_id, s["name"], s["capacity"], *latlon(s["node"]), s["node"]])
        for i, h in enumerate(w["hospitals"]):
            c.execute("INSERT INTO hospitals VALUES (%s,%s,%s,%s,%s,'open',%s)", [uuid.uuid5(ctx.tenant_id, f"hospital-{i}"), ctx.tenant_id, h["name"], h["beds"], round(0.6 * h["beds"]), h["node"]])
        audit.record(c, ctx, "scenario.loaded", "tenant", ctx.tenant_id, {"seed": w["seed"]})
        mean, sd = world.forecast(0)
        return 201, {"hour": 0, "neighbourhoods": world.N, "population": sum(w["pop"]), "road_segments": len(w["edges"]), "bridges": len(world.BRIDGES), "shelters": len(w["shelters"]),
                     "shelter_places": sum(s["capacity"] for s in w["shelters"]), "hospitals": len(w["hospitals"]), "units": {t: sum(u["type"] == t for u in w["units"]) for t in ("boat", "ambulance", "engine")},
                     "forecast": {"peak_surge_m": mean, "plus_minus_m": sd, "peak_in_hours": world.PEAK_HOUR}}
    return run(ctx, idem, {}, work)


class AdvanceIn(BaseModel):
    to_hour: int = Field(ge=1, le=12)


def ingest(c, ctx, hour, text, node, source, hint=None):
    """One citizen report: classify it, then either open an incident, add it to one already open nearby, or file it as noise."""
    model = clf()
    proba = model.predict_proba([text])[0]
    kind = str(model.classes_[proba.argmax()])
    cred = round(float(proba.max()) * dict(world.SOURCES)[source], 4)
    x, y = node % world.W, node // world.W
    inc_id, action = None, "filtered_as_noise"
    if kind != "noise":
        inc = c.execute("""SELECT id, confidence FROM incidents WHERE status IN ('open','dispatched') AND type = %s AND abs(x - %s) + abs(y - %s) <= %s AND hour >= %s
                            ORDER BY reported_at LIMIT 1 FOR UPDATE""", [kind, x, y, engine.MERGE_BLOCKS, hour - 2]).fetchone()
        if inc:
            inc_id, action = inc["id"], "corroborated"
            c.execute("UPDATE incidents SET confidence = %s, reports = reports + 1, severity = coalesce(%s, severity) WHERE id = %s", [round(engine.corroborate(float(inc["confidence"]), cred), 4), hint, inc_id])
        else:
            inc_id, action = uuid.uuid7(), "created"
            c.execute("INSERT INTO incidents (id, tenant_id, type, severity, status, lat, lon, source, confidence, node, x, y, hour) VALUES (%s,%s,%s,%s,'open',%s,%s,%s,%s,%s,%s,%s,%s)",
                      [inc_id, ctx.tenant_id, kind, hint or world.DEFAULT_SEVERITY[kind], *latlon(node), source, cred, node, x, y, hour])
    rid = uuid.uuid7()
    c.execute("INSERT INTO reports (id, tenant_id, incident_id, body, source, node, hour, predicted_type, credibility) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)", [rid, ctx.tenant_id, inc_id, text, source, node, hour, kind, cred])
    return {"report_id": rid, "predicted_type": kind, "credibility": cred, "action": action, "incident_id": inc_id}


def step(c, ctx, w, h, out):
    for d in c.execute("SELECT d.id, d.incident_id, d.resource_id, d.hospital_id, i.node FROM dispatches d JOIN incidents i ON i.id = d.incident_id WHERE d.status = 'en_route'").fetchall():
        c.execute("UPDATE dispatches SET status = 'arrived' WHERE id = %s", [d["id"]])
        c.execute("UPDATE incidents SET status = 'resolved' WHERE id = %s", [d["incident_id"]])
        c.execute("UPDATE resources SET status = 'available', node = %s, lat = %s, lon = %s, last_seen_at = now() WHERE id = %s", [d["node"], *latlon(d["node"]), d["resource_id"]])
        if d["hospital_id"]:
            c.execute("UPDATE hospitals SET occupied = least(beds, occupied + 1) WHERE id = %s", [d["hospital_id"]])
        out["units_arrived"] += 1
    water = world.closed(w, world.TRUE_PEAK, h, w["elev_true"])
    was = closed_now(c, w)
    for i, before in enumerate(was):
        a, _, kind = w["edges"][i]
        now = water[i] or (h >= 5 and i in w["debris"])
        if now and not before:
            event = "bridge_closed" if kind == "bridge" else "flooded" if water[i] else "debris"
            c.execute("INSERT INTO road_events (id, tenant_id, road_segment_id, event_type, severity, hour) VALUES (%s,%s,%s,%s,%s,%s)", [uuid.uuid7(), ctx.tenant_id, f"e{i}", event, 5 if kind == "bridge" else 3, h])
            out["roads_closed"] += 1
            if kind == "bridge":
                out["events"].append(f"hour {h}: the bridge at row {a // world.W} closed")
        elif before and not now:
            c.execute("UPDATE road_events SET ends_at = now() WHERE road_segment_id = %s AND ends_at IS NULL", [f"e{i}"])
            out["roads_reopened"] += 1
    if h == world.PEAK_HOUR:
        c.execute("UPDATE hospitals SET status = 'diverting' WHERE name = 'West General'")
        out["events"].append(f"hour {h}: West General lost mains power and is diverting new patients")
    c.execute("UPDATE shelters SET occupancy = least(capacity, round(planned_inflow * least(1, %s / 5.0) + capacity * 0.04 * %s))", [h, h])
    for rep in (x for x in w["reports"] if x["hour"] == h):
        r = ingest(c, ctx, h, rep["text"], rep["node"], rep["source"], rep["severity_hint"])
        out["reports"] += 1
        out[{"created": "incidents_opened", "corroborated": "reports_merged_into_open_incidents", "filtered_as_noise": "filtered_as_noise"}[r["action"]]] += 1


@app.post("/v1/scenarios/hurricane:advance", tags=["scenario"], summary="(+) Move the scenario clock forward: roads flood, bridges close, units arrive, shelters fill, citizen reports come in")
def advance(body: AdvanceIn, ctx: Ctx = Depends(auth("scenario:control")), idem: str | None = IdemKey):
    def work(c):
        w, hour = state(c)
        if body.to_hour <= hour:
            raise Problem(409, "clock_only_moves_forward", f"the scenario is already at hour {hour}")
        out = {"from_hour": hour, "to_hour": body.to_hour, "units_arrived": 0, "roads_closed": 0, "roads_reopened": 0, "reports": 0, "incidents_opened": 0, "reports_merged_into_open_incidents": 0, "filtered_as_noise": 0, "events": []}
        for h in range(hour + 1, body.to_hour + 1):
            step(c, ctx, w, h, out)
        c.execute("UPDATE sim_clock SET hour = %s", [body.to_hour])
        out["open_incidents"] = c.execute("SELECT type, count(*) AS n, count(*) FILTER (WHERE confidence < %s) AS unverified FROM incidents WHERE status = 'open' GROUP BY type ORDER BY n DESC", [engine.MIN_CONFIDENCE]).fetchall()
        out["bridges_open"] = len(world.BRIDGES) - c.execute("SELECT count(*) AS n FROM road_events WHERE ends_at IS NULL AND event_type = 'bridge_closed'").fetchone()["n"]
        out["roads_closed_now"] = sum(closed_now(c, w))
        out["shelters"] = c.execute("SELECT name, capacity, occupancy FROM shelters ORDER BY name").fetchall()
        out["hospitals"] = c.execute("SELECT name, beds, occupied, status FROM hospitals ORDER BY name").fetchall()
        audit.record(c, ctx, "scenario.advanced", "tenant", ctx.tenant_id, {k: out[k] for k in ("from_hour", "to_hour", "roads_closed", "reports", "incidents_opened")})
        return 200, out
    return run(ctx, idem, body, work)


# ---------------------------------------------------------------- ingestion-service + incident-service

class ReportIn(BaseModel):
    text: str = Field(min_length=3, max_length=500)
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    source: Literal["911", "app", "social"] = "app"
    severity: int | None = Field(None, ge=1, le=5, description="Only honoured from the 911 source, where a call-taker set it")


@app.post("/v1/incidents", status_code=201, tags=["incidents"], summary="Submit a citizen report; it opens an incident, corroborates one nearby, or is filed as noise")
def report(body: ReportIn, ctx: Ctx = Depends(auth("incidents:write")), idem: str | None = IdemKey):
    def work(c):
        _, hour = state(c)
        out = ingest(c, ctx, hour, body.text.lower(), to_node(body.lat, body.lon), body.source, body.severity if body.source == "911" else None)
        if out["incident_id"]:
            out["incident"] = c.execute("SELECT type, severity, confidence, reports, status FROM incidents WHERE id = %s", [out["incident_id"]]).fetchone()
            out["dispatchable"] = float(out["incident"]["confidence"]) >= engine.MIN_CONFIDENCE
        audit.record(c, ctx, "report.received", "report", out["report_id"], {"action": out["action"], "type": out["predicted_type"]})
        return 201, out
    return run(ctx, idem, body, work)


@app.get("/v1/incidents", tags=["incidents"], summary="Incidents by bounding box (min_lon,min_lat,max_lon,max_lat), minimum severity and status; cursor-paginated")
def incidents(bbox: str | None = None, severity: int = 1, status: str | None = None, limit: int = 50, cursor: uuid.UUID | None = None, ctx: Ctx = Depends(auth("ops:read"))):
    try:
        box = [float(v) for v in bbox.split(",")] if bbox else [-180, -90, 180, 90]
        assert len(box) == 4 and box[0] <= box[2] and box[1] <= box[3]
    except (ValueError, AssertionError):
        raise Problem(422, "bad_bbox", "bbox is min_lon,min_lat,max_lon,max_lat") from None
    limit = max(1, min(limit, 200))
    with db.tx(ctx.tenant_id) as c:
        rows = c.execute("""SELECT id, type, severity, status, lat, lon, x, y, source, confidence, reports, hour, confidence < %s AS needs_verification FROM incidents
                             WHERE lon BETWEEN %s AND %s AND lat BETWEEN %s AND %s AND severity >= %s AND (%s::text IS NULL OR status = %s) AND (%s::uuid IS NULL OR id > %s)
                             ORDER BY id LIMIT %s""", [engine.MIN_CONFIDENCE, box[0], box[2], box[1], box[3], severity, status, status, cursor, cursor, limit + 1]).fetchall()
    return jsonable_encoder({"items": rows[:limit], "next_cursor": rows[limit - 1]["id"] if len(rows) > limit else None})


# ---------------------------------------------------------------- resource-service

class TelemetryIn(BaseModel):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    status: Literal["available", "out_of_service"] | None = None


@app.post("/v1/resources/{resource_id}/telemetry", tags=["resources"], summary="Responder position and status. Going out of service releases the unit's incident back to the queue")
def telemetry(resource_id: uuid.UUID, body: TelemetryIn, ctx: Ctx = Depends(auth("resources:write")), idem: str | None = IdemKey):
    def work(c):
        u = c.execute("SELECT id, name, status FROM resources WHERE id = %s FOR UPDATE", [resource_id]).fetchone()
        if not u:
            raise Problem(404, "resource_not_found")
        released = None
        if body.status == "out_of_service" or (body.status == "available" and u["status"] == "out_of_service"):
            d = c.execute("UPDATE dispatches SET status = 'superseded' WHERE resource_id = %s AND status = 'en_route' RETURNING incident_id", [resource_id]).fetchone()
            if d:
                released = d["incident_id"]
                c.execute("UPDATE incidents SET status = 'open' WHERE id = %s", [released])
            c.execute("UPDATE resources SET status = %s WHERE id = %s", [body.status, resource_id])
            audit.record(c, ctx, "resource.status_changed", "resource", resource_id, {"status": body.status, "released_incident": released})
        n = to_node(body.lat, body.lon)
        c.execute("UPDATE resources SET lat = %s, lon = %s, node = %s, last_seen_at = now() WHERE id = %s", [body.lat, body.lon, n, resource_id])
        return 200, {"resource_id": resource_id, "name": u["name"], "status": body.status or u["status"], "released_incident": released}
    return run(ctx, idem, body, work)


@app.get("/v1/resources", tags=["resources"], summary="(+) Response units and where they are")
def resources(ctx: Ctx = Depends(auth("ops:read"))):
    with db.tx(ctx.tenant_id) as c:
        return jsonable_encoder({"items": c.execute("SELECT id, name, resource_type, status, node, lat, lon, last_seen_at FROM resources ORDER BY resource_type, name").fetchall()})


# ---------------------------------------------------------------- ml-risk-service

@app.get("/v1/risk/isolation", tags=["risk"], summary="(+) Chance that each neighbourhood has no open road to a working hospital at the target hour")
def isolation(target_hour: int = world.PEAK_HOUR, ctx: Ctx = Depends(auth("ops:read"))):
    with db.tx(ctx.tenant_id) as c:
        w, hour = state(c)
        hospitals = open_hospitals(c)
    if not hour <= target_hour <= 12:
        raise Problem(422, "bad_target_hour", f"choose an hour from {hour} (now) to 12")
    p, wet = risk(w, hour, target_hour, hospitals)
    naive = engine.naive_risk(w, hour, target_hour)
    mean, sd = world.forecast(hour)
    cut = [n for n in range(world.N) if p[n] >= 0.5 and wet[n] < 0.5]
    top = sorted(range(world.N), key=lambda n: -p[n] * w["pop"][n])[:8]
    return {"made_at_hour": hour, "target_hour": target_hour, "samples": 300, "forecast": {"peak_surge_m": round(mean, 2), "plus_minus_m": round(sd, 2)}, "model_version": engine.VERSION,
            "p_isolated": [round(float(v), 3) for v in p], "p_flooded": [round(float(v), 3) for v in wet],
            "likely_isolated": int((p >= 0.5).sum()), "of_which_dry_but_cut_off": len(cut), "expected_isolated_population": int(round(float((p * np.array(w["pop"])).sum()))),
            "population_at_half_or_more": int(sum(w["pop"][n] for n in range(world.N) if p[n] >= 0.5)),
            "baseline": {"name": "survey elevation below the forecast water level", "flagged": int(naive.sum())},
            "top": [{"node": n, "x": n % world.W, "y": n // world.W, "p_isolated": round(float(p[n]), 3), "p_flooded": round(float(wet[n]), 3), "population": w["pop"][n]} for n in top]}


@app.get("/v1/risk/isolation/verification", tags=["risk"], summary="(+) Score an earlier forecast against which neighbourhoods are actually cut off now")
def verify_forecast(made_at_hour: int = 0, ctx: Ctx = Depends(auth("ops:read"))):
    with db.tx(ctx.tenant_id) as c:
        w, hour = state(c)
        actual = engine.isolated(w, closed_now(c, w), open_hospitals(c))
        silent = {r["node"] for r in c.execute("SELECT DISTINCT node FROM reports")}
    if not 0 <= made_at_hour < hour:
        raise Problem(422, "bad_made_at_hour", f"choose an hour before now ({hour})")
    p, _ = risk(w, made_at_hour, hour, [h["node"] for h in w["hospitals"]])
    wet = np.array(world.flooded(world.TRUE_PEAK, hour, w["elev_true"]))

    def table(pred):
        flag = pred >= 0.5
        return {"brier": round(float(((pred - actual) ** 2).mean()), 4), "hits": int((flag & actual).sum()), "misses": int((~flag & actual).sum()), "false_alarms": int((flag & ~actual).sum())}
    return {"made_at_hour": made_at_hour, "verified_at_hour": hour, "actually_isolated": int(actual.sum()), "of_which_dry_but_cut_off": int((actual & ~wet).sum()),
            "isolated_population": int(sum(w["pop"][n] for n in range(world.N) if actual[n])), "model": table(p), "baseline": {"name": "survey elevation below the forecast water level", **table(engine.naive_risk(w, made_at_hour, hour))},
            "isolated_and_silent": int(sum(1 for n in range(world.N) if actual[n] and n not in silent)), "actual": [bool(v) for v in actual]}


# ---------------------------------------------------------------- optimization-service

@app.post("/v1/optimize/dispatch", tags=["optimize"], summary="Assign every free or en-route unit to the open incidents, minimising severity-weighted response time over today's road network")
def optimize_dispatch(ctx: Ctx = Depends(auth("dispatch:optimize")), idem: str | None = IdemKey):
    def work(c):
        w, hour = state(c)
        closed = closed_now(c, w)
        inc = c.execute("SELECT id, type, severity, node, x, y, confidence, reports FROM incidents WHERE status IN ('open','dispatched') AND confidence >= %s ORDER BY reported_at, id FOR UPDATE", [engine.MIN_CONFIDENCE]).fetchall()
        held = c.execute("SELECT count(*) AS n FROM incidents WHERE status = 'open' AND confidence < %s", [engine.MIN_CONFIDENCE]).fetchone()["n"]
        units = c.execute("SELECT id, name, resource_type, node FROM resources WHERE status IN ('available','en_route') ORDER BY id FOR UPDATE").fetchall()
        before = {d["resource_id"]: d["incident_id"] for d in c.execute("UPDATE dispatches SET status = 'superseded' WHERE status = 'en_route' RETURNING resource_id, incident_id").fetchall()}
        c.execute("UPDATE incidents SET status = 'open' WHERE status = 'dispatched'")
        c.execute("UPDATE resources SET status = 'available' WHERE status = 'en_route'")
        eta = engine.etas(w, closed, [(u["node"], u["resource_type"]) for u in units], [(i["node"], i["type"]) for i in inc])
        sev = [i["severity"] for i in inc]
        pairs, base = engine.assign(sev, eta), engine.greedy(eta)
        road = dijkstra(engine.graph(w, closed), directed=False)
        beds = {h["id"]: h for h in c.execute("SELECT id, name, node, beds - occupied AS free FROM hospitals WHERE status = 'open'").fetchall()}
        plan, rows, no_hospital = uuid.uuid7(), [], 0
        for i in sorted(pairs, key=lambda i: -sev[i]):
            u, hosp = units[pairs[i]], None
            if inc[i]["type"] == "medical":
                hosp = min((h for h in beds.values() if h["free"] > 0 and np.isfinite(road[inc[i]["node"], h["node"]])), key=lambda h: road[inc[i]["node"], h["node"]], default=None)
                if hosp:
                    hosp["free"] -= 1
                else:
                    no_hospital += 1
            c.execute("INSERT INTO dispatches (id, tenant_id, incident_id, resource_id, eta_seconds, status, plan_id, hospital_id) VALUES (%s,%s,%s,%s,%s,'en_route',%s,%s)",
                      [uuid.uuid7(), ctx.tenant_id, inc[i]["id"], u["id"], int(eta[i, pairs[i]]), plan, hosp["id"] if hosp else None])
            rows.append({"incident_id": inc[i]["id"], "type": inc[i]["type"], "severity": sev[i], "x": inc[i]["x"], "y": inc[i]["y"], "unit_id": u["id"], "unit": u["name"], "unit_type": u["resource_type"],
                         "eta_min": round(float(eta[i, pairs[i]]) / 60, 1), "hospital": hosp["name"] if hosp else None, "changed": before.get(u["id"]) not in (None, inc[i]["id"])})
        c.execute("UPDATE incidents SET status = 'dispatched' WHERE id = ANY(%s)", [[r["incident_id"] for r in rows]])
        c.execute("UPDATE resources SET status = 'en_route' WHERE id = ANY(%s)", [[r["unit_id"] for r in rows]])
        best, naive = engine.score(sev, eta, pairs), engine.score(sev, eta, base)
        waiting = [{"incident_id": inc[i]["id"], "type": inc[i]["type"], "severity": sev[i], "x": inc[i]["x"], "y": inc[i]["y"], "reason": "no unit of the right kind can reach it" if not np.isfinite(eta[i]).any() else "all suitable units are on more severe calls"}
                   for i in range(len(inc)) if i not in pairs]
        out = {"plan_id": plan, "hour": hour, "incidents": len(inc), "held_for_verification": held, "units": len(units), "roads_closed": sum(closed), "assignments": rows, "waiting": waiting,
               "medical_without_reachable_hospital": no_hospital, "units_reassigned": sum(r["changed"] for r in rows), "units_kept": sum(before.get(r["unit_id"]) == r["incident_id"] for r in rows),
               "optimized": best, "baseline": {"name": "report order, nearest suitable unit", **naive},
               "weighted_wait_reduction": round(1 - best["weighted_wait"] / naive["weighted_wait"], 4) if naive["weighted_wait"] else 0, "solver": "exact assignment (scipy linear_sum_assignment)"}
        audit.record(c, ctx, "dispatch.optimized", "plan", plan, {"incidents": len(inc), "assigned": len(rows), "reassigned": out["units_reassigned"], "engine": engine.VERSION})
        return 200, out
    return run(ctx, idem, {}, work)


def evacuation(c, w, hour, min_probability, target_hour):
    p, _ = risk(w, hour, target_hour, open_hospitals(c))
    src = [n for n in range(world.N) if p[n] >= min_probability]
    shelters = c.execute("SELECT id, name, node, capacity, occupancy FROM shelters ORDER BY name").fetchall()
    dist, pred = dijkstra(engine.graph(w, closed_now(c, w)), directed=False, indices=src, return_predecessors=True) if src else (np.zeros((0, world.N)), None)
    minutes = dist[:, [s["node"] for s in shelters]] / 60
    supply = [round(w["pop"][n] * EVACUATING_SHARE) for n in src]
    flows, unmet = engine.allocate(supply, [s["capacity"] - s["occupancy"] for s in shelters], minutes)
    inflow = [sum(v for (_, j), v in flows.items() if j == k) for k in range(len(shelters))]
    routes = [{"from_node": src[i], "x": src[i] % world.W, "y": src[i] // world.W, "shelter": shelters[j]["name"], "people": v, "minutes": round(float(minutes[i, j]), 1),
               "path": _path(pred, i, src[i], shelters[j]["node"])} for (i, j), v in sorted(flows.items())]
    return {"made_at_hour": hour, "target_hour": target_hour, "min_probability": min_probability, "neighbourhoods": len(src), "population": sum(w["pop"][n] for n in src), "assumed_share_needing_shelter": EVACUATING_SHARE,
            "evacuees": sum(supply), "placed": sum(flows.values()), "unmet": sum(unmet), "person_minutes": round(float(sum(v * minutes[i, j] for (i, j), v in flows.items()))),
            "shelters": [{"name": s["name"], "capacity": s["capacity"], "occupancy": s["occupancy"], "inflow": inflow[k], "id": s["id"]} for k, s in enumerate(shelters)], "routes": routes, "nodes": src}


def _path(pred, row, src, dst):
    out = [dst]
    while out[-1] != src and pred[row, out[-1]] >= 0:
        out.append(int(pred[row, out[-1]]))
    return out[::-1]


@app.get("/v1/evacuation/routes", tags=["optimize"], summary="Which shelter each at-risk neighbourhood should go to, and by which roads, within shelter capacity")
def evacuation_routes(min_probability: float = 0.5, target_hour: int = world.PEAK_HOUR, ctx: Ctx = Depends(auth("ops:read"))):
    if not 0 < min_probability <= 1:
        raise Problem(422, "bad_probability", "min_probability is between 0 and 1")
    with db.tx(ctx.tenant_id) as c:
        w, hour = state(c)
        return jsonable_encoder(evacuation(c, w, hour, min_probability, max(hour, target_hour)))


# ---------------------------------------------------------------- notification-service

class AlertIn(BaseModel):
    message: str = Field(min_length=10, max_length=300)
    min_probability: float = Field(0.5, gt=0, le=1)
    target_hour: int = Field(world.PEAK_HOUR, ge=0, le=12)


@app.post("/v1/alerts/broadcast", status_code=201, tags=["alerts"], summary="Record an evacuation alert for the at-risk neighbourhoods and reserve their shelter places. Nothing is transmitted")
def broadcast(body: AlertIn, ctx: Ctx = Depends(auth("alerts:broadcast")), idem: str | None = IdemKey):
    def work(c):
        w, hour = state(c)
        plan = evacuation(c, w, hour, body.min_probability, max(hour, body.target_hour))
        for s in plan["shelters"]:
            c.execute("UPDATE shelters SET planned_inflow = %s WHERE id = %s", [s["inflow"], s["id"]])
        aid = uuid.uuid7()
        c.execute("INSERT INTO alerts (id, tenant_id, message, nodes, recipients, created_by) VALUES (%s,%s,%s,%s,%s,%s)", [aid, ctx.tenant_id, body.message, plan["nodes"], plan["population"], ctx.actor_id])
        audit.record(c, ctx, "alert.broadcast", "alert", aid, {"neighbourhoods": plan["neighbourhoods"], "recipients": plan["population"], "min_probability": body.min_probability})
        return 201, {"alert_id": aid, "neighbourhoods": plan["neighbourhoods"], "recipients": plan["population"], "shelter_places_reserved": plan["placed"], "without_a_place": plan["unmet"],
                     "delivery": "recorded only: this build has no SMS, cell-broadcast or siren integration"}
    return run(ctx, idem, body, work)


# ---------------------------------------------------------------- geospatial-service

@app.get("/v1/map", tags=["map"], summary="(+) Everything the command-centre map draws: roads and closures, shelters, hospitals, units and open incidents")
def county_map(ctx: Ctx = Depends(auth("ops:read"))):
    with db.tx(ctx.tenant_id) as c:
        w, hour = state(c)
        closed = closed_now(c, w)
        out = {"hour": hour, "w": world.W, "h": world.H, "river_after_column": world.RIVER, "elevation": w["elev"], "population": w["pop"],
               "edges": [[a, b, kind == "bridge", closed[i]] for i, (a, b, kind) in enumerate(w["edges"])],
               "shelters": c.execute("SELECT name, node, capacity, occupancy FROM shelters ORDER BY name").fetchall(),
               "hospitals": c.execute("SELECT name, node, beds, occupied, status FROM hospitals ORDER BY name").fetchall(),
               "units": c.execute("SELECT name, resource_type, status, node FROM resources ORDER BY name").fetchall(),
               "incidents": c.execute("SELECT type, severity, status, node, confidence FROM incidents WHERE status <> 'resolved' ORDER BY id").fetchall(),
               "resolved": c.execute("SELECT count(*) AS n FROM incidents WHERE status = 'resolved'").fetchone()["n"]}
    return jsonable_encoder(out)
