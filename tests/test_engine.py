"""Unit and property tests for the solvers. No database."""
import random

import numpy as np
import pytest

from drc import engine, world

W = world.build(7)
HOSP = [h["node"] for h in W["hospitals"]]


def test_higher_surge_never_reopens_a_road():
    low, high = world.closed(W, 4.0, 6, W["elev"]), world.closed(W, 5.5, 6, W["elev"])
    assert all(h or not lo for lo, h in zip(low, high)) and sum(high) > sum(low) > 0
    assert not any(world.closed(W, 4.9, 0, W["elev"]))                        # calm six hours out


def test_isolation_risk_is_a_probability_and_covers_flooded_blocks():
    p, wet = engine.isolation_risk(W, 0, 6, HOSP, n=60)
    assert ((p >= 0) & (p <= 1)).all() and (p >= wet - 1e-9).all()             # a block under water is always cut off
    assert p[world.node(7, 0)] > 0.9 and p[world.node(3, 9)] < 0.05            # the river mouth against high ground inland
    p_all_shut, _ = engine.isolation_risk(W, 0, 6, [], n=5)
    assert (p_all_shut == 1).all()                                             # with no hospital working, everywhere counts as cut off


def test_only_boats_reach_flooded_blocks_and_types_are_respected():
    closed = world.closed(W, world.TRUE_PEAK, 6, W["elev_true"])
    wet = next(n for n, f in enumerate(world.flooded(world.TRUE_PEAK, 6, W["elev_true"])) if f)
    eta = engine.etas(W, closed, [(world.node(9, 8), "boat"), (world.node(9, 8), "ambulance"), (world.node(9, 8), "engine")], [(wet, "water_rescue"), (wet, "medical"), (world.node(9, 9), "fire")])
    assert np.isfinite(eta[0, 0]) and np.isinf(eta[0, 1]) and np.isinf(eta[0, 2])
    assert np.isfinite(eta[1, 0]) and np.isinf(eta[1, 1])                      # ambulance may serve medical but cannot drive there
    assert np.isinf(eta[2, 0]) and np.isinf(eta[2, 1]) and eta[2, 2] == engine.ROAD_S


@pytest.mark.parametrize("seed", range(40))
def test_assignment_never_loses_to_greedy_and_uses_each_unit_once(seed):
    r = random.Random(seed)
    n, m = r.randint(1, 30), r.randint(1, 20)
    eta = np.array([[r.choice([np.inf, r.uniform(60, 3000)]) for _ in range(m)] for _ in range(n)])
    sev = [r.randint(1, 5) for _ in range(n)]
    best, base = engine.assign(sev, eta), engine.greedy(eta)
    assert len(set(best.values())) == len(best) and all(np.isfinite(eta[i, u]) for i, u in best.items())
    assert engine.score(sev, eta, best)["weighted_wait"] <= engine.score(sev, eta, base)["weighted_wait"] + 1e-6


def test_with_one_unit_the_most_severe_incident_gets_it():
    eta = np.array([[600.0], [600.0], [600.0]])
    assert engine.assign([2, 5, 3], eta) == {1: 0} and engine.greedy(eta) == {0: 0}
    assert engine.assign([], np.zeros((0, 3))) == {}


@pytest.mark.parametrize("seed", range(20))
def test_shelter_allocation_respects_capacity_and_loses_nobody(seed):
    r = random.Random(seed)
    n, m = r.randint(1, 12), r.randint(1, 6)
    supply, free = [r.randint(0, 400) for _ in range(n)], [r.randint(0, 500) for _ in range(m)]
    minutes = np.array([[r.choice([np.inf, r.uniform(2, 40)]) for _ in range(m)] for _ in range(n)])
    flows, unmet = engine.allocate(supply, free, minutes)
    for i in range(n):
        assert sum(v for (a, _), v in flows.items() if a == i) + unmet[i] == supply[i]
    for j in range(m):
        assert sum(v for (_, b), v in flows.items() if b == j) <= free[j]
    assert all(np.isfinite(minutes[i, j]) for i, j in flows)
    reachable = sum(min(supply[i], sum(free[j] for j in range(m) if np.isfinite(minutes[i, j]))) for i in range(n))
    assert sum(flows.values()) <= min(sum(supply), sum(free), reachable)


def test_allocation_prefers_the_nearer_shelter_until_it_is_full():
    flows, unmet = engine.allocate([100], [60, 100], np.array([[5.0, 30.0]]))
    assert flows == {(0, 0): 60, (0, 1): 40} and unmet == [0]


def test_corroboration_raises_confidence_without_reaching_certainty():
    c = engine.corroborate(0.4, 0.4)
    assert 0.4 < c < 1 and c == pytest.approx(0.64)


def test_keyword_baseline_and_text_generator():
    assert engine.keyword_type("house on fire flames coming through the roof") == "fire" and engine.keyword_type("is school closed tomorrow") == "noise"
    assert engine.keyword_type("water main broke and the road is washing out") == "utility"
    assert world.corpus(5, 50) == world.corpus(5, 50)                          # deterministic
