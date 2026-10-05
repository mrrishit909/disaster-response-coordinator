"""The models and solvers: report classifier, isolation-risk forecast, dispatch assignment and shelter allocation."""
import re

import numpy as np
from scipy.optimize import linear_sum_assignment, linprog
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from . import world

VERSION = "drc-1.0"
ROAD_S, BOAT_S = 90, 300                 # seconds per 1 km segment: by road, and by boat over a flooded one
CAN_SERVE = {"water_rescue": {"boat"}, "medical": {"ambulance", "boat"}, "fire": {"engine"}, "utility": {"engine"}}
UNSERVED_S = 7200                        # an incident left without a unit is costed as a two-hour wait
MERGE_BLOCKS = 0                         # a report joins an open incident of the same type this many blocks away (see docs/evaluation.md)
MIN_CONFIDENCE = 0.5                     # below this an incident waits for a second report or a person
KEYWORDS = [("fire", r"fire|smoke|flames|burning"), ("water_rescue", r"flood|boat|roof|trapped|stranded|under water"),
            ("medical", r"pain|breathe|bleeding|ambulance|insulin|oxygen|seizure|labor|unconscious|sick|dizzy"), ("utility", r"line|pole|tree|traffic|main|drain")]


# ---------------------------------------------------------------- citizen reports

def train_classifier(pairs):
    return make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)), LogisticRegression(max_iter=1000, C=5)).fit([t for t, _ in pairs], [k for _, k in pairs])


def keyword_type(text):
    return next((k for k, pat in KEYWORDS if re.search(pat, text)), "noise")


def corroborate(confidence, credibility):
    """Two independent reports of the same thing: the chance that both are wrong is the product."""
    return 1 - (1 - confidence) * (1 - credibility)


# ---------------------------------------------------------------- road network

def graph(w, closed, boat=False):
    a, b, t = [], [], []
    for (u, v, _), c in zip(w["edges"], closed):
        if c and not boat:
            continue
        a += [u, v]
        b += [v, u]
        t += [BOAT_S if c else ROAD_S] * 2
    return csr_matrix((t, (a, b)), shape=(world.N, world.N))


def isolated(w, closed, hospitals):
    """True for each neighbourhood with no open road to any working hospital."""
    _, comp = connected_components(graph(w, closed), directed=False)
    ok = {comp[h] for h in hospitals}
    return np.array([c not in ok for c in comp])


def isolation_risk(w, hour_made, target_hour, hospitals, n=300, seed=0):
    """Monte Carlo over what is uncertain before the storm: the surge peak and the elevation survey.
    -> (P(isolated), P(under water)) per neighbourhood at `target_hour`."""
    rng = np.random.default_rng([seed, hour_made, target_hour])
    mean, sd = world.forecast(hour_made)
    elev, iso, wet = np.array(w["elev"]), np.zeros(world.N), np.zeros(world.N)
    for _ in range(n):
        peak, e = rng.normal(mean, sd), elev + rng.normal(0, world.SURVEY_ERROR, world.N)
        iso += isolated(w, world.closed(w, peak, target_hour, e), hospitals)
        wet += world.flooded(peak, target_hour, e)
    return iso / n, wet / n


def naive_risk(w, hour_made, target_hour):
    """Baseline: a neighbourhood is at risk if the survey puts it below the forecast water level."""
    return np.array(world.flooded(world.forecast(hour_made)[0], target_hour, w["elev"]), dtype=float)


# ---------------------------------------------------------------- dispatch

def etas(w, closed, units, incidents):
    """Seconds for each unit to reach each incident; inf where the unit type cannot serve it or cannot get there.
    units: [(node, type)], incidents: [(node, type)] -> array [incident, unit]"""
    road = dijkstra(graph(w, closed), directed=False)
    water = dijkstra(graph(w, closed, boat=True), directed=False)
    out = np.full((len(incidents), len(units)), np.inf)
    for i, (n, kind) in enumerate(incidents):
        for u, (m, t) in enumerate(units):
            if t in CAN_SERVE[kind]:
                out[i, u] = (water if t == "boat" else road)[m, n]
    return out


def weight(severity):
    return np.asarray(severity, dtype=float) ** 2


def assign(severity, eta):
    """Minimise severity-weighted response time over all incidents at once (exact assignment).
    Each incident also has a private 'nobody comes' column, so with too few units the least severe wait."""
    n, m = eta.shape
    if not n:
        return {}
    wt = weight(severity)
    cost = np.full((n, m + n), 1e12)
    cost[:, :m] = np.where(np.isfinite(eta), wt[:, None] * np.where(np.isfinite(eta), eta, 0), 1e12)
    cost[np.arange(n), m + np.arange(n)] = wt * UNSERVED_S
    rows, cols = linear_sum_assignment(cost)
    return {int(i): int(u) for i, u in zip(rows, cols) if u < m}


def greedy(eta):
    """Baseline: incidents in the order they were reported, each given the nearest free unit that can serve it."""
    free, out = set(range(eta.shape[1])), {}
    for i in range(eta.shape[0]):
        best = min(free, key=lambda u: eta[i, u], default=None)
        if best is not None and np.isfinite(eta[i, best]):
            out[i] = best
            free.remove(best)
    return out


def score(severity, eta, pairs):
    wt = weight(severity)
    got = [i for i in pairs]
    severe = [i for i, s in enumerate(severity) if s >= 4]
    served = [eta[i, pairs[i]] for i in severe if i in pairs]
    return {"weighted_wait": float(sum(wt[i] * eta[i, pairs[i]] for i in got) + sum(wt[i] * UNSERVED_S for i in range(len(wt)) if i not in pairs)),
            "assigned": len(got), "severe": len(severe), "severe_unserved": len(severe) - len(served), "severe_mean_eta_s": round(float(np.mean(served)), 0) if served else None}


# ---------------------------------------------------------------- shelters

def allocate(supply, free, minutes):
    """Send `supply[i]` evacuees from each origin to shelters with `free[j]` places, minimising person-minutes travelled.
    A transportation problem (LP; whole-number supplies give a whole-number answer). Unreachable shelters are excluded;
    people who fit nowhere are returned as unmet. -> (flows {(i, j): people}, unmet per origin)"""
    n, m = len(supply), len(free)
    if not n or not m:
        return {}, list(supply)
    big = 4 * 60.0
    c = np.concatenate([np.where(np.isfinite(minutes), minutes, 1e6).ravel(), np.full(n, big)])
    a_eq = np.zeros((n, n * m + n))
    for i in range(n):
        a_eq[i, i * m:(i + 1) * m] = 1
        a_eq[i, n * m + i] = 1
    a_ub = np.zeros((m, n * m + n))
    for j in range(m):
        a_ub[j, j:n * m:m] = 1
    bounds = [(0, 0 if not np.isfinite(minutes[i, j]) else None) for i in range(n) for j in range(m)] + [(0, None)] * n
    res = linprog(c, A_ub=a_ub, b_ub=free, A_eq=a_eq, b_eq=supply, bounds=bounds, method="highs")
    x = np.rint(res.x).astype(int)
    return {(i, j): int(x[i * m + j]) for i in range(n) for j in range(m) if x[i * m + j] > 0}, [int(v) for v in x[n * m:]]

