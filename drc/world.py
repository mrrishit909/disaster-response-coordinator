"""A deterministic synthetic coastal county and one hurricane. Nothing here is real: the map, the people, the storm
and every citizen report are generated from a seed, so each engine can be checked against a known answer."""
import functools
import math
import random

W, H, RIVER = 14, 10, 6                      # a 14 x 10 km grid of neighbourhoods; the river runs between columns 6 and 7
N = W * H
BRIDGES = {1: 1.4, 4: 1.1, 6: 2.2, 8: 2.6}   # row -> river level (m) at which the bridge closes
FORECAST_PEAK, FORECAST_SD, PEAK_HOUR, TRUE_PEAK = 4.5, 0.7, 6, 4.9
SURVEY_ERROR = 0.25                          # the elevation map is off by about this much (m)
FLOOD_DEPTH = 0.3                            # a road is impassable once either end is under this much water
LAT0, LON0, DLAT, DLON = 27.70, -82.64, 0.009, 0.0102

PHRASES = {
    "water_rescue": ["water is up to the windows and we are on the roof", "family trapped in the attic water rising fast", "car stuck in flood water with people inside",
                     "we need a boat the street is a river", "neighbors stranded on the second floor by flooding", "water coming in the house and we cannot get out",
                     "elderly couple trapped by rising water", "kids and a dog on the roof flood water everywhere", "the whole block is under water people waving from windows"],
    "medical": ["my father is having chest pain and cannot breathe", "woman in labor contractions two minutes apart", "grandma needs her oxygen the tank is almost empty",
                "man unconscious and not responding after a fall", "diabetic out of insulin and feeling very weak", "deep cut on his leg the bleeding will not stop",
                "child having a seizure we need an ambulance", "dialysis patient missed treatment and is very sick", "water got in downstairs and my mother dropped her heart pills she is dizzy"],
    "fire": ["house on fire flames coming through the roof", "transformer exploded and the garage is burning", "smell of gas and smoke from the kitchen",
             "apartment fire and people are still inside", "generator caught fire next to the house", "smoke pouring out of the store on the corner"],
    "utility": ["power line down across the road and sparking", "tree fell on the lines and the street is blocked", "traffic light is out at the intersection",
                "water main broke and the road is washing out", "pole leaning over the sidewalk with wires hanging low", "storm drain blocked water pooling on the road"],
    "noise": ["is the power out for anyone else", "stay safe everyone praying for the coast", "does anyone know if the grocery store is open", "the wind is really loud tonight",
              "when will the bridges reopen", "is school closed tomorrow", "my cable is out again", "thanks to all the first responders", "watching the water from my porch wow",
              "anyone else smell rain lol", "the fire station on main looks busy", "heard a boat went by"]}
OPENERS = ["", "", "please help ", "urgent ", "help ", "hello ", "emergency ", "pls send someone "]
CLOSERS = ["", "", " please hurry", " near the church", " on bay street", " by the school", " corner of 4th", " we are scared", " thank you"]
SEVERITY = {"water_rescue": (3, 5), "medical": (3, 5), "fire": (4, 5), "utility": (1, 2)}
DEFAULT_SEVERITY = {"water_rescue": 4, "medical": 4, "fire": 4, "utility": 2}
SOURCES = [("911", 0.95), ("app", 0.7), ("social", 0.4)]


def node(x, y):
    return y * W + x


def level(peak, hour, y):
    """Storm-surge water level (m): rises to the peak at landfall + 2 h, and is lower further from the coast (row 0)."""
    return peak * math.exp(-((hour - PEAK_HOUR) / 2.2) ** 2) * math.exp(-y / 9)


def forecast(hour):
    """The surge forecast as it stands at `hour`: it narrows and moves toward the truth as landfall nears."""
    k = min(1.0, hour / PEAK_HOUR)
    return FORECAST_PEAK + (TRUE_PEAK - FORECAST_PEAK) * k, FORECAST_SD * max(0.15, 1 - k)


def closed(w, peak, hour, elev, debris=()):
    """Which road segments are impassable. `elev` is the elevation list to believe (survey, sampled or true)."""
    out = []
    for i, (a, b, kind) in enumerate(w["edges"]):
        ya, yb = a // W, b // W
        if kind == "bridge":
            out.append(level(peak, hour, ya) > BRIDGES[ya])
        else:
            out.append(max(level(peak, hour, ya) - elev[a], level(peak, hour, yb) - elev[b]) > FLOOD_DEPTH or i in debris)
    return out


def flooded(peak, hour, elev):
    return [level(peak, hour, i // W) - e > FLOOD_DEPTH for i, e in enumerate(elev)]


def text(r, kind):
    s = r.choice(OPENERS) + r.choice(PHRASES[kind]) + r.choice(CLOSERS)
    if r.random() < 0.25:                                    # typed in a hurry
        i = r.randrange(len(s))
        s = s[:i] + s[i + 1:]
    return s


def corpus(seed, n=3000):
    """Labelled reports for fitting and scoring the classifier (a different seed from any scenario)."""
    r = random.Random(seed)
    kinds = list(PHRASES)
    return [(text(r, k), k) for k in (r.choice(kinds) for _ in range(n))]


@functools.lru_cache(maxsize=8)
def build(seed=7):
    r = random.Random(seed)
    xy = [(x, y) for y in range(H) for x in range(W)]
    hills = [(r.randrange(W), r.randrange(0, 4)) for _ in range(5)]
    elev = [round(max(0.2, 0.8 * y + 0.5 * abs(x - 6.5) + r.uniform(-0.5, 0.5) + sum(2.4 * math.exp(-((x - hx) ** 2 + (y - hy) ** 2) / 1.2) for hx, hy in hills)), 2) for x, y in xy]
    elev_true = [round(e + r.gauss(0, SURVEY_ERROR), 2) for e in elev]
    edges = []
    for x, y in xy:
        if x + 1 < W and (x != RIVER or y in BRIDGES):
            edges.append((node(x, y), node(x + 1, y), "bridge" if x == RIVER else "road"))
        if y + 1 < H:
            edges.append((node(x, y), node(x, y + 1), "road"))
    w = {"seed": seed, "xy": xy, "elev": elev, "elev_true": elev_true, "edges": edges, "pop": [r.randrange(600, 3000) for _ in xy]}
    high = lambda side: [n for n, (x, y) in enumerate(xy) if y >= 5 and (x <= RIVER) == side]      # noqa: E731
    w["shelters"] = [{"name": f"{'West' if s else 'East'} shelter {i + 1}", "node": n, "capacity": r.choice([300, 450, 600, 800])}
                     for s in (True, False) for i, n in enumerate(r.sample(high(s), 4))]
    w["hospitals"] = [{"name": "West General", "node": node(3, 7), "beds": 50}, {"name": "Eastside Medical", "node": node(10, 5), "beds": 60}, {"name": "North County Hospital", "node": node(9, 8), "beds": 80}]
    stations = [node(2, 5), node(4, 8), node(9, 4), node(11, 7)]
    w["units"] = [{"name": f"{t.title()} {i + 1}", "type": t, "node": stations[i % 4]} for t, k in (("boat", 6), ("ambulance", 10), ("engine", 8)) for i in range(k)]
    w["debris"] = frozenset(r.sample([i for i, e in enumerate(edges) if e[2] == "road" and e[0] // W >= 3], 4))     # closures no flood model predicts
    w["incidents"], w["reports"], recent = [], [], {}
    for hour, count in ((3, 6), (4, 16), (5, 22), (6, 28), (7, 18), (8, 10)):
        wet = [n for n, f in enumerate(flooded(TRUE_PEAK, hour, elev_true)) if f]
        dry = [n for n in range(N) if n not in set(wet)]
        for _ in range(count):
            kind = "water_rescue" if wet and r.random() < 0.6 else r.choices(["medical", "fire", "utility"], [5, 2, 3])[0]
            free = [n for n in (wet if kind == "water_rescue" else dry) if hour - recent.get((kind, n), -9) > 2]      # one emergency of a kind per block at a time
            if not free:
                continue
            inc = {"hour": hour, "type": kind, "node": r.choice(free), "severity": r.randint(*SEVERITY[kind])}
            recent[kind, inc["node"]] = hour
            w["incidents"].append(inc)
            for _ in range(r.choice([1, 1, 2, 2, 3, 4])):
                src = r.choices(SOURCES, [3, 4, 3])[0][0]
                x, y = xy[inc["node"]]
                if r.random() < 0.2:                             # located one block off
                    x = min(W - 1, max(0, x + r.choice([-1, 1])))
                w["reports"].append({"hour": hour, "text": text(r, kind), "source": src, "node": node(x, y), "incident": len(w["incidents"]) - 1,
                                     "severity_hint": inc["severity"] if src == "911" else None})
        for _ in range(count // 3 + 1):
            w["reports"].append({"hour": hour, "text": text(r, "noise"), "source": r.choice(["app", "social"]), "node": r.randrange(N), "incident": None, "severity_hint": None})
    return w
