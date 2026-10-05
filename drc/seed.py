"""Tenants, demo tokens and the report classifier (fitted on one synthetic corpus, scored on another, then registered).
The county itself is loaded per tenant by POST /v1/scenarios/hurricane.

    python -m drc.seed [--if-empty]
"""
import hashlib
import pickle
import sys
import uuid

from psycopg.types.json import Jsonb

from core import db

from . import engine, world

TENANTS = {"gulf": ("Gulf County Emergency Operations", 7), "other": ("Second County EOC", 23)}       # name, world seed
ROLES = ["viewer", "dispatcher", "commander"]


def token(tenant, role):
    """Demo credentials for local use only."""
    return f"{tenant}-{role}-demo"


def tenant_id(key):
    return uuid.uuid5(uuid.NAMESPACE_DNS, f"{key}.disaster-response.example")


def train():
    clf = engine.train_classifier(world.corpus(101))
    test = world.corpus(202)
    acc = sum(p == k for p, (_, k) in zip(clf.predict([t for t, _ in test]), test)) / len(test)
    base = sum(engine.keyword_type(t) == k for t, k in test) / len(test)
    return clf, {"model": {"accuracy": round(acc, 4), "n_test": len(test)}, "baseline": {"name": "keyword rules", "accuracy": round(base, 4)}}


def main(if_empty=False):
    db.migrate()
    with db.tx() as c:
        if if_empty and c.execute("SELECT 1 FROM tenants LIMIT 1").fetchone():
            return print("already seeded")
    clf, metrics = train()
    out = {"metrics": metrics}
    with db.tx() as c:
        c.execute("INSERT INTO model_registry (name, version, data_snapshot, metrics, baseline, artifact) VALUES ('report-classifier',%s,%s,%s,%s,%s)",
                  [engine.VERSION, "synthetic report corpora, seeds 101 (fit) and 202 (score)", Jsonb(metrics["model"]), Jsonb(metrics["baseline"]), pickle.dumps(clf)])
        for key, (name, _) in TENANTS.items():
            t = tenant_id(key)
            c.execute("INSERT INTO tenants (id, name) VALUES (%s,%s)", [t, name])
            for role in ROLES:
                c.execute("INSERT INTO api_tokens VALUES (%s,%s,%s,%s,%s)", [hashlib.sha256(token(key, role).encode()).hexdigest(), t, uuid.uuid5(t, role), f"{role} ({name})", role])
            out[key] = t
    print("classifier registered:", metrics)
    print("demo tokens:", ", ".join(token(t, r) for t in TENANTS for r in ROLES))
    return out


if __name__ == "__main__":
    main("--if-empty" in sys.argv)
