export DATABASE_URL ?= postgresql://app:app-dev-only@127.0.0.1:55201/app
PY = ./venv/bin/python
BASE = http://127.0.0.1:8201

bootstrap:      ## local virtualenv + database container
	python3 -m venv venv && ./venv/bin/pip install -q -r requirements-dev.txt
	docker compose up -d db
seed:           ## migrate and load the synthetic world (no-op if already seeded)
	$(PY) -m drc.seed --if-empty
dev:            ## API with reload on :8201, jobs run by a local worker
	($(PY) -m core.jobs drc.api &) && $(PY) -m uvicorn drc.api:app --reload --port 8201
test:
	$(PY) -m pytest --cov=drc --cov=core --cov-report=term-missing
demo:           ## the full stack in Docker, then the demo scenario against it
	docker compose up --build -d --wait && $(PY) -m core.scenario $(BASE)
load-test:      ## run `make demo` first
	$(PY) -m core.loadtest $(BASE) gulf-viewer-demo 32 10 "GET /v1/map" "GET /v1/incidents?severity=3&limit=50" "GET /v1/risk/isolation" "GET /v1/evacuation/routes" 
reset:          ## drop all data (volume included)
	docker compose down -v
.PHONY: bootstrap seed dev test demo load-test reset
