-- Domain model. Names follow the blueprint's schema; additions are marked (+).
-- The blueprint uses PostGIS GEOGRAPHY columns. This build stores plain latitude / longitude (and the grid cell the
-- point falls in), because the synthetic county is a grid and every spatial question here is a road-network question.

CREATE TABLE sim_clock (tenant_id uuid PRIMARY KEY REFERENCES tenants, hour int NOT NULL, seed int NOT NULL);                 -- (+) where the scenario is
SELECT enable_tenant_rls('sim_clock');

CREATE TABLE incidents (
  id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, type text NOT NULL,
  severity smallint NOT NULL CHECK (severity BETWEEN 1 AND 5), status text NOT NULL CHECK (status IN ('open','dispatched','resolved')),
  lat double precision NOT NULL, lon double precision NOT NULL, source text, reported_at timestamptz NOT NULL DEFAULT now(),
  confidence numeric(5,4) NOT NULL, metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  node int NOT NULL, x int NOT NULL, y int NOT NULL, hour int NOT NULL, reports int NOT NULL DEFAULT 1);                       -- (+)
CREATE INDEX incidents_open ON incidents (tenant_id, status, type);
CREATE INDEX incidents_where ON incidents (tenant_id, lat, lon);
SELECT enable_tenant_rls('incidents');

CREATE TABLE reports (                                                                                                          -- (+) every citizen report, kept even when filtered out
  id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, incident_id uuid REFERENCES incidents ON DELETE SET NULL,
  body text NOT NULL, source text NOT NULL, node int NOT NULL, hour int NOT NULL, predicted_type text NOT NULL, credibility numeric(5,4) NOT NULL,
  received_at timestamptz NOT NULL DEFAULT now());
CREATE INDEX reports_incident ON reports (incident_id);
SELECT enable_tenant_rls('reports');

CREATE TABLE resources (
  id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, resource_type text NOT NULL,
  status text NOT NULL CHECK (status IN ('available','en_route','out_of_service')), capacity jsonb NOT NULL DEFAULT '{}'::jsonb,
  lat double precision NOT NULL, lon double precision NOT NULL, last_seen_at timestamptz,
  name text NOT NULL, node int NOT NULL);                                                                                       -- (+)
SELECT enable_tenant_rls('resources');

CREATE TABLE shelters (id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, name text NOT NULL, capacity int NOT NULL, occupancy int NOT NULL DEFAULT 0,
  lat double precision NOT NULL, lon double precision NOT NULL, node int NOT NULL, planned_inflow int NOT NULL DEFAULT 0,      -- (+)
  CHECK (occupancy BETWEEN 0 AND capacity));
SELECT enable_tenant_rls('shelters');

CREATE TABLE hospitals (id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, name text NOT NULL, beds int NOT NULL, occupied int NOT NULL,   -- (+)
  status text NOT NULL CHECK (status IN ('open','diverting')), node int NOT NULL, CHECK (occupied BETWEEN 0 AND beds));
SELECT enable_tenant_rls('hospitals');

CREATE TABLE road_events (id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, road_segment_id text NOT NULL, event_type text NOT NULL, severity int NOT NULL,
  starts_at timestamptz NOT NULL DEFAULT now(), ends_at timestamptz, hour int NOT NULL);
CREATE UNIQUE INDEX road_events_active ON road_events (tenant_id, road_segment_id) WHERE ends_at IS NULL;
SELECT enable_tenant_rls('road_events');

CREATE TABLE dispatches (id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, incident_id uuid NOT NULL REFERENCES incidents, resource_id uuid NOT NULL REFERENCES resources,
  eta_seconds int NOT NULL, status text NOT NULL CHECK (status IN ('en_route','arrived','superseded')), created_at timestamptz NOT NULL DEFAULT now(),
  plan_id uuid NOT NULL, hospital_id uuid REFERENCES hospitals);                                                                 -- (+)
CREATE INDEX dispatches_incident ON dispatches (incident_id);
CREATE INDEX dispatches_resource ON dispatches (resource_id);
CREATE UNIQUE INDEX dispatches_one_job_per_unit ON dispatches (resource_id) WHERE status = 'en_route';
SELECT enable_tenant_rls('dispatches');

CREATE TABLE alerts (id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, message text NOT NULL, nodes int[] NOT NULL, recipients int NOT NULL,          -- (+)
  created_by uuid NOT NULL, created_at timestamptz NOT NULL DEFAULT now());
SELECT enable_tenant_rls('alerts');
