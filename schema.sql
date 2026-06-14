-- ============================================================
-- NafidhAI — Agent Runtime Database Schema
-- PostgreSQL 15+
-- All tables scoped to tenant_id (row-level isolation)
-- audit_logs is append-only: no UPDATE, no DELETE
-- ============================================================

-- Enable pgvector for future embedding support
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ─────────────────────────────────────────────
-- TENANTS (reference table)
-- ─────────────────────────────────────────────
CREATE TABLE tenants (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name            TEXT NOT NULL,
    name_ar         TEXT,                          -- Arabic display name
    tier            TEXT NOT NULL CHECK (tier IN ('starter', 'professional', 'enterprise')),
    region          TEXT NOT NULL CHECK (region IN ('ksa', 'uae', 'other')),
    is_active       BOOLEAN NOT NULL DEFAULT TRUE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ─────────────────────────────────────────────
-- AGENT DEFINITIONS
-- Stores the serialized LangGraph DAG definition
-- ─────────────────────────────────────────────
CREATE TABLE agent_definitions (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id           UUID NOT NULL REFERENCES tenants(id) ON DELETE RESTRICT,
    name                TEXT NOT NULL,
    name_ar             TEXT,
    description         TEXT,
    description_ar      TEXT,
    version             INTEGER NOT NULL DEFAULT 1,
    is_active           BOOLEAN NOT NULL DEFAULT TRUE,

    -- JSON definition of the LangGraph DAG
    -- Structure: { nodes: [...], edges: [...], entry_point: str, config: {...} }
    graph_definition    JSONB NOT NULL,

    -- LLM config for this agent (model, temperature, max_tokens, etc.)
    llm_config          JSONB NOT NULL DEFAULT '{}',

    -- Connector IDs this agent is authorized to use
    allowed_connector_ids UUID[] NOT NULL DEFAULT '{}',

    -- Execution limits
    max_steps           INTEGER NOT NULL DEFAULT 50,
    timeout_seconds     INTEGER NOT NULL DEFAULT 300,

    created_by          UUID NOT NULL,               -- user_id
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Enforce tenant isolation at DB level
    CONSTRAINT uq_agent_name_per_tenant UNIQUE (tenant_id, name, version)
);

CREATE INDEX idx_agent_definitions_tenant ON agent_definitions(tenant_id);
CREATE INDEX idx_agent_definitions_active ON agent_definitions(tenant_id, is_active);

-- ─────────────────────────────────────────────
-- AGENT RUNS
-- One record per agent execution
-- ─────────────────────────────────────────────
CREATE TYPE agent_run_status AS ENUM (
    'queued', 'running', 'completed', 'failed', 'timeout', 'cancelled'
);

CREATE TABLE agent_runs (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id           UUID NOT NULL REFERENCES tenants(id) ON DELETE RESTRICT,
    agent_definition_id UUID NOT NULL REFERENCES agent_definitions(id) ON DELETE RESTRICT,

    status              agent_run_status NOT NULL DEFAULT 'queued',

    -- Sanitized input — PII fields masked before storage
    input_payload       JSONB NOT NULL,

    -- Final output from the agent DAG
    output_payload      JSONB,

    -- Error details if status = failed
    error_code          TEXT,
    error_message       TEXT,
    error_node          TEXT,                        -- which node failed

    -- Trigger context
    trigger_type        TEXT NOT NULL CHECK (
                            trigger_type IN ('api', 'webhook', 'schedule', 'manual', 'agent_chain')
                        ),
    triggered_by        UUID,                        -- user_id or NULL for system triggers
    triggered_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Timing
    started_at          TIMESTAMPTZ,
    completed_at        TIMESTAMPTZ,
    duration_ms         INTEGER GENERATED ALWAYS AS (
                            EXTRACT(MILLISECONDS FROM (completed_at - started_at))::INTEGER
                        ) STORED,

    -- LLM usage for cost tracking
    total_tokens_used   INTEGER NOT NULL DEFAULT 0,
    total_llm_calls     INTEGER NOT NULL DEFAULT 0,

    -- Correlation
    correlation_id      UUID NOT NULL DEFAULT uuid_generate_v4(),
    parent_run_id       UUID REFERENCES agent_runs(id),   -- for chained agents

    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_agent_runs_tenant ON agent_runs(tenant_id);
CREATE INDEX idx_agent_runs_definition ON agent_runs(tenant_id, agent_definition_id);
CREATE INDEX idx_agent_runs_status ON agent_runs(tenant_id, status);
CREATE INDEX idx_agent_runs_triggered_at ON agent_runs(tenant_id, triggered_at DESC);
CREATE INDEX idx_agent_runs_correlation ON agent_runs(correlation_id);

-- ─────────────────────────────────────────────
-- AGENT STEPS
-- One record per node execution within a run
-- ─────────────────────────────────────────────
CREATE TYPE step_status AS ENUM ('running', 'completed', 'failed', 'skipped', 'timeout');

CREATE TABLE agent_steps (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id           UUID NOT NULL REFERENCES tenants(id) ON DELETE RESTRICT,
    agent_run_id        UUID NOT NULL REFERENCES agent_runs(id) ON DELETE CASCADE,

    step_index          INTEGER NOT NULL,            -- execution order within run
    node_name           TEXT NOT NULL,               -- LangGraph node identifier
    node_type           TEXT NOT NULL CHECK (
                            node_type IN ('llm', 'tool', 'condition', 'connector', 'human_in_loop')
                        ),

    status              step_status NOT NULL DEFAULT 'running',

    -- Input/output for this specific node
    input_state         JSONB,
    output_state        JSONB,

    -- For LLM nodes: which model was called, tokens used
    llm_model           TEXT,
    prompt_tokens       INTEGER,
    completion_tokens   INTEGER,
    llm_latency_ms      INTEGER,

    -- For connector nodes: which connector, response code
    connector_id        UUID,
    connector_response_code INTEGER,

    -- Retry tracking
    attempt_number      INTEGER NOT NULL DEFAULT 1,
    max_attempts        INTEGER NOT NULL DEFAULT 3,

    error_code          TEXT,
    error_message       TEXT,

    started_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at        TIMESTAMPTZ,
    duration_ms         INTEGER GENERATED ALWAYS AS (
                            EXTRACT(MILLISECONDS FROM (completed_at - started_at))::INTEGER
                        ) STORED,

    CONSTRAINT uq_step_per_run UNIQUE (agent_run_id, step_index, attempt_number)
);

CREATE INDEX idx_agent_steps_tenant ON agent_steps(tenant_id);
CREATE INDEX idx_agent_steps_run ON agent_steps(agent_run_id);
CREATE INDEX idx_agent_steps_node ON agent_steps(tenant_id, node_name);

-- ─────────────────────────────────────────────
-- AUDIT LOGS
-- Immutable append-only log
-- No UPDATE or DELETE ever issued against this table
-- Enforced at: application layer + DB role permissions
-- ─────────────────────────────────────────────
CREATE TABLE audit_logs (
    id              BIGSERIAL PRIMARY KEY,           -- sequential for ordering guarantees
    tenant_id       UUID NOT NULL,                   -- denormalized for query performance
    user_id         UUID,                            -- NULL for system actions
    session_id      UUID,

    -- What happened
    action          TEXT NOT NULL,                   -- e.g. AGENT_RUN_STARTED, LLM_CALLED
    resource_type   TEXT NOT NULL,                   -- e.g. agent_run, agent_definition
    resource_id     UUID,

    -- Correlation
    agent_run_id    UUID,
    correlation_id  UUID,

    -- Result
    result          TEXT NOT NULL CHECK (result IN ('success', 'failure', 'partial')),
    error_code      TEXT,

    -- Context
    ip_address      INET,
    user_agent      TEXT,
    request_id      UUID,

    -- Payload: structured details of the action (PII masked)
    metadata        JSONB NOT NULL DEFAULT '{}',

    -- Compliance fields
    data_classification TEXT NOT NULL DEFAULT 'internal'
                        CHECK (data_classification IN ('public', 'internal', 'confidential', 'restricted')),

    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Audit log is ALWAYS queried by tenant + time window
CREATE INDEX idx_audit_logs_tenant_time ON audit_logs(tenant_id, created_at DESC);
CREATE INDEX idx_audit_logs_user ON audit_logs(tenant_id, user_id, created_at DESC);
CREATE INDEX idx_audit_logs_run ON audit_logs(agent_run_id);
CREATE INDEX idx_audit_logs_action ON audit_logs(tenant_id, action, created_at DESC);

-- ─────────────────────────────────────────────
-- ROW LEVEL SECURITY
-- Belt-and-suspenders isolation on top of app-layer tenant_id scoping
-- ─────────────────────────────────────────────
ALTER TABLE agent_definitions ENABLE ROW LEVEL SECURITY;
ALTER TABLE agent_runs ENABLE ROW LEVEL SECURITY;
ALTER TABLE agent_steps ENABLE ROW LEVEL SECURITY;
ALTER TABLE audit_logs ENABLE ROW LEVEL SECURITY;

-- App connects as 'nafidhAI_app' role
-- This role can only see rows where tenant_id matches its session variable
CREATE POLICY tenant_isolation_agent_definitions
    ON agent_definitions
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID);

CREATE POLICY tenant_isolation_agent_runs
    ON agent_runs
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID);

CREATE POLICY tenant_isolation_agent_steps
    ON agent_steps
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID);

CREATE POLICY tenant_isolation_audit_logs
    ON audit_logs
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID);

-- ─────────────────────────────────────────────
-- AUDIT LOG IMMUTABILITY
-- Grant INSERT only — no UPDATE, no DELETE — on audit_logs
-- ─────────────────────────────────────────────
CREATE ROLE nafidhAI_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON agent_definitions TO nafidhAI_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON agent_runs TO nafidhAI_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON agent_steps TO nafidhAI_app;
GRANT SELECT, INSERT ON audit_logs TO nafidhAI_app;  -- INSERT ONLY — no UPDATE/DELETE

-- Prevent even superuser accidental deletes in production:
CREATE RULE no_update_audit_logs AS ON UPDATE TO audit_logs DO INSTEAD NOTHING;
CREATE RULE no_delete_audit_logs AS ON DELETE TO audit_logs DO INSTEAD NOTHING;
