-- gold.account_state: Postgres mirror of Redis's per-account AccountState.
-- Deliberately a serialized-JSON blob, not exploded columns -- unlike
-- gold.txn_features (which IS generated from core.schema.FEATURE_NAMES),
-- AccountState's shape changing must never require regenerating this DDL.
--
-- state_json is TEXT, not JSONB: the licensed pw.io.postgres.write connector
-- writes a Pathway `str` column, and its Rust driver refuses to write into a
-- JSONB destination column ("declared Pathway type 'str' is not compatible with
-- PostgreSQL type 'jsonb'") -- confirmed by running it against a live Postgres.
-- TEXT is what both the licensed connector and the psycopg fallback
-- (pipelines/pathway/sinks/postgres_sink.py) can agree on; it costs native
-- ->>'field' querying, which nothing here relies on (core.serde.state_from_json
-- is the only reader).
--
-- Written by conquer3.pipelines.pathway (Layer 3b) via either sink -- never
-- hand-edited, never TRUNCATEd by anything but a full Pathway static backfill.
CREATE TABLE IF NOT EXISTS gold.account_state (
    account_id    TEXT NOT NULL PRIMARY KEY,
    state_version INTEGER NOT NULL,
    state_json    TEXT NOT NULL,
    updated_at_us BIGINT NOT NULL
);
