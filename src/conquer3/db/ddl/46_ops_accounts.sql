-- ops.accounts: demo ledger backing the UI's "Score a transaction" form (Layer 9).
-- name_acc is PaySim's nameOrig/nameDest identity space; balance is this ops ledger's
-- own running total -- unrelated to bronze/silver/gold, which record what the
-- historical simulation's balances *were*, not what a live UI-initiated transfer owes.
-- Rows are auto-provisioned by conquer3.db.accounts.execute_transfer on first use, not
-- seeded here.
CREATE TABLE IF NOT EXISTS ops.accounts (
    name_acc   TEXT NOT NULL PRIMARY KEY,
    balance    NUMERIC(18,2) NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
