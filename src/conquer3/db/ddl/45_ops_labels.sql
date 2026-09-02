-- Ground truth assigned via the UI's Inspection tab (Layer 9) -- either through the
-- inline label editor or a label-CSV import. `event_id` is the same identity the
-- replay driver and bronze_to_silver both derive, so this joins directly onto
-- silver.txn without a second identity scheme.
--
-- `unlabeled` is deliberately NOT a value in this table -- it is row absence.
-- is_fraud is NOT NULL because a boolean cannot express three states; a UI
-- selection of "unlabeled" deletes the row instead of writing a NULL/sentinel.
CREATE TABLE IF NOT EXISTS ops.prediction_labels (
    event_id   TEXT NOT NULL PRIMARY KEY,
    is_fraud   BOOLEAN NOT NULL,
    source     TEXT NOT NULL,          -- 'ui' | 'csv'
    labeled_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
