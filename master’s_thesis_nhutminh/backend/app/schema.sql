-- PlateReader LR — PostgreSQL schema (PostgreSQL >= 13, gen_random_uuid() built in).
-- Idempotent: the API runs this on every startup.
-- Uploaded images are NOT stored (ICPR26-LRLPR license forbids redistribution).

-- One uploaded zip = one batch job (roadmap Tuần 7: /predict_batch).
CREATE TABLE IF NOT EXISTS batches (
    id               UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at      TIMESTAMPTZ,
    filename         TEXT        NOT NULL,
    status           TEXT        NOT NULL DEFAULT 'queued'
                     CHECK (status IN ('queued', 'running', 'completed', 'failed')),
    decode_mode      TEXT        NOT NULL CHECK (decode_mode IN ('constrained', 'greedy')),
    total_tracks     INTEGER     NOT NULL CHECK (total_tracks > 0),
    processed_tracks INTEGER     NOT NULL DEFAULT 0 CHECK (processed_tracks >= 0),
    error            TEXT
);

-- One row per recognised track (single /predict call or one track of a batch).
CREATE TABLE IF NOT EXISTS predictions (
    id             UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    batch_id       UUID        REFERENCES batches(id) ON DELETE CASCADE,
    track_id       TEXT,
    plate          VARCHAR(16) NOT NULL,
    track_conf     REAL        NOT NULL CHECK (track_conf BETWEEN 0 AND 1),
    char_conf      JSONB       NOT NULL,             -- [0.99, 0.87, ...] một số / ký tự
    suggestions    JSONB       NOT NULL,             -- [["ABC1D23", 0.91], ...] top-k beam
    status         TEXT        NOT NULL
                   CHECK (status IN ('accepted', 'needs_review', 'reviewed')),
    decode_mode    TEXT        NOT NULL CHECK (decode_mode IN ('constrained', 'greedy')),
    num_frames     SMALLINT    NOT NULL CHECK (num_frames BETWEEN 1 AND 5),
    latency_ms     REAL        NOT NULL,
    model_version  TEXT        NOT NULL,
    ground_truth   VARCHAR(16)                       -- chỉ có khi zip kèm annotations.json
);

CREATE INDEX IF NOT EXISTS idx_predictions_batch  ON predictions (batch_id, track_id);
CREATE INDEX IF NOT EXISTS idx_predictions_status ON predictions (status, created_at DESC);

-- Human corrections (roadmap Tuần 8, tab Review). Append-only history:
-- the latest row per prediction is the effective one.
CREATE TABLE IF NOT EXISTS reviews (
    id             BIGSERIAL   PRIMARY KEY,
    prediction_id  UUID        NOT NULL REFERENCES predictions(id) ON DELETE CASCADE,
    plate          VARCHAR(16) NOT NULL,
    reviewer       VARCHAR(64),
    note           TEXT,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_reviews_prediction ON reviews (prediction_id, created_at DESC);
