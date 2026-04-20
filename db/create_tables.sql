-- Run these statements once in the Supabase SQL editor
-- to create the two required tables.

-- ─────────────────────────────────────────
-- 1. Training / analysis data (from CSV)
-- ─────────────────────────────────────────
CREATE TABLE IF NOT EXISTS obesity_data (
    id            BIGSERIAL PRIMARY KEY,
    gender        TEXT,
    age           NUMERIC,
    height        NUMERIC,
    weight        NUMERIC,
    family_history TEXT,          -- family_history_with_overweight
    favc          TEXT,
    fcvc          NUMERIC,
    ncp           NUMERIC,
    caec          TEXT,
    smoke         TEXT,
    ch2o          NUMERIC,
    scc           TEXT,
    faf           NUMERIC,
    tue           NUMERIC,
    calc          TEXT,
    mtrans        TEXT,
    obesity       TEXT            -- target column
);

-- ─────────────────────────────────────────
-- 2. Prediction log (runtime entries)
-- ─────────────────────────────────────────
CREATE TABLE IF NOT EXISTS predictions_log (
    id              BIGSERIAL PRIMARY KEY,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    gender          TEXT,
    age             NUMERIC,
    height          NUMERIC,
    weight          NUMERIC,
    family_history  TEXT,
    favc            TEXT,
    fcvc            NUMERIC,
    ncp             NUMERIC,
    caec            TEXT,
    smoke           TEXT,
    ch2o            NUMERIC,
    scc             TEXT,
    faf             NUMERIC,
    tue             NUMERIC,
    calc            TEXT,
    mtrans          TEXT,
    predicted_class TEXT,
    probability     NUMERIC       -- max class probability (0-1)
);
