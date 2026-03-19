-- Arena Evaluation System — D1 Migration
-- Run via: npx wrangler d1 execute qwen-tts-db --file=schema-arena.sql

-- ── Arena Sessions ──────────────────────────────────────────────────────────────
-- One evaluation batch: a set of candidates compared pairwise via Swiss or RR.

CREATE TABLE IF NOT EXISTS arena_sessions (
    session_id              TEXT PRIMARY KEY,
    voice_id                TEXT NOT NULL,
    status                  TEXT NOT NULL DEFAULT 'assembling',
    -- 'assembling' | 'generating' | 'active' | 'completed' | 'cancelled'
    algorithm               TEXT NOT NULL DEFAULT 'swiss',
    -- 'swiss' | 'round_robin'
    current_round           INTEGER NOT NULL DEFAULT 0,
    total_rounds            INTEGER,
    test_texts_json         TEXT NOT NULL,
    seed                    INTEGER DEFAULT 42,
    settings_json           TEXT NOT NULL DEFAULT '{}',
    ranking_json            TEXT NOT NULL DEFAULT '{}',
    winner_candidate_id     TEXT,
    promoted                INTEGER DEFAULT 0,
    notes                   TEXT,
    created_at              INTEGER NOT NULL,
    completed_at            INTEGER
);

CREATE INDEX IF NOT EXISTS idx_arena_sessions_voice
  ON arena_sessions(voice_id, created_at DESC);

-- ── Arena Candidates ────────────────────────────────────────────────────────────
-- Individual checkpoint entries within a session.

CREATE TABLE IF NOT EXISTS arena_candidates (
    candidate_id            TEXT PRIMARY KEY,
    session_id              TEXT NOT NULL,
    voice_id                TEXT NOT NULL,
    checkpoint_r2_prefix    TEXT NOT NULL,
    job_id                  TEXT,
    run_name                TEXT,
    epoch                   INTEGER,
    source                  TEXT NOT NULL,
    -- 'champion_carry' | 'second_carry' | 'third_carry' | 'new'
    seed_rank               INTEGER,
    final_rank              INTEGER,
    wins                    INTEGER DEFAULT 0,
    losses                  INTEGER DEFAULT 0,
    ties                    INTEGER DEFAULT 0,
    bye_count               INTEGER DEFAULT 0,
    buchholz                REAL DEFAULT 0,
    retention_status        TEXT DEFAULT 'active',
    -- 'active' | 'champion' | 'second' | 'third' | 'eliminated' | 'purged'
    auto_scores_json        TEXT,
    created_at              INTEGER NOT NULL,
    eliminated_at           INTEGER
);

CREATE INDEX IF NOT EXISTS idx_arena_candidates_session
  ON arena_candidates(session_id);
CREATE INDEX IF NOT EXISTS idx_arena_candidates_voice_retention
  ON arena_candidates(voice_id, retention_status);

-- ── Arena Matches ───────────────────────────────────────────────────────────────
-- Individual pairwise comparisons within a session.

CREATE TABLE IF NOT EXISTS arena_matches (
    match_id                TEXT PRIMARY KEY,
    session_id              TEXT NOT NULL,
    round_number            INTEGER NOT NULL,
    candidate_a_id          TEXT NOT NULL,
    candidate_b_id          TEXT NOT NULL,
    display_order           TEXT NOT NULL DEFAULT 'ab',
    -- 'ab' | 'ba' (randomized: which candidate plays as "Sample 1")
    text_index              INTEGER NOT NULL,
    audio_a_r2_key          TEXT,
    audio_b_r2_key          TEXT,
    winner                  TEXT,
    -- 'a' | 'b' | 'tie' | 'both_bad' | NULL (not yet voted)
    confidence              TEXT,
    -- 'clear' | 'slight' | NULL
    replay_count_a          INTEGER DEFAULT 0,
    replay_count_b          INTEGER DEFAULT 0,
    created_at              INTEGER NOT NULL,
    voted_at                INTEGER
);

CREATE INDEX IF NOT EXISTS idx_arena_matches_session
  ON arena_matches(session_id, round_number);
CREATE INDEX IF NOT EXISTS idx_arena_matches_candidates
  ON arena_matches(session_id, candidate_a_id, candidate_b_id);

-- ── Arena Calibration Overrides ─────────────────────────────────────────────────
-- Learned ranking weights from arena preference data.

CREATE TABLE IF NOT EXISTS arena_calibration_overrides (
    override_id             TEXT PRIMARY KEY,
    voice_id                TEXT NOT NULL DEFAULT '__global__',
    weights_json            TEXT NOT NULL,
    effective_weights_json  TEXT,
    matchup_count           INTEGER NOT NULL,
    effective_matchup_count INTEGER NOT NULL DEFAULT 0,
    ranking_pseudo_pairs_count INTEGER NOT NULL DEFAULT 0,
    accuracy                REAL,
    confidence              TEXT NOT NULL,
    state                   TEXT NOT NULL DEFAULT 'shadow',
    version                 INTEGER NOT NULL DEFAULT 1,
    alpha                   REAL NOT NULL DEFAULT 0,
    weight_shifts_json      TEXT,
    gate_diagnostics_json   TEXT,
    rollback_reason         TEXT,
    shadow_accuracy         REAL,
    created_at              INTEGER NOT NULL,
    updated_at              INTEGER NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_arena_calibration_voice
  ON arena_calibration_overrides(voice_id);
