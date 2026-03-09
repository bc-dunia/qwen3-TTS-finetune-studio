-- Qwen3-TTS Studio — D1 Database Schema
-- Stores metadata for voices, training jobs, and generation history.
-- Actual model files and audio are stored in R2.

-- ── Voices ──────────────────────────────────────────────────────────
-- Each voice represents a fine-tuned speaker model.
-- Maps to ElevenLabs Voice object for API compatibility.

CREATE TABLE IF NOT EXISTS voices (
    voice_id          TEXT PRIMARY KEY,           -- UUID
    name              TEXT NOT NULL,              -- Display name (ElevenLabs: name)
    description       TEXT DEFAULT '',            -- Voice description
    speaker_name      TEXT NOT NULL,              -- Internal speaker identifier for Qwen3-TTS
    model_size        TEXT DEFAULT '1.7B',        -- '1.7B' or '0.6B'
    model_id          TEXT DEFAULT 'qwen3-tts-1.7b', -- Model identifier for API
    category          TEXT DEFAULT 'cloned',      -- 'cloned' | 'premade' (ElevenLabs compat)
    status            TEXT DEFAULT 'created',     -- 'created' | 'training' | 'ready' | 'failed'
    checkpoint_r2_prefix TEXT,                    -- R2 prefix: checkpoints/{voice_id}/{run_name}/checkpoint-epoch-N
    run_name          TEXT,                       -- Training run name
    epoch             INTEGER,                    -- Best checkpoint epoch number
    sample_audio_r2_key TEXT,                     -- R2 key for preview audio
    ref_audio_r2_key  TEXT,                       -- R2 key for reference audio (used in training)
    labels_json       TEXT DEFAULT '{}',          -- JSON: {"accent": "...", "gender": "...", ...}
    settings_json     TEXT DEFAULT '{}',          -- JSON: voice_settings (stability, similarity_boost, etc.)
    created_at        INTEGER NOT NULL,           -- Unix timestamp (ms)
    updated_at        INTEGER NOT NULL            -- Unix timestamp (ms)
);

CREATE INDEX IF NOT EXISTS idx_voices_status ON voices(status);
CREATE INDEX IF NOT EXISTS idx_voices_name ON voices(name);


-- ── Training Jobs ───────────────────────────────────────────────────
-- Tracks RunPod GPU pod training sessions.
-- Status updates come from the training handler writing to R2,
-- polled by the Worker on frontend request.

CREATE TABLE IF NOT EXISTS training_jobs (
    job_id            TEXT PRIMARY KEY,           -- UUID
    voice_id          TEXT NOT NULL REFERENCES voices(voice_id),
    runpod_pod_id     TEXT,                       -- RunPod pod ID (set after pod creation)
    status            TEXT DEFAULT 'pending',     -- See status enum below
    config_json       TEXT NOT NULL,              -- Full training config (JSON)
    progress_json     TEXT DEFAULT '{}',          -- Current progress: {epoch, step, loss, eta}
    dataset_r2_prefix TEXT NOT NULL,              -- R2 prefix for training dataset
    error_message     TEXT,
    started_at        INTEGER,                    -- Unix timestamp (ms)
    completed_at      INTEGER,                    -- Unix timestamp (ms)
    created_at        INTEGER NOT NULL,
    updated_at        INTEGER NOT NULL
);

-- Training job status values:
--   pending     → Job created, waiting for pod
--   provisioning → RunPod pod starting
--   downloading → Downloading dataset from R2
--   preparing   → Running prepare_data.py (audio code extraction)
--   training    → Running sft_12hz.py (SFT)
--   uploading   → Uploading checkpoints to R2
--   completed   → Successfully finished
--   failed      → Error occurred
--   cancelled   → User cancelled

CREATE INDEX IF NOT EXISTS idx_training_jobs_voice ON training_jobs(voice_id);
CREATE INDEX IF NOT EXISTS idx_training_jobs_status ON training_jobs(status);


-- ── Dataset Preprocess Cache ───────────────────────────────────────
-- Tracks reusable long-form preprocessing artifacts (transcripts,
-- segments, reference audio) so follow-up training runs can skip ASR.

CREATE TABLE IF NOT EXISTS dataset_preprocess_caches (
    cache_id           TEXT PRIMARY KEY,          -- UUID
    voice_id           TEXT NOT NULL REFERENCES voices(voice_id),
    dataset_r2_prefix  TEXT NOT NULL,            -- Raw dataset prefix being cached
    dataset_signature  TEXT NOT NULL,            -- SHA-256 fingerprint of raw source objects
    cache_r2_prefix    TEXT NOT NULL,            -- R2 prefix holding cached train_raw + segments
    train_raw_r2_key   TEXT NOT NULL,
    ref_audio_r2_key   TEXT,
    reference_profile_r2_key TEXT,
    source_file_count  INTEGER,
    segments_created   INTEGER,
    segments_accepted  INTEGER,
    accepted_duration_min REAL,
    created_at         INTEGER NOT NULL,
    updated_at         INTEGER NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_dataset_preprocess_caches_lookup
  ON dataset_preprocess_caches(voice_id, dataset_r2_prefix, dataset_signature);
CREATE INDEX IF NOT EXISTS idx_dataset_preprocess_caches_voice
  ON dataset_preprocess_caches(voice_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS dataset_preprocess_cache_entries (
    entry_id           TEXT PRIMARY KEY,
    cache_id           TEXT NOT NULL REFERENCES dataset_preprocess_caches(cache_id),
    seq                INTEGER NOT NULL,
    audio_path         TEXT NOT NULL,
    audio_r2_key       TEXT NOT NULL,
    text               TEXT NOT NULL,
    included           INTEGER NOT NULL DEFAULT 1,
    created_at         INTEGER NOT NULL,
    updated_at         INTEGER NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_dataset_preprocess_cache_entries_seq
  ON dataset_preprocess_cache_entries(cache_id, seq);
CREATE INDEX IF NOT EXISTS idx_dataset_preprocess_cache_entries_cache
  ON dataset_preprocess_cache_entries(cache_id, included, seq);


-- ── Generations ─────────────────────────────────────────────────────
-- History of TTS generation requests.
-- Maps to ElevenLabs history items for API compatibility.

CREATE TABLE IF NOT EXISTS generations (
    generation_id     TEXT PRIMARY KEY,           -- UUID
    voice_id          TEXT NOT NULL REFERENCES voices(voice_id),
    model_id          TEXT DEFAULT 'qwen3-tts-1.7b',
    text              TEXT NOT NULL,              -- Input text
    audio_r2_key      TEXT,                       -- R2 key for generated audio
    output_format     TEXT DEFAULT 'wav_24000',   -- Output audio format
    duration_ms       INTEGER,                    -- Audio duration in milliseconds
    latency_ms        INTEGER,                    -- Generation latency in milliseconds
    settings_json     TEXT DEFAULT '{}',          -- Voice settings used
    review_json       TEXT,                       -- Post-generation review results
    created_at        INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_generations_voice ON generations(voice_id);
CREATE INDEX IF NOT EXISTS idx_generations_created ON generations(created_at DESC);


-- ── API Keys ────────────────────────────────────────────────────────
-- Simple API key auth for the ElevenLabs-compatible API.

CREATE TABLE IF NOT EXISTS api_keys (
    key_hash          TEXT PRIMARY KEY,           -- SHA-256 of the API key
    name              TEXT NOT NULL,              -- Key name/label
    permissions       TEXT DEFAULT 'all',         -- 'all' | 'read' | 'inference'
    is_active         INTEGER DEFAULT 1,          -- 0 = disabled
    last_used_at      INTEGER,
    created_at        INTEGER NOT NULL
);


-- ── Models ──────────────────────────────────────────────────────────
-- Available base models. Pre-populated.

CREATE TABLE IF NOT EXISTS models (
    model_id          TEXT PRIMARY KEY,
    name              TEXT NOT NULL,
    description       TEXT,
    can_do_text_to_speech INTEGER DEFAULT 1,
    can_be_finetuned  INTEGER DEFAULT 1,
    max_characters_request INTEGER DEFAULT 5000,
    languages_json    TEXT DEFAULT '[]'           -- JSON array of {language_id, name}
);

-- Seed default models
INSERT OR IGNORE INTO models (model_id, name, description, languages_json) VALUES
    ('qwen3-tts-1.7b', 'Qwen3-TTS 1.7B', 'High quality 1.7B parameter TTS model. Best for quality and expressiveness.',
     '[{"language_id":"zh","name":"Chinese"},{"language_id":"en","name":"English"},{"language_id":"ja","name":"Japanese"},{"language_id":"ko","name":"Korean"}]'),
    ('qwen3-tts-0.6b', 'Qwen3-TTS 0.6B', 'Lightweight 0.6B parameter TTS model. Faster inference, lower VRAM.',
     '[{"language_id":"zh","name":"Chinese"},{"language_id":"en","name":"English"},{"language_id":"ja","name":"Japanese"},{"language_id":"ko","name":"Korean"}]');
