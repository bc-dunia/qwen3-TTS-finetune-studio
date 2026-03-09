CREATE TABLE IF NOT EXISTS dataset_preprocess_caches (
  cache_id TEXT PRIMARY KEY,
  voice_id TEXT NOT NULL REFERENCES voices(voice_id),
  dataset_r2_prefix TEXT NOT NULL,
  dataset_signature TEXT NOT NULL,
  cache_r2_prefix TEXT NOT NULL,
  train_raw_r2_key TEXT NOT NULL,
  ref_audio_r2_key TEXT,
  reference_profile_r2_key TEXT,
  source_file_count INTEGER,
  segments_created INTEGER,
  segments_accepted INTEGER,
  accepted_duration_min REAL,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_dataset_preprocess_caches_lookup
  ON dataset_preprocess_caches(voice_id, dataset_r2_prefix, dataset_signature);
CREATE INDEX IF NOT EXISTS idx_dataset_preprocess_caches_voice
  ON dataset_preprocess_caches(voice_id, updated_at DESC);
