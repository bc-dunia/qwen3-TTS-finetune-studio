CREATE TABLE IF NOT EXISTS dataset_preprocess_cache_entries (
  entry_id TEXT PRIMARY KEY,
  cache_id TEXT NOT NULL REFERENCES dataset_preprocess_caches(cache_id),
  seq INTEGER NOT NULL,
  audio_path TEXT NOT NULL,
  audio_r2_key TEXT NOT NULL,
  text TEXT NOT NULL,
  included INTEGER NOT NULL DEFAULT 1,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_dataset_preprocess_cache_entries_seq
  ON dataset_preprocess_cache_entries(cache_id, seq);
CREATE INDEX IF NOT EXISTS idx_dataset_preprocess_cache_entries_cache
  ON dataset_preprocess_cache_entries(cache_id, included, seq);
