ALTER TABLE voices ADD COLUMN candidate_checkpoint_r2_prefix TEXT;
ALTER TABLE voices ADD COLUMN candidate_run_name TEXT;
ALTER TABLE voices ADD COLUMN candidate_epoch INTEGER;
ALTER TABLE voices ADD COLUMN candidate_score REAL;
ALTER TABLE voices ADD COLUMN candidate_job_id TEXT;
ALTER TABLE voices ADD COLUMN active_round_id TEXT;

CREATE INDEX IF NOT EXISTS idx_voices_active_round ON voices(active_round_id);

ALTER TABLE training_jobs ADD COLUMN round_id TEXT;
ALTER TABLE training_jobs ADD COLUMN dataset_snapshot_id TEXT;
ALTER TABLE training_jobs ADD COLUMN supervisor_json TEXT DEFAULT '{}';

CREATE INDEX IF NOT EXISTS idx_training_jobs_round ON training_jobs(round_id);
CREATE INDEX IF NOT EXISTS idx_training_jobs_snapshot ON training_jobs(dataset_snapshot_id);

CREATE TABLE IF NOT EXISTS dataset_snapshots (
  snapshot_id TEXT PRIMARY KEY,
  voice_id TEXT NOT NULL REFERENCES voices(voice_id),
  dataset_name TEXT,
  dataset_r2_prefix TEXT NOT NULL,
  dataset_signature TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'draft',
  source_cache_id TEXT,
  cache_r2_prefix TEXT,
  train_raw_r2_key TEXT,
  ref_audio_r2_key TEXT,
  reference_profile_r2_key TEXT,
  reference_text TEXT,
  source_file_count INTEGER,
  segments_created INTEGER,
  segments_accepted INTEGER,
  accepted_duration_min REAL,
  created_from_job_id TEXT,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_dataset_snapshots_lookup
  ON dataset_snapshots(voice_id, dataset_r2_prefix, dataset_signature);
CREATE INDEX IF NOT EXISTS idx_dataset_snapshots_voice
  ON dataset_snapshots(voice_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS training_rounds (
  round_id TEXT PRIMARY KEY,
  voice_id TEXT NOT NULL REFERENCES voices(voice_id),
  dataset_snapshot_id TEXT REFERENCES dataset_snapshots(snapshot_id),
  round_index INTEGER NOT NULL,
  status TEXT NOT NULL DEFAULT 'draft',
  production_checkpoint_r2_prefix TEXT,
  production_run_name TEXT,
  production_epoch INTEGER,
  candidate_checkpoint_r2_prefix TEXT,
  candidate_run_name TEXT,
  candidate_epoch INTEGER,
  candidate_score REAL,
  candidate_job_id TEXT,
  summary_json TEXT DEFAULT '{}',
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL,
  started_at INTEGER,
  completed_at INTEGER
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_training_rounds_voice_round
  ON training_rounds(voice_id, round_index);
CREATE INDEX IF NOT EXISTS idx_training_rounds_voice
  ON training_rounds(voice_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_training_rounds_snapshot
  ON training_rounds(dataset_snapshot_id);
