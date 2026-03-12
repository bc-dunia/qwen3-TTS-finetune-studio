ALTER TABLE voices ADD COLUMN checkpoint_preset TEXT;
ALTER TABLE voices ADD COLUMN checkpoint_score REAL;
ALTER TABLE voices ADD COLUMN checkpoint_job_id TEXT;
ALTER TABLE voices ADD COLUMN candidate_preset TEXT;

ALTER TABLE training_rounds ADD COLUMN production_preset TEXT;
ALTER TABLE training_rounds ADD COLUMN production_score REAL;
ALTER TABLE training_rounds ADD COLUMN production_job_id TEXT;
ALTER TABLE training_rounds ADD COLUMN champion_checkpoint_r2_prefix TEXT;
ALTER TABLE training_rounds ADD COLUMN champion_run_name TEXT;
ALTER TABLE training_rounds ADD COLUMN champion_epoch INTEGER;
ALTER TABLE training_rounds ADD COLUMN champion_preset TEXT;
ALTER TABLE training_rounds ADD COLUMN champion_score REAL;
ALTER TABLE training_rounds ADD COLUMN champion_job_id TEXT;
ALTER TABLE training_rounds ADD COLUMN selected_checkpoint_r2_prefix TEXT;
ALTER TABLE training_rounds ADD COLUMN selected_run_name TEXT;
ALTER TABLE training_rounds ADD COLUMN selected_epoch INTEGER;
ALTER TABLE training_rounds ADD COLUMN selected_preset TEXT;
ALTER TABLE training_rounds ADD COLUMN selected_score REAL;
ALTER TABLE training_rounds ADD COLUMN selected_job_id TEXT;
ALTER TABLE training_rounds ADD COLUMN adoption_mode TEXT;

CREATE TABLE IF NOT EXISTS training_checkout_ledger (
  entry_id TEXT PRIMARY KEY,
  round_id TEXT REFERENCES training_rounds(round_id),
  job_id TEXT NOT NULL REFERENCES training_jobs(job_id),
  voice_id TEXT NOT NULL REFERENCES voices(voice_id),
  checkpoint_r2_prefix TEXT NOT NULL,
  run_name TEXT,
  epoch INTEGER,
  preset TEXT,
  score REAL,
  ok INTEGER,
  passed_samples INTEGER,
  total_samples INTEGER,
  message TEXT,
  role TEXT NOT NULL,
  source TEXT NOT NULL,
  adoption_mode TEXT,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_training_checkout_ledger_job
  ON training_checkout_ledger(job_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_training_checkout_ledger_round
  ON training_checkout_ledger(round_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_training_checkout_ledger_voice
  ON training_checkout_ledger(voice_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_training_checkout_ledger_checkpoint
  ON training_checkout_ledger(checkpoint_r2_prefix, created_at DESC);
