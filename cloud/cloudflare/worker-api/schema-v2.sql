ALTER TABLE training_jobs ADD COLUMN last_heartbeat_at INTEGER;
ALTER TABLE training_jobs ADD COLUMN summary_json TEXT DEFAULT '{}';
ALTER TABLE training_jobs ADD COLUMN metrics_json TEXT DEFAULT '{}';
ALTER TABLE training_jobs ADD COLUMN log_r2_prefix TEXT;
ALTER TABLE training_jobs ADD COLUMN job_token TEXT;

CREATE TABLE IF NOT EXISTS training_log_chunks (
  job_id TEXT NOT NULL,
  seq INTEGER NOT NULL,
  r2_key TEXT NOT NULL,
  created_at INTEGER NOT NULL,
  bytes INTEGER,
  lines INTEGER,
  PRIMARY KEY (job_id, seq)
);

CREATE INDEX IF NOT EXISTS idx_log_chunks_job ON training_log_chunks(job_id, seq);
