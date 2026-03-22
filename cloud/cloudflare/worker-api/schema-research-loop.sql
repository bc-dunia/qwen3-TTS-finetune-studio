CREATE TABLE IF NOT EXISTS voice_research_state (
  voice_id TEXT PRIMARY KEY,
  cycle_count INTEGER NOT NULL DEFAULT 0,
  current_bottleneck TEXT,
  active_hypothesis TEXT,
  stable_lessons_json TEXT NOT NULL DEFAULT '[]',
  pending_action TEXT,
  pending_action_params_json TEXT,
  dataset_snapshot_id TEXT,
  calibration_summary_json TEXT,
  scoring_policy_version INTEGER NOT NULL DEFAULT 1,
  autonomy_mode TEXT NOT NULL DEFAULT 'supervised',
  last_retrospective_json TEXT,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS voice_research_journal (
  entry_id TEXT PRIMARY KEY,
  voice_id TEXT NOT NULL,
  cycle_id INTEGER NOT NULL,
  trigger TEXT NOT NULL,
  linked_ids_json TEXT,
  observations TEXT NOT NULL,
  hypothesis TEXT,
  decision TEXT NOT NULL,
  decision_params_json TEXT,
  expected_signal TEXT,
  outcome TEXT,
  confidence TEXT NOT NULL DEFAULT 'low',
  created_at INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_voice_research_journal_voice
  ON voice_research_journal(voice_id, cycle_id DESC);
CREATE INDEX IF NOT EXISTS idx_voice_research_journal_trigger
  ON voice_research_journal(voice_id, trigger, created_at DESC);
