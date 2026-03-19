-- Arena Calibration Feedback Loop — Migration 001
-- Adds calibration state machine fields to arena_calibration_overrides.
-- Run via: npx wrangler d1 execute qwen-tts-db --remote --file=schema-arena-migration-001.sql

ALTER TABLE arena_calibration_overrides ADD COLUMN effective_weights_json TEXT;
ALTER TABLE arena_calibration_overrides ADD COLUMN state TEXT NOT NULL DEFAULT 'shadow';
ALTER TABLE arena_calibration_overrides ADD COLUMN version INTEGER NOT NULL DEFAULT 1;
ALTER TABLE arena_calibration_overrides ADD COLUMN alpha REAL NOT NULL DEFAULT 0;
ALTER TABLE arena_calibration_overrides ADD COLUMN effective_matchup_count INTEGER NOT NULL DEFAULT 0;
ALTER TABLE arena_calibration_overrides ADD COLUMN ranking_pseudo_pairs_count INTEGER NOT NULL DEFAULT 0;
ALTER TABLE arena_calibration_overrides ADD COLUMN rollback_reason TEXT;
ALTER TABLE arena_calibration_overrides ADD COLUMN shadow_accuracy REAL;
