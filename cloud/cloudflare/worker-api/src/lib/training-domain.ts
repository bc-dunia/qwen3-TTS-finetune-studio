/**
 * training-domain.ts — Canonical training configuration, defaults, helpers, and validation presets.
 *
 * This is the SINGLE SOURCE OF TRUTH for training parameters.
 * All other modules (training-advisor, training-checkout, campaign-planner, routes/training)
 * MUST import from here instead of maintaining local copies.
 */

import type { TrainingConfig } from "../types";

// ── Parsing helpers ────────────────────────────────────────────────────────────

export function readNumber(value: unknown): number | null {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
}

export function readText(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

export function readTimestamp(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === "string" && value.trim()) {
    const parsed = Date.parse(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

export function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

// ── Path helpers ───────────────────────────────────────────────────────────────

export function stripSlashes(value: string): string {
  return value.replace(/^\/+|\/+$/g, "");
}

export function parseRunNameFromCheckpointPrefix(prefix: string): string | null {
  const parts = prefix.split("/");
  if (parts.length < 4 || parts[0] !== "checkpoints") {
    return null;
  }
  return parts[2] || null;
}

export function extractDatasetNameFromPrefix(datasetPrefix: string): string | null {
  const parts = stripSlashes(datasetPrefix).split("/");
  if (parts.length < 3 || parts[0] !== "datasets") {
    return null;
  }
  const name = parts.slice(2).join("/").trim();
  return name || null;
}

// ── Queue / concurrency constants ──────────────────────────────────────────────

export const MAX_CONCURRENT_PODS = 3;
export const DEFAULT_MAX_ACTIVE_TRAINING_JOBS_PER_VOICE = 1;
export const DEFAULT_MAX_ACTIVE_TRAINING_JOBS_GLOBAL = 3;

export const ACTIVE_JOB_STATUSES = new Set([
  "queued",
  "pending",
  "running",
  "provisioning",
  "downloading",
  "preprocessing",
  "preparing",
  "training",
  "uploading",
]);

export const ACTIVE_RUNTIME_STATUSES = new Set([
  "running",
  "downloading",
  "preprocessing",
  "preparing",
  "training",
  "uploading",
]);

export const TERMINAL_JOB_STATUSES = new Set(["completed", "failed", "cancelled"]);

// ── Training config defaults (SINGLE SOURCE OF TRUTH) ──────────────────────────

export interface TrainingDefaults {
  model_size: string;
  batch_size: number;
  learning_rate: number;
  num_epochs: number;
  gradient_accumulation_steps: number;
  subtalker_loss_weight: number;
  save_every_n_epochs: number;
  seed: number;
  gpu_type_id: string;
}

const DEFAULTS_0_6B: TrainingDefaults = {
  model_size: "0.6B",
  batch_size: 2,
  learning_rate: 2.5e-6,
  num_epochs: 12,
  gradient_accumulation_steps: 4,
  subtalker_loss_weight: 0.3,
  save_every_n_epochs: 1,
  seed: 303,
  gpu_type_id: "NVIDIA L40S",
};

const DEFAULTS_1_7B: TrainingDefaults = {
  model_size: "1.7B",
  batch_size: 2,
  learning_rate: 2e-5,
  num_epochs: 15,
  gradient_accumulation_steps: 4,
  subtalker_loss_weight: 0.3,
  save_every_n_epochs: 5,
  seed: 42,
  gpu_type_id: "NVIDIA A100-SXM4-80GB",
};

export function getTrainingDefaults(modelSize: string): TrainingDefaults {
  if (modelSize.includes("0.6")) {
    return { ...DEFAULTS_0_6B };
  }
  return { ...DEFAULTS_1_7B };
}

export function getDefaultTrainingConfig(
  modelSize: string,
  language: string | undefined,
): TrainingConfig {
  const defaults = getTrainingDefaults(modelSize);
  const whisperLanguage = (language ?? "ko").trim() || "ko";
  return {
    model_size: defaults.model_size,
    batch_size: defaults.batch_size,
    num_epochs: defaults.num_epochs,
    learning_rate: defaults.learning_rate,
    gradient_accumulation_steps: defaults.gradient_accumulation_steps,
    subtalker_loss_weight: defaults.subtalker_loss_weight,
    save_every_n_epochs: defaults.save_every_n_epochs,
    seed: defaults.seed,
    whisper_language: whisperLanguage,
    gpu_type_id: defaults.gpu_type_id,
  };
}

export function sanitizeConfig(
  source: Partial<TrainingConfig>,
  modelSize: string,
  language: string | undefined,
): TrainingConfig {
  const defaults = getDefaultTrainingConfig(modelSize, language);
  return {
    model_size: modelSize,
    batch_size: readNumber(source.batch_size) ?? defaults.batch_size,
    num_epochs: readNumber(source.num_epochs) ?? defaults.num_epochs,
    learning_rate: readNumber(source.learning_rate) ?? defaults.learning_rate,
    gradient_accumulation_steps:
      readNumber(source.gradient_accumulation_steps) ?? defaults.gradient_accumulation_steps,
    subtalker_loss_weight:
      readNumber(source.subtalker_loss_weight) ?? defaults.subtalker_loss_weight,
    save_every_n_epochs:
      readNumber(source.save_every_n_epochs) ?? defaults.save_every_n_epochs,
    seed: readNumber(source.seed) ?? defaults.seed,
    whisper_language: readText(source.whisper_language) ?? defaults.whisper_language,
    gpu_type_id: readText(source.gpu_type_id) ?? defaults.gpu_type_id,
  };
}

// ── Validation presets (SINGLE SOURCE OF TRUTH) ────────────────────────────────

export interface ValidationThresholds {
  asr_min: number;
  speaker_min: number;
  health_min: number;
  tone_min: number;
  speed_min: number;
  overall_min: number;
  duration_min: number;
}

export interface ValidationWeights {
  asr: number;
  speaker: number;
  tone: number;
  speed: number;
  overall: number;
  duration: number;
}

/**
 * Gated thresholds — hard minimums that must pass before a checkpoint
 * is considered "usable" at all.
 */
export const VALIDATION_GATE_THRESHOLDS: ValidationThresholds = {
  asr_min: 0.8,
  speaker_min: 0.75,
  health_min: 0.72,
  tone_min: 0.55,       // Raised from 0.4 — 40% likeness was too permissive
  speed_min: 0.2,
  overall_min: 0.85,
  duration_min: 0.45,
};

/**
 * Ranking weights for comparing checkpoints that have passed the gate.
 * ASR and speaker are weighted highest because they directly affect
 * perceived voice quality. Duration is a weak signal.
 */
export const VALIDATION_RANKING_WEIGHTS: ValidationWeights = {
  asr: 0.25,
  speaker: 0.25,
  tone: 0.20,
  speed: 0.15,
  overall: 0.10,
  duration: 0.05,
};

export interface CheckpointScores {
  asr_score?: number | null;
  speaker_score?: number | null;
  health_score?: number | null;
  tone_score?: number | null;
  speed_score?: number | null;
  overall_score?: number | null;
  duration_score?: number | null;
}

/**
 * Check if a checkpoint passes all hard minimum gates.
 */
export function passesValidationGate(
  scores: CheckpointScores,
  thresholds: ValidationThresholds = VALIDATION_GATE_THRESHOLDS,
): boolean {
  const asr = readNumber(scores.asr_score);
  const speaker = readNumber(scores.speaker_score);
  const health = readNumber(scores.health_score);
  const tone = readNumber(scores.tone_score);
  const speed = readNumber(scores.speed_score);
  const overall = readNumber(scores.overall_score);
  const duration = readNumber(scores.duration_score);

  if (asr !== null && asr < thresholds.asr_min) return false;
  if (speaker !== null && speaker < thresholds.speaker_min) return false;
  if (health !== null && health < thresholds.health_min) return false;
  if (tone !== null && tone < thresholds.tone_min) return false;
  if (speed !== null && speed < thresholds.speed_min) return false;
  if (overall !== null && overall < thresholds.overall_min) return false;
  if (duration !== null && duration < thresholds.duration_min) return false;

  return true;
}

/**
 * Compute a weighted ranking score for comparing checkpoints.
 * Only meaningful for checkpoints that already pass the gate.
 */
export function computeRankingScore(
  scores: CheckpointScores,
  weights: ValidationWeights = VALIDATION_RANKING_WEIGHTS,
): number {
  const asr = readNumber(scores.asr_score) ?? 0;
  const speaker = readNumber(scores.speaker_score) ?? 0;
  const tone = readNumber(scores.tone_score) ?? 0;
  const speed = readNumber(scores.speed_score) ?? 0;
  const overall = readNumber(scores.overall_score) ?? 0;
  const duration = readNumber(scores.duration_score) ?? 0;

  return (
    asr * weights.asr +
    speaker * weights.speaker +
    tone * weights.tone +
    speed * weights.speed +
    overall * weights.overall +
    duration * weights.duration
  );
}

/**
 * Select the best checkpoint from a list using gated weighted scoring.
 * Returns the index of the best candidate, or -1 if none pass the gate.
 */
export function selectBestCheckpoint(
  candidates: CheckpointScores[],
  thresholds: ValidationThresholds = VALIDATION_GATE_THRESHOLDS,
  weights: ValidationWeights = VALIDATION_RANKING_WEIGHTS,
): number {
  let bestIndex = -1;
  let bestScore = -Infinity;

  for (let i = 0; i < candidates.length; i++) {
    const candidate = candidates[i];
    if (!passesValidationGate(candidate, thresholds)) {
      continue;
    }
    const score = computeRankingScore(candidate, weights);
    if (score > bestScore) {
      bestScore = score;
      bestIndex = i;
    }
  }

  return bestIndex;
}

// ── Seed utilities ─────────────────────────────────────────────────────────────

const SEED_SEQUENCE_0_6B = [303, 202, 77];
const SEED_SEQUENCE_1_7B = [808, 202, 42, 303];

export function pickAlternateSeed(seed: number, modelSize: string): number {
  const sequence = modelSize.includes("0.6") ? SEED_SEQUENCE_0_6B : SEED_SEQUENCE_1_7B;
  const currentIndex = sequence.indexOf(seed);
  if (currentIndex === -1) return sequence[0];
  return sequence[(currentIndex + 1) % sequence.length];
}

// ── Priority tiers for queue scheduling ────────────────────────────────────────

export type QueuePriority = 1 | 2 | 3;

/**
 * Priority tier assignment:
 * Tier 1 (highest): Manual / user-triggered runs
 * Tier 2: Campaign runs for voices with no usable checkpoint yet
 * Tier 3 (lowest): Improvement campaigns for already-usable voices
 */
export function getJobPriority(
  job: { campaign_id?: string | null },
  voiceHasCheckpoint: boolean,
): QueuePriority {
  if (!job.campaign_id) return 1;
  if (!voiceHasCheckpoint) return 2;
  return 3;
}
