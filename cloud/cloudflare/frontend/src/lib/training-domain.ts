import type { TrainingConfig } from './api'

export function readNumber(value: unknown): number | null {
  const numeric = Number(value)
  return Number.isFinite(numeric) ? numeric : null
}

export function readText(value: unknown): string | null {
  return typeof value === 'string' && value.trim() ? value.trim() : null
}

export function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value))
}

export function parseRunNameFromCheckpointPrefix(prefix: string): string | null {
  const parts = prefix.split('/')
  if (parts.length < 4 || parts[0] !== 'checkpoints') return null
  return parts[2] || null
}

export const MAX_CONCURRENT_PODS = 3

export interface TrainingDefaults {
  model_size: string
  batch_size: number
  learning_rate: number
  num_epochs: number
  gradient_accumulation_steps: number
  subtalker_loss_weight: number
  save_every_n_epochs: number
  seed: number
  gpu_type_id: string
}

const DEFAULTS_0_6B: TrainingDefaults = {
  model_size: '0.6B',
  batch_size: 2,
  learning_rate: 2.5e-6,
  num_epochs: 12,
  gradient_accumulation_steps: 4,
  subtalker_loss_weight: 0.3,
  save_every_n_epochs: 1,
  seed: 303,
  gpu_type_id: 'NVIDIA L40S',
}

const DEFAULTS_1_7B: TrainingDefaults = {
  model_size: '1.7B',
  batch_size: 2,
  learning_rate: 2e-5,
  num_epochs: 15,
  gradient_accumulation_steps: 4,
  subtalker_loss_weight: 0.3,
  save_every_n_epochs: 5,
  seed: 42,
  gpu_type_id: 'NVIDIA A100-SXM4-80GB',
}

export function getTrainingDefaults(modelSize: string): TrainingDefaults {
  return modelSize.includes('0.6') ? { ...DEFAULTS_0_6B } : { ...DEFAULTS_1_7B }
}

export function getDefaultTrainingConfig(
  modelSize: string,
  language: string | undefined,
): TrainingConfig {
  const defaults = getTrainingDefaults(modelSize)
  const whisperLanguage = (language ?? 'ko').trim() || 'ko'
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
  }
}

export function sanitizeConfig(
  source: Partial<TrainingConfig>,
  modelSize: string,
  language: string | undefined,
): TrainingConfig {
  const defaults = getDefaultTrainingConfig(modelSize, language)
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
  }
}

export function pickAlternateSeed(seed: number, modelSize: string): number {
  const sequence = modelSize.includes('0.6') ? [303, 202, 77] : [808, 202, 42, 303]
  const currentIndex = sequence.indexOf(seed)
  if (currentIndex === -1) return sequence[0]
  return sequence[(currentIndex + 1) % sequence.length]
}
