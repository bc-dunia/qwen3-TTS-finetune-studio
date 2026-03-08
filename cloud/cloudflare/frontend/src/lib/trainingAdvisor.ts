import type { TrainingConfig, TrainingJob, Voice } from './api'

export type TrainingAdviceMode =
  | 'compare-first'
  | 'dataset-first'
  | 'tone-explore'
  | 'stability-reset'
  | 'hold-current'

export interface TrainingAdvice {
  mode: TrainingAdviceMode
  title: string
  summary: string
  confidence: 'high' | 'medium'
  reasons: string[]
  suggestedConfig: TrainingConfig | null
  compareFirst: boolean
  reviewDatasetFirst: boolean
}

function readNumber(value: unknown): number | null {
  const numeric = Number(value)
  return Number.isFinite(numeric) ? numeric : null
}

function readText(value: unknown): string | null {
  return typeof value === 'string' && value.trim() ? value.trim() : null
}

function getDefaultTrainingConfig(modelSize: string, language: string | undefined): TrainingConfig {
  const whisperLanguage = (language ?? 'ko').trim() || 'ko'
  if (modelSize.includes('0.6')) {
    return {
      model_size: '0.6B',
      batch_size: 2,
      num_epochs: 12,
      learning_rate: 0.0000025,
      gradient_accumulation_steps: 4,
      subtalker_loss_weight: 0.3,
      save_every_n_epochs: 1,
      seed: 303,
      whisper_language: whisperLanguage,
      gpu_type_id: 'NVIDIA L40S',
    }
  }

  return {
    model_size: '1.7B',
    batch_size: 2,
    num_epochs: 15,
    learning_rate: 0.00002,
    gradient_accumulation_steps: 4,
    subtalker_loss_weight: 0.3,
    save_every_n_epochs: 5,
    seed: 42,
    whisper_language: whisperLanguage,
    gpu_type_id: 'NVIDIA A100-SXM4-80GB',
  }
}

function pickAlternateSeed(seed: number, modelSize: string): number {
  const sequence = modelSize.includes('0.6') ? [303, 202, 77] : [202, 42, 303]
  const currentIndex = sequence.indexOf(seed)
  if (currentIndex === -1) return sequence[0]
  return sequence[(currentIndex + 1) % sequence.length]
}

function sanitizeConfig(source: Partial<TrainingConfig>, modelSize: string, language: string | undefined): TrainingConfig {
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

function getSelectedScore(job: TrainingJob): number | null {
  return readNumber(job.summary?.selected_score)
}

function getSelectedPrefix(job: TrainingJob): string | null {
  return readText(job.summary?.selected_checkpoint_prefix)
}

function getValidationMessage(job: TrainingJob): string | null {
  return readText(job.summary?.validation_message) ?? readText(job.error_message)
}

function getBaseConfig(voice: Voice, jobs: TrainingJob[]): TrainingConfig {
  const validated = jobs
    .filter((job) => job.summary?.validation_passed === true)
    .sort((a, b) => (getSelectedScore(b) ?? -1) - (getSelectedScore(a) ?? -1))
  const latest = [...jobs].sort((a, b) => (Number(b.created_at) || 0) - (Number(a.created_at) || 0))
  const source = validated[0]?.config ?? latest[0]?.config ?? {}
  return sanitizeConfig(source, voice.model_size, voice.labels?.language)
}

function tuneForTone(base: TrainingConfig, modelSize: string): TrainingConfig {
  const is06b = modelSize.includes('0.6')
  return {
    ...base,
    learning_rate: Math.max(is06b ? 0.000002 : 0.0000045, Number(base.learning_rate) * (is06b ? 0.92 : 0.84)),
    num_epochs: Math.min(is06b ? 16 : 18, Number(base.num_epochs) + (is06b ? 2 : 1)),
    subtalker_loss_weight: is06b
      ? Math.min(0.32, Math.max(0.28, Number(base.subtalker_loss_weight ?? 0.3)))
      : Math.max(0.16, Number(base.subtalker_loss_weight ?? 0.2) - 0.02),
    save_every_n_epochs: 1,
    seed: pickAlternateSeed(Number(base.seed ?? 42), modelSize),
  }
}

function tuneForStability(base: TrainingConfig, modelSize: string): TrainingConfig {
  const is06b = modelSize.includes('0.6')
  return {
    ...base,
    learning_rate: Math.max(is06b ? 0.000002 : 0.000005, Number(base.learning_rate) * 0.8),
    num_epochs: Math.min(is06b ? 14 : 16, Number(base.num_epochs)),
    subtalker_loss_weight: Math.min(is06b ? 0.32 : 0.3, Number(base.subtalker_loss_weight ?? (is06b ? 0.3 : 0.2)) + 0.02),
    save_every_n_epochs: 1,
  }
}

function describeConfig(config: TrainingConfig): string {
  return `batch=${config.batch_size} epochs=${config.num_epochs} lr=${config.learning_rate} grad_acc=${config.gradient_accumulation_steps ?? 4} subtalker=${config.subtalker_loss_weight ?? 0} seed=${config.seed ?? 0}`
}

export function buildTrainingAdvice(voice: Voice | null, jobs: TrainingJob[]): TrainingAdvice | null {
  if (!voice) return null

  const voiceJobs = [...jobs]
    .filter((job) => job.voice_id === voice.voice_id)
    .sort((a, b) => (Number(b.created_at) || 0) - (Number(a.created_at) || 0))

  const baseConfig = getBaseConfig(voice, voiceJobs)
  if (voiceJobs.length === 0) {
    return {
      mode: 'tone-explore',
      title: 'Start With The Safe Baseline',
      summary: `No training history yet. Start from the conservative preset, then compare before pushing style harder. ${describeConfig(baseConfig)}`,
      confidence: 'medium',
      reasons: [
        'No recent runs exist for this voice yet.',
        'The baseline preset is still the safest first checkpoint search path.',
      ],
      suggestedConfig: baseConfig,
      compareFirst: false,
      reviewDatasetFirst: false,
    }
  }

  const validated = voiceJobs
    .filter((job) => job.summary?.validation_passed === true && getSelectedScore(job) !== null)
    .sort((a, b) => (getSelectedScore(b) ?? -1) - (getSelectedScore(a) ?? -1))
  const rejected = voiceJobs.filter(
    (job) => job.summary?.validation_checked === true && job.summary?.validation_passed !== true,
  )
  const currentPrefix = readText(voice.checkpoint_r2_prefix)
  const currentValidatedJob =
    validated.find((job) => getSelectedPrefix(job) === currentPrefix) ??
    voiceJobs.find((job) => getSelectedPrefix(job) === currentPrefix) ??
    null
  const bestValidatedJob = validated[0] ?? null
  const currentScore = currentValidatedJob ? getSelectedScore(currentValidatedJob) : null
  const bestScore = bestValidatedJob ? getSelectedScore(bestValidatedJob) : null

  const closeAlternatives = validated.filter((job) => {
    const score = getSelectedScore(job)
    if (!bestValidatedJob || score === null || bestScore === null) return false
    if (job.job_id === bestValidatedJob.job_id) return false
    return Math.abs(bestScore - score) <= 0.008
  })

  const recentFailureMessages = rejected
    .slice(0, 4)
    .map((job) => getValidationMessage(job)?.toLowerCase())
    .filter((message): message is string => Boolean(message))

  const asrFailures = recentFailureMessages.filter((message) => message.includes('asr_score') || message.includes('missing asr')).length
  const toneFailures = recentFailureMessages.filter((message) => message.includes('tone_score')).length
  const overallFailures = recentFailureMessages.filter((message) => message.includes('overall_score') || message.includes('quality threshold')).length
  const infraFailures = recentFailureMessages.filter((message) => message.includes('no audio') || message.includes('request does not exist') || message.includes('stalled') || message.includes('recovery')).length

  if (bestValidatedJob && currentValidatedJob && bestScore !== null && currentScore !== null) {
    const gap = bestScore - currentScore
    if (gap >= 0 && gap <= 0.008 && (closeAlternatives.length > 0 || bestValidatedJob.job_id !== currentValidatedJob.job_id)) {
      return {
        mode: 'compare-first',
        title: 'Listen Before Spending Another Run',
        summary: `Validated checkpoints are clustered too tightly to trust the metric alone. Current=${currentScore.toFixed(3)} best=${bestScore.toFixed(3)}. Compare them side by side first.`,
        confidence: 'high',
        reasons: [
          `${validated.length} validated runs exist in the current history.`,
          `Top validated gap is only ${(gap).toFixed(3)}.`,
          `Tone complaints are more likely to need listening judgment than another blind run.`,
        ],
        suggestedConfig: tuneForTone(baseConfig, voice.model_size),
        compareFirst: true,
        reviewDatasetFirst: false,
      }
    }
  }

  if (asrFailures >= 2) {
    return {
      mode: 'dataset-first',
      title: 'Fix Text Alignment Before More Training',
      summary: 'Recent runs are failing mostly on ASR mismatch. More epochs will not recover tone if the transcript/reference path is drifting.',
      confidence: 'high',
      reasons: [
        `${asrFailures} recent failures mention ASR mismatch.`,
        'This usually points to transcript quality, segmentation, or reference-text mismatch.',
        'Review Dataset Studio first, then retry a conservative run.',
      ],
      suggestedConfig: tuneForStability(baseConfig, voice.model_size),
      compareFirst: false,
      reviewDatasetFirst: true,
    }
  }

  if (toneFailures >= 1 || (validated.length > 0 && voice.model_size.includes('1.7'))) {
    const suggestion = tuneForTone(baseConfig, voice.model_size)
    return {
      mode: 'tone-explore',
      title: 'Run A Tone-Preservation Exploration',
      summary: `Speaker match is already in range, but the current history still looks too neutral. Try a lower-LR, lower-subtalker run focused on phrasing retention. ${describeConfig(suggestion)}`,
      confidence: toneFailures >= 2 ? 'high' : 'medium',
      reasons: [
        toneFailures > 0
          ? `${toneFailures} recent failures mention tone loss directly.`
          : 'Validated runs exist, but they are not obviously solving speaking habit retention.',
        'Lower learning rate and slightly lighter subtalker pressure tends to preserve signature phrasing better.',
        'Seed rotation helps surface a different local optimum without changing the dataset.',
      ],
      suggestedConfig: suggestion,
      compareFirst: validated.length >= 2,
      reviewDatasetFirst: false,
    }
  }

  if (overallFailures >= 2 || infraFailures >= 2) {
    const suggestion = tuneForStability(baseConfig, voice.model_size)
    return {
      mode: 'stability-reset',
      title: 'Stabilize Before Pushing Style Again',
      summary: `Recent attempts are failing on overall quality or infra noise. Pull the run back to a more conservative setup first. ${describeConfig(suggestion)}`,
      confidence: 'medium',
      reasons: [
        overallFailures >= 2
          ? `${overallFailures} recent failures mention overall quality threshold.`
          : `${infraFailures} recent failures mention no-audio or provisioning-style noise.`,
        'A conservative rerun is safer than pushing style knobs while the run is unstable.',
      ],
      suggestedConfig: suggestion,
      compareFirst: false,
      reviewDatasetFirst: false,
    }
  }

  return {
    mode: 'hold-current',
    title: 'Current Champion Is Stable',
    summary: 'The current live checkpoint remains the cleanest validated option. Only queue another run if you are explicitly chasing tone, not headline score.',
    confidence: 'medium',
    reasons: [
      currentScore !== null ? `Current live score: ${currentScore.toFixed(3)}.` : 'Current live checkpoint has no recent validation score.',
      validated.length > 0 ? `${validated.length} validated runs were reviewed.` : 'No newer validated run is clearly better yet.',
    ],
    suggestedConfig: tuneForTone(baseConfig, voice.model_size),
    compareFirst: validated.length >= 2,
    reviewDatasetFirst: false,
  }
}
