import { useState, useEffect, useRef } from 'react'
import { Link, useSearchParams } from 'react-router'
import {
  fetchVoices,
  fetchVoiceDatasets,
  fetchTrainingJobs,
  startTraining,
  fetchTrainingJob,
  cancelTrainingJob,
  reconcileTrainingJob,
  revalidateTrainingJob,
  fetchTrainingLogs,
  fetchTrainingLogChunkText,
  type DatasetInfo,
  type TrainingLogChunk,
  type Voice,
  type TrainingJob,
  type TrainingConfig,
  formatDateTime,
  formatDurationMs,
  formatTime,
} from '../lib/api'
import { TrainingAdviceCard } from '../components/TrainingAdviceCard'
import { buildTrainingAdvice } from '../lib/trainingAdvisor'

const ACTIVE_JOB_STATUSES = new Set([
  'pending',
  'running',
  'provisioning',
  'downloading',
  'preprocessing',
  'preparing',
  'training',
  'uploading',
])

function isJobActiveStatus(status: string): boolean {
  return ACTIVE_JOB_STATUSES.has(status)
}

function needsValidationFollowup(job: TrainingJob): boolean {
  return job.status === 'completed' && job.summary?.validation_checked !== true
}

function shouldPollJob(job: TrainingJob): boolean {
  return isJobActiveStatus(job.status) || needsValidationFollowup(job)
}

function getRecommendedTrainingPreset(modelSize: string) {
  if (modelSize.includes('0.6')) {
    return {
      batchSize: 2,
      epochs: 12,
      learningRate: 0.0000025,
      gradientAccumulationSteps: 4,
      subtalkerLossWeight: 0.3,
      saveEveryNEpochs: 1,
      seed: 303,
      gpuTypeId: 'NVIDIA L40S',
    }
  }

  return {
    batchSize: 2,
    epochs: 15,
    learningRate: 0.00002,
    gradientAccumulationSteps: 4,
    subtalkerLossWeight: 0.3,
    saveEveryNEpochs: 5,
    seed: 42,
    gpuTypeId: 'NVIDIA A100-SXM4-80GB',
  }
}

function inferDatasetNameFromRefAudioKey(refAudioKey: string | null | undefined): string | null {
  if (!refAudioKey || !refAudioKey.endsWith('/ref_audio.wav')) {
    return null
  }

  const parts = refAudioKey.split('/').filter(Boolean)
  if (parts.length < 4 || parts[0] !== 'datasets') {
    return null
  }

  return parts[2] || null
}

function getTrainingResetAt(voice: Voice | null | undefined): number | null {
  const raw = voice?.labels?.training_reset_at
  if (!raw) return null
  const parsed = Number(raw)
  return Number.isFinite(parsed) && parsed > 0 ? parsed : null
}

type JobCardMeta = {
  voiceName: string
  voiceRunNumber: number
  globalRunNumber: number
  startedAt: number | null
  finishedAt: number | null
  lastTouchedAt: number | null
  durationMs: number | null
  elapsedMs: number | null
  estimatedFinishAt: number | null
}

function readNumber(value: unknown): number | null {
  const numeric = Number(value)
  return Number.isFinite(numeric) ? numeric : null
}

function readTimestamp(value: unknown): number | null {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value
  }

  if (typeof value === 'string' && value.trim()) {
    const numeric = Number(value)
    if (Number.isFinite(numeric)) {
      return numeric
    }
    const parsed = Date.parse(value)
    return Number.isFinite(parsed) ? parsed : null
  }

  return null
}

function average(values: number[]): number | null {
  if (values.length === 0) return null
  return values.reduce((sum, value) => sum + value, 0) / values.length
}

function getJobStartedAt(job: TrainingJob): number | null {
  return readTimestamp(job.started_at) ?? readTimestamp(job.created_at)
}

function getJobCompletedAt(job: TrainingJob): number | null {
  return readTimestamp(job.completed_at) ?? readTimestamp(job.summary?.completed_at)
}

function getJobDurationMs(job: TrainingJob): number | null {
  const summaryDuration = readNumber(job.summary?.duration_ms)
  if (summaryDuration !== null && summaryDuration >= 0) {
    return summaryDuration
  }

  const startedAt = getJobStartedAt(job)
  const completedAt = getJobCompletedAt(job)
  if (startedAt === null || completedAt === null) {
    return null
  }

  return Math.max(0, completedAt - startedAt)
}

function getJobProgressRatio(job: TrainingJob): number | null {
  const totalSteps = readNumber(job.progress.total_steps)
  const step = readNumber(job.progress.step)
  if (totalSteps !== null && totalSteps > 0 && step !== null && step >= 0) {
    return Math.min(1, Math.max(0, step / totalSteps))
  }

  const totalEpochs = readNumber(job.progress.total_epochs)
  const epoch = readNumber(job.progress.epoch)
  if (totalEpochs !== null && totalEpochs > 0 && epoch !== null && epoch >= 0) {
    return Math.min(1, Math.max(0, epoch / totalEpochs))
  }

  return null
}

function buildJobCardMeta(jobs: TrainingJob[], voices: Voice[], now: number): Map<string, JobCardMeta> {
  const voiceNames = new Map(voices.map((voice) => [voice.voice_id, voice.name]))
  const durationBucketsByVoiceModel = new Map<string, number[]>()
  const durationBucketsByModel = new Map<string, number[]>()

  for (const job of jobs) {
    const durationMs = getJobDurationMs(job)
    if (durationMs === null || durationMs <= 0) continue
    const modelSize = job.config.model_size ?? '1.7B'
    const voiceModelKey = `${job.voice_id}::${modelSize}`
    const byVoiceModel = durationBucketsByVoiceModel.get(voiceModelKey) ?? []
    byVoiceModel.push(durationMs)
    durationBucketsByVoiceModel.set(voiceModelKey, byVoiceModel)

    const byModel = durationBucketsByModel.get(modelSize) ?? []
    byModel.push(durationMs)
    durationBucketsByModel.set(modelSize, byModel)
  }

  const voiceCounts = new Map<string, number>()
  const result = new Map<string, JobCardMeta>()
  const jobsByCreatedAt = [...jobs].sort(
    (a, b) => (readTimestamp(a.created_at) ?? 0) - (readTimestamp(b.created_at) ?? 0),
  )

  jobsByCreatedAt.forEach((job, index) => {
    const voiceRunNumber = (voiceCounts.get(job.voice_id) ?? 0) + 1
    voiceCounts.set(job.voice_id, voiceRunNumber)

    const startedAt = getJobStartedAt(job)
    const finishedAt = getJobCompletedAt(job)
    const durationMs = getJobDurationMs(job)
    const lastTouchedAt =
      finishedAt ??
      readTimestamp(job.last_heartbeat_at) ??
      readTimestamp(job.updated_at) ??
      startedAt
    const elapsedMs =
      startedAt !== null
        ? Math.max(0, (finishedAt ?? lastTouchedAt ?? now) - startedAt)
        : null
    const modelSize = job.config.model_size ?? '1.7B'
    const voiceModelKey = `${job.voice_id}::${modelSize}`
    const averageDurationMs =
      average(durationBucketsByVoiceModel.get(voiceModelKey) ?? []) ??
      average(durationBucketsByModel.get(modelSize) ?? [])
    const progressRatio = getJobProgressRatio(job)

    let estimatedFinishAt: number | null = null
    if (isJobActiveStatus(job.status) && startedAt !== null) {
      const progressEstimateMs =
        progressRatio !== null && progressRatio > 0.03 && elapsedMs !== null
          ? elapsedMs / progressRatio
          : null
      const totalEstimateMs =
        progressEstimateMs !== null && averageDurationMs !== null
          ? Math.max(elapsedMs ?? 0, Math.round((progressEstimateMs + averageDurationMs) / 2))
          : progressEstimateMs !== null
            ? Math.max(elapsedMs ?? 0, Math.round(progressEstimateMs))
            : averageDurationMs
      estimatedFinishAt = totalEstimateMs !== null ? startedAt + totalEstimateMs : null
    }

    result.set(job.job_id, {
      voiceName: voiceNames.get(job.voice_id) ?? job.voice_id,
      voiceRunNumber,
      globalRunNumber: index + 1,
      startedAt,
      finishedAt,
      lastTouchedAt,
      durationMs,
      elapsedMs,
      estimatedFinishAt,
    })
  })

  return result
}

export function Training() {
  const [searchParams] = useSearchParams()
  const [voices, setVoices] = useState<Voice[]>([])
  const [loadingVoices, setLoadingVoices] = useState(true)

  // Form state
  const [selectedVoiceId, setSelectedVoiceId] = useState('')
  const [batchSize, setBatchSize] = useState(4)
  const [epochs, setEpochs] = useState(8)
  const [learningRate, setLearningRate] = useState(0.00001)
  const [trainingLanguage, setTrainingLanguage] = useState('ko')
  const [trainingSeed, setTrainingSeed] = useState(77)
  const [gradientAccumulationSteps, setGradientAccumulationSteps] = useState(4)
  const [subtalkerLossWeight, setSubtalkerLossWeight] = useState(0.3)
  const [saveEveryNEpochs, setSaveEveryNEpochs] = useState(1)
  const [gpuTypeId, setGpuTypeId] = useState('NVIDIA A100-SXM4-80GB')
  const [availableDatasets, setAvailableDatasets] = useState<DatasetInfo[]>([])
  const [loadingDatasets, setLoadingDatasets] = useState(false)
  const [selectedDatasetName, setSelectedDatasetName] = useState('')
  const [starting, setStarting] = useState(false)
  const [formError, setFormError] = useState('')

  // Jobs
  const [jobs, setJobs] = useState<TrainingJob[]>([])
  const [loadingJobs, setLoadingJobs] = useState(false)
  const jobsRef = useRef<TrainingJob[]>([])
  const requestedVoiceId = searchParams.get('voiceId') ?? ''
  const requestedDatasetName = searchParams.get('datasetName') ?? ''

  // Keep ref in sync
  useEffect(() => {
    jobsRef.current = jobs
  }, [jobs])

  // Load voices
  useEffect(() => {
    let cancelled = false

    async function load() {
      try {
        const data = await fetchVoices()
        if (cancelled) return
        setVoices(data.voices)
        if (data.voices.length > 0) {
          const requestedVoice = requestedVoiceId
            ? data.voices.find((voice) => voice.voice_id === requestedVoiceId)
            : null
          if (requestedVoice) {
            setSelectedVoiceId(requestedVoice.voice_id)
          } else if (!selectedVoiceId) {
            setSelectedVoiceId(data.voices[0].voice_id)
          }
        }
      } catch {
        // silently fail
      } finally {
        if (!cancelled) setLoadingVoices(false)
      }
    }

    load()
    return () => { cancelled = true }
  }, [requestedVoiceId, selectedVoiceId])

  useEffect(() => {
    let cancelled = false

    async function loadJobs() {
      setLoadingJobs(true)
      try {
        const data = await fetchTrainingJobs(undefined, 100)
        if (cancelled) return
        setJobs(data.jobs)
      } catch {
        // silently fail
      } finally {
        if (!cancelled) setLoadingJobs(false)
      }
    }

    loadJobs()
    return () => {
      cancelled = true
    }
  }, [])

  // Poll active jobs every 5 seconds
  const hasActiveJobs = jobs.some((j) => shouldPollJob(j))
  const selectedVoice = voices.find((voice) => voice.voice_id === selectedVoiceId)
  const selectedPreset = selectedVoice
    ? getRecommendedTrainingPreset(selectedVoice.model_size)
    : null
  const linkedDatasetName = inferDatasetNameFromRefAudioKey(selectedVoice?.ref_audio_r2_key)
  const effectiveDatasetName = selectedDatasetName || linkedDatasetName
  const selectedVoiceResetAt = getTrainingResetAt(selectedVoice)
  const selectedVoiceJobs = jobs
    .filter((job) => job.voice_id === selectedVoiceId)
    .filter((job) => selectedVoiceResetAt === null || job.created_at >= selectedVoiceResetAt)
  const trainingAdvice = buildTrainingAdvice(selectedVoice ?? null, selectedVoiceJobs)

  useEffect(() => {
    if (!selectedVoice) return
    const preset = getRecommendedTrainingPreset(selectedVoice.model_size)
    setBatchSize(preset.batchSize)
    setEpochs(preset.epochs)
    setLearningRate(preset.learningRate)
    setTrainingSeed(preset.seed)
    setGradientAccumulationSteps(preset.gradientAccumulationSteps)
    setSubtalkerLossWeight(preset.subtalkerLossWeight)
    setSaveEveryNEpochs(preset.saveEveryNEpochs)
    setGpuTypeId(preset.gpuTypeId)
  }, [selectedVoice?.voice_id, selectedVoice?.model_size])

  useEffect(() => {
    if (!selectedVoice) {
      setAvailableDatasets([])
      setSelectedDatasetName('')
      return
    }

    const voice = selectedVoice
    let cancelled = false
    async function loadDatasets() {
      setLoadingDatasets(true)
      try {
        const data = await fetchVoiceDatasets(voice.voice_id)
        if (cancelled) return
        setAvailableDatasets(data.datasets)
        const linked = inferDatasetNameFromRefAudioKey(voice.ref_audio_r2_key)
        const fallback = data.datasets[0]?.name ?? ''
        if (requestedDatasetName && data.datasets.some((dataset) => dataset.name === requestedDatasetName)) {
          setSelectedDatasetName(requestedDatasetName)
        } else {
          setSelectedDatasetName(linked ?? fallback)
        }
      } catch {
        if (!cancelled) {
          setAvailableDatasets([])
          setSelectedDatasetName(
            requestedDatasetName || inferDatasetNameFromRefAudioKey(voice.ref_audio_r2_key) || '',
          )
        }
      } finally {
        if (!cancelled) setLoadingDatasets(false)
      }
    }

    loadDatasets()
    return () => {
      cancelled = true
    }
  }, [selectedVoice, requestedDatasetName])

  useEffect(() => {
    if (!hasActiveJobs) return

    const interval = setInterval(async () => {
      const currentJobs = jobsRef.current
      const activeJobIds = currentJobs
        .filter((j) => shouldPollJob(j))
        .map((j) => j.job_id)

      if (activeJobIds.length === 0) return

      try {
        const updates = await Promise.all(
          activeJobIds.map((id) => fetchTrainingJob(id)),
        )
        setJobs((prev) =>
          prev.map((j) => {
            const updated = updates.find((u) => u.job_id === j.job_id)
            return updated ?? j
          }),
        )
      } catch {
        // silently ignore polling errors
      }
    }, 5000)

    return () => clearInterval(interval)
  }, [hasActiveJobs])

  async function handleStartTraining(e: { preventDefault: () => void }) {
    e.preventDefault()
    if (!selectedVoiceId || !selectedVoice) return

    setStarting(true)
    setFormError('')

    const preset = getRecommendedTrainingPreset(selectedVoice.model_size)
    const config: TrainingConfig = {
      batch_size: batchSize,
      num_epochs: epochs,
      learning_rate: learningRate,
      model_size: selectedVoice.model_size,
      gradient_accumulation_steps: gradientAccumulationSteps,
      subtalker_loss_weight: subtalkerLossWeight,
      save_every_n_epochs: saveEveryNEpochs,
      seed: trainingSeed,
      whisper_language: trainingLanguage,
      gpu_type_id: gpuTypeId,
    }

    try {
      const result = await startTraining(selectedVoiceId, config, {
        datasetName: effectiveDatasetName || undefined,
      })

      // Add the new job to the list
      const newJob: TrainingJob = {
        job_id: result.job_id,
        status: result.status as TrainingJob['status'],
        voice_id: selectedVoiceId,
        created_at: Date.now(),
        last_heartbeat_at: null,
        summary: {},
        metrics: {},
        config,
        progress: {
          epoch: 0,
          total_epochs: epochs,
          loss: 0,
          step: 0,
          total_steps: 0,
        },
      }
      setJobs((prev) => [newJob, ...prev])
    } catch (err) {
      setFormError(err instanceof Error ? err.message : 'Failed to start training')
    } finally {
      setStarting(false)
    }
  }

  async function handleCancel(jobId: string) {
    try {
      await cancelTrainingJob(jobId)
      setJobs((prev) =>
        prev.map((j) =>
          j.job_id === jobId ? { ...j, status: 'cancelled' as const } : j,
        ),
      )
    } catch {
      // silently fail
    }
  }

  function replaceJob(updated: TrainingJob) {
    setJobs((prev) => {
      const exists = prev.some((job) => job.job_id === updated.job_id)
      if (!exists) {
        return [updated, ...prev]
      }
      return prev.map((job) => (job.job_id === updated.job_id ? updated : job))
    })
  }

  async function handleRefreshAllJobs() {
    const data = await fetchTrainingJobs(undefined, 100)
    setJobs(data.jobs)
  }

  async function handleRefreshJob(jobId: string) {
    const updated = await fetchTrainingJob(jobId)
    replaceJob(updated)
  }

  async function handleReconcile(jobId: string) {
    await reconcileTrainingJob(jobId)
    await handleRefreshJob(jobId)
  }

  async function handleRevalidate(jobId: string) {
    const response = await revalidateTrainingJob(jobId)
    replaceJob(response.job)
  }

  function applySuggestedConfig(config: TrainingConfig) {
    setBatchSize(config.batch_size)
    setEpochs(config.num_epochs)
    setLearningRate(config.learning_rate)
    setTrainingSeed(config.seed ?? 42)
    setGradientAccumulationSteps(config.gradient_accumulation_steps ?? 4)
    setSubtalkerLossWeight(
      config.subtalker_loss_weight ?? (config.model_size?.includes('0.6') ? 0.3 : 0.2),
    )
    setSaveEveryNEpochs(config.save_every_n_epochs ?? 1)
    setTrainingLanguage(config.whisper_language ?? 'ko')
    setGpuTypeId(config.gpu_type_id ?? '')
  }

  const jobsByRecency = [...jobs].sort(
    (a, b) => (readTimestamp(b.created_at) ?? 0) - (readTimestamp(a.created_at) ?? 0),
  )
  const jobCardMeta = buildJobCardMeta(jobsByRecency, voices, Date.now())
  const activeJobs = jobsByRecency.filter((j) => shouldPollJob(j))
  const completedJobs = jobsByRecency.filter(
    (j) => !shouldPollJob(j) && (j.status === 'completed' || j.status === 'failed' || j.status === 'cancelled'),
  )

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h1 className="text-heading text-2xl font-bold">Training</h1>
            <p className="text-subtle text-sm mt-1">Fine-tune voice models</p>
          </div>
          <button
            onClick={() => { void handleRefreshAllJobs() }}
            className="inline-flex items-center rounded-lg border border-edge px-3 py-2 text-[11px] font-semibold text-primary transition-colors hover:border-accent hover:text-accent"
            type="button"
          >
            Refresh Jobs
          </button>
        </div>
      </div>

      <div className="grid lg:grid-cols-[380px_1fr] gap-6">
        {/* Start Training Form */}
        <div className="bg-raised border border-edge rounded-xl p-5">
          <h2 className="text-heading font-semibold text-sm mb-5">Start Training</h2>

          <div className="mb-4 rounded-lg border border-edge bg-surface px-3 py-2.5">
            <p className="text-subtle text-xs">
              Upload a real training set on the Voices page first. The defaults here are conservative and tuned for voice similarity over aggressive overfitting.
            </p>
            {selectedVoice && selectedPreset && (
              <p className="text-muted text-[10px] font-mono mt-2">
                model={selectedVoice.model_size} dataset={effectiveDatasetName || 'root-prefix-fallback'} batch={batchSize} epochs={epochs} lr={learningRate} seed={trainingSeed}
              </p>
            )}
            {selectedVoice && !effectiveDatasetName && (
              <p className="text-warning text-[10px] font-mono mt-2">
                No finalized dataset linked on this voice. Training will fall back to the root dataset prefix.
              </p>
            )}
            <Link
              to="/voices"
              className="inline-flex items-center gap-1 mt-2 text-accent text-xs font-medium hover:text-accent-light"
            >
              Go to Voices upload
            </Link>
          </div>

          {selectedVoice && (
            <div className="mb-4">
              <TrainingAdviceCard
                voiceId={selectedVoice.voice_id}
                advice={trainingAdvice}
                onApplyConfig={applySuggestedConfig}
                showTrainingLink={false}
              />
            </div>
          )}

          {voices.length === 0 && !loadingVoices && (
            <div className="mb-4 rounded-lg border border-warning/20 bg-warning-dim px-3 py-2 text-warning text-xs">
              No voices found yet. Create a voice and upload audio first.
            </div>
          )}

          <form onSubmit={handleStartTraining} className="space-y-4">
            {/* Voice */}
            <div>
              <label htmlFor="training-voice" className="text-subtle text-xs font-medium mb-1.5 block">Voice</label>
              {loadingVoices ? (
                <div className="h-10 bg-surface rounded-lg animate-pulse" />
              ) : (
                <select
                  id="training-voice"
                  value={selectedVoiceId}
                  onChange={(e) => setSelectedVoiceId(e.target.value)}
                  className="w-full bg-surface border border-edge rounded-lg px-3 py-2.5 text-sm text-primary focus:border-accent transition-colors appearance-none"
                >
                  {voices.length === 0 && (
                    <option value="">No voices available</option>
                  )}
                  {voices.map((v) => (
                    <option key={v.voice_id} value={v.voice_id}>
                      {v.name}
                    </option>
                  ))}
                </select>
              )}
            </div>

            <div>
              <label htmlFor="training-dataset" className="text-subtle text-xs font-medium mb-1.5 block">Dataset</label>
              {loadingDatasets ? (
                <div className="h-10 bg-surface rounded-lg animate-pulse" />
              ) : availableDatasets.length > 0 ? (
                <select
                  id="training-dataset"
                  value={selectedDatasetName}
                  onChange={(e) => setSelectedDatasetName(e.target.value)}
                  className="w-full bg-surface border border-edge rounded-lg px-3 py-2.5 text-sm text-primary focus:border-accent transition-colors appearance-none"
                >
                  {availableDatasets.map((dataset) => (
                    <option key={dataset.name} value={dataset.name}>
                      {dataset.name}
                    </option>
                  ))}
                </select>
              ) : (
                <div className="rounded-lg border border-edge bg-surface px-3 py-2.5 text-sm text-muted">
                  No finalized dataset found. Use Dataset Studio first.
                </div>
              )}
            </div>

            {/* Batch Size */}
            <div>
              <label htmlFor="training-batch-size" className="text-subtle text-xs font-medium mb-1.5 block">Batch Size</label>
              <input
                id="training-batch-size"
                type="number"
                value={batchSize}
                onChange={(e) => setBatchSize(parseInt(e.target.value) || 1)}
                min={1}
                max={32}
                className="w-full bg-surface border border-edge rounded-lg px-3 py-2.5 text-sm text-primary font-mono focus:border-accent transition-colors"
              />
            </div>

            {/* Epochs */}
            <div>
              <label htmlFor="training-epochs" className="text-subtle text-xs font-medium mb-1.5 block">Epochs</label>
              <input
                id="training-epochs"
                type="number"
                value={epochs}
                onChange={(e) => setEpochs(parseInt(e.target.value) || 1)}
                min={1}
                max={50}
                className="w-full bg-surface border border-edge rounded-lg px-3 py-2.5 text-sm text-primary font-mono focus:border-accent transition-colors"
              />
              <p className="text-muted text-[10px] font-mono mt-1">Recommended: 5–15 (more epochs = higher similarity, risk of overfitting)</p>
            </div>

            {/* Learning Rate */}
            <div>
              <label htmlFor="training-learning-rate" className="text-subtle text-xs font-medium mb-1.5 block">Learning Rate</label>
              <input
                id="training-learning-rate"
                type="text"
                value={learningRate}
                onChange={(e) => {
                  const val = parseFloat(e.target.value)
                  if (!isNaN(val)) setLearningRate(val)
                }}
                className="w-full bg-surface border border-edge rounded-lg px-3 py-2.5 text-sm text-primary font-mono focus:border-accent transition-colors"
              />
              <p className="text-muted text-[10px] font-mono mt-1">Recommended: 1e-5 to 5e-5</p>
            </div>

            {/* Language */}
            <div>
              <label htmlFor="training-language" className="text-subtle text-xs font-medium mb-1.5 block">Language</label>
              <select
                id="training-language"
                value={trainingLanguage}
                onChange={(e) => setTrainingLanguage(e.target.value)}
                className="w-full bg-surface border border-edge rounded-lg px-3 py-2.5 text-sm text-primary focus:border-accent transition-colors appearance-none"
              >
                <option value="ko">Korean</option>
                <option value="en">English</option>
                <option value="ja">Japanese</option>
                <option value="zh">Chinese</option>
              </select>
            </div>

            <div>
              <label htmlFor="training-seed" className="text-subtle text-xs font-medium mb-1.5 block">Seed</label>
              <input
                id="training-seed"
                type="number"
                value={trainingSeed}
                onChange={(e) => setTrainingSeed(parseInt(e.target.value, 10) || 1)}
                className="w-full bg-surface border border-edge rounded-lg px-3 py-2.5 text-sm text-primary font-mono focus:border-accent transition-colors"
              />
            </div>

            <div>
              <label htmlFor="training-grad-acc" className="text-subtle text-xs font-medium mb-1.5 block">Gradient Accumulation</label>
              <input
                id="training-grad-acc"
                type="number"
                min={1}
                value={gradientAccumulationSteps}
                onChange={(e) => setGradientAccumulationSteps(parseInt(e.target.value, 10) || 1)}
                className="w-full bg-surface border border-edge rounded-lg px-3 py-2.5 text-sm text-primary font-mono focus:border-accent transition-colors"
              />
            </div>

            <div>
              <label htmlFor="training-subtalker" className="text-subtle text-xs font-medium mb-1.5 block">Subtalker Loss Weight</label>
              <input
                id="training-subtalker"
                type="number"
                step="0.05"
                value={subtalkerLossWeight}
                onChange={(e) => setSubtalkerLossWeight(parseFloat(e.target.value) || 0)}
                className="w-full bg-surface border border-edge rounded-lg px-3 py-2.5 text-sm text-primary font-mono focus:border-accent transition-colors"
              />
            </div>

            <div>
              <label htmlFor="training-save-every" className="text-subtle text-xs font-medium mb-1.5 block">Save Every N Epochs</label>
              <input
                id="training-save-every"
                type="number"
                min={1}
                value={saveEveryNEpochs}
                onChange={(e) => setSaveEveryNEpochs(parseInt(e.target.value, 10) || 1)}
                className="w-full bg-surface border border-edge rounded-lg px-3 py-2.5 text-sm text-primary font-mono focus:border-accent transition-colors"
              />
            </div>

            <div>
              <label htmlFor="training-gpu" className="text-subtle text-xs font-medium mb-1.5 block">GPU Type</label>
              <input
                id="training-gpu"
                type="text"
                value={gpuTypeId}
                onChange={(e) => setGpuTypeId(e.target.value)}
                className="w-full bg-surface border border-edge rounded-lg px-3 py-2.5 text-sm text-primary font-mono focus:border-accent transition-colors"
              />
            </div>

            {/* Error */}
            {formError && (
              <div className="bg-error-dim border border-error/20 rounded-lg px-3 py-2 text-error text-xs">
                {formError}
              </div>
            )}

            {/* Submit */}
            <button
              type="submit"
              disabled={!selectedVoiceId || starting || voices.length === 0}
              className="w-full bg-accent hover:bg-accent-light text-void font-semibold text-sm py-2.5 rounded-lg disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            >
              {starting ? 'Starting...' : 'Start Training'}
            </button>
          </form>
        </div>

        {/* Jobs Panel */}
        <div className="space-y-6">
          {/* Active Jobs */}
          <div className="bg-raised border border-edge rounded-xl p-5">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-heading font-semibold text-sm">
                Active Jobs
                {activeJobs.length > 0 && (
                  <span className="ml-2 text-accent font-mono text-xs">({activeJobs.length})</span>
                )}
              </h2>
              {hasActiveJobs && (
                <span className="inline-flex items-center gap-1.5 text-accent text-[10px] font-mono">
                  <span className="w-1.5 h-1.5 rounded-full bg-accent animate-pulse-glow" />
                  POLLING
                </span>
              )}
            </div>

            {activeJobs.length === 0 ? (
              <p className="text-muted text-sm text-center py-8">No active or validating training jobs</p>
            ) : (
              <div className="space-y-3">
                {activeJobs.map((job) => (
                  <JobCard
                    key={job.job_id}
                    job={job}
                    meta={jobCardMeta.get(job.job_id)}
                    onCancel={handleCancel}
                    onRefresh={handleRefreshJob}
                    onReconcile={handleReconcile}
                    onRevalidate={handleRevalidate}
                  />
                ))}
              </div>
            )}
          </div>

          {completedJobs.length > 0 && (
            <div className="bg-raised border border-edge rounded-xl p-5">
              <h2 className="text-heading font-semibold text-sm mb-4">
                History
                <span className="ml-2 text-muted font-mono text-xs">({completedJobs.length})</span>
              </h2>
              <div className="space-y-3">
                {completedJobs.map((job) => (
                  <JobCard
                    key={job.job_id}
                    job={job}
                    meta={jobCardMeta.get(job.job_id)}
                    onCancel={handleCancel}
                    onRefresh={handleRefreshJob}
                    onReconcile={handleReconcile}
                    onRevalidate={handleRevalidate}
                  />
                ))}
              </div>
            </div>
          )}

          {/* Empty state */}
          {jobs.length === 0 && !loadingJobs && (
            <div className="text-center py-12">
              <div className="w-16 h-16 mx-auto mb-4 rounded-2xl bg-raised border border-edge flex items-center justify-center">
                <svg className="w-8 h-8 text-muted" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                  <title>Training jobs icon</title>
                  <polyline points="22,12 18,12 15,21 9,3 6,12 2,12" />
                </svg>
              </div>
              <h3 className="text-heading font-semibold mb-1">No training jobs</h3>
              <p className="text-subtle text-sm">Start a training job to fine-tune a voice model</p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

// ── Job Card ───────────────────────────────────────────────────────────────────

function JobCard({
  job,
  meta,
  onCancel,
  onRefresh,
  onReconcile,
  onRevalidate,
}: {
  job: TrainingJob
  meta?: JobCardMeta
  onCancel: (jobId: string) => void
  onRefresh: (jobId: string) => Promise<void>
  onReconcile: (jobId: string) => Promise<void>
  onRevalidate: (jobId: string) => Promise<void>
}) {
  const [busyAction, setBusyAction] = useState<'refresh' | 'reconcile' | 'revalidate' | ''>('')
  const [actionError, setActionError] = useState('')
  const [showLogs, setShowLogs] = useState(false)
  const isActive = isJobActiveStatus(job.status)
  const isValidationPending = needsValidationFollowup(job)
  const isPolling = isActive || isValidationPending
  const epoch = typeof job.progress.epoch === 'number' ? job.progress.epoch : 0
  const totalEpochs = typeof job.progress.total_epochs === 'number' ? job.progress.total_epochs : 0
  const loss = typeof job.progress.loss === 'number' ? job.progress.loss : null
  const step = typeof job.progress.step === 'number' ? job.progress.step : 0
  const progressPct =
    totalEpochs > 0
      ? (epoch / totalEpochs) * 100
      : 0
  const durationMs = typeof job.summary?.duration_ms === 'number' ? job.summary.duration_ms : null
  const finalLoss = typeof job.summary?.final_loss === 'number' ? job.summary.final_loss : null
  const finalEpoch = typeof job.summary?.final_epoch === 'number' ? job.summary.final_epoch : null
  const summaryTotalEpochs = typeof job.summary?.total_epochs === 'number' ? job.summary.total_epochs : null
  const selectedCheckpointEpoch = typeof job.summary?.selected_checkpoint_epoch === 'number'
    ? job.summary.selected_checkpoint_epoch
    : null
  const evaluatedCheckpoints = Array.isArray(job.summary?.evaluated_checkpoints)
    ? job.summary?.evaluated_checkpoints
    : []
  const canCompareCheckpoints = evaluatedCheckpoints.length > 0
  const validationPassed = job.summary?.validation_passed === true
  const validationChecked = job.summary?.validation_checked === true
  const validationRejected = validationChecked && !validationPassed && evaluatedCheckpoints.length > 0
  const statusLabel = isValidationPending ? 'validating' : validationRejected ? 'rejected' : job.status
  const validationMessage = typeof job.summary?.validation_message === 'string'
    ? job.summary.validation_message
    : null
  const lastMessage = typeof job.summary?.last_message === 'string'
    ? job.summary.last_message
    : null
  const startedAt = meta?.startedAt ?? readTimestamp(job.created_at)
  const finishedAt = meta?.finishedAt ?? getJobCompletedAt(job)
  const lastTouchedAt = meta?.lastTouchedAt ?? readTimestamp(job.updated_at)
  const timelineLabel = isActive
    ? 'ETA'
    : isValidationPending
      ? 'Train Done'
      : 'Finished'
  const timelineValue = isActive
    ? meta?.estimatedFinishAt ?? null
    : isValidationPending
      ? finishedAt
      : finishedAt

  const statusStyles: Record<string, { bg: string; text: string }> = {
    pending: { bg: 'bg-warning-dim', text: 'text-warning' },
    running: { bg: 'bg-accent-dim', text: 'text-accent' },
    provisioning: { bg: 'bg-warning-dim', text: 'text-warning' },
    downloading: { bg: 'bg-accent-dim', text: 'text-accent' },
    preprocessing: { bg: 'bg-accent-dim', text: 'text-accent' },
    preparing: { bg: 'bg-accent-dim', text: 'text-accent' },
    training: { bg: 'bg-accent-dim', text: 'text-accent' },
    uploading: { bg: 'bg-accent-dim', text: 'text-accent' },
    completed: { bg: 'bg-accent-dim', text: 'text-accent' },
    validating: { bg: 'bg-accent-dim', text: 'text-accent' },
    rejected: { bg: 'bg-warning-dim', text: 'text-warning' },
    failed: { bg: 'bg-error-dim', text: 'text-error' },
    cancelled: { bg: 'bg-raised', text: 'text-muted' },
  }

  const style = statusStyles[statusLabel] ?? statusStyles.pending

  async function runAction(kind: 'refresh' | 'reconcile' | 'revalidate', action: () => Promise<void>) {
    setBusyAction(kind)
    setActionError('')
    try {
      await action()
    } catch (err) {
      setActionError(err instanceof Error ? err.message : 'Action failed')
    } finally {
      setBusyAction('')
    }
  }

  return (
    <div className="bg-surface rounded-lg p-4">
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div>
          <div className="text-primary text-sm font-semibold">
            {meta?.voiceName ?? job.voice_id}
          </div>
          <div className="mt-1 flex flex-wrap gap-x-3 gap-y-1 text-[10px] font-mono text-muted">
            <span>run #{meta?.voiceRunNumber ?? '—'}</span>
            <span>global #{meta?.globalRunNumber ?? '—'}</span>
            <span>{job.job_id.slice(0, 12)}</span>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-mono uppercase tracking-wider ${style.bg} ${style.text}`}>
            {statusLabel}
          </span>
          {isActive && (
            <button
              onClick={() => onCancel(job.job_id)}
              className="text-muted hover:text-error text-xs font-medium transition-colors"
              type="button"
            >
              Cancel
            </button>
          )}
        </div>
      </div>

      {/* Progress Bar */}
      {isPolling && (
        <div className="mb-3">
          <div className="w-full h-2 bg-edge rounded-full overflow-hidden">
            <div
              className="h-full bg-accent rounded-full transition-[width] duration-500"
              style={{ width: `${isValidationPending ? 100 : progressPct}%` }}
            />
          </div>
        </div>
      )}

      <div className="mb-3 flex flex-wrap gap-x-4 gap-y-1 text-[10px] font-mono text-muted">
        <span>created={formatDateTime(job.created_at)}</span>
        {startedAt !== null && <span>started={formatTime(startedAt)}</span>}
        {timelineValue !== null && <span>{timelineLabel.toLowerCase()}={formatTime(timelineValue)}</span>}
        {meta && meta.elapsedMs !== null && isPolling && <span>elapsed={formatDurationMs(meta.elapsedMs)}</span>}
        {meta && meta.durationMs !== null && !isPolling && <span>duration={formatDurationMs(meta.durationMs)}</span>}
        {lastTouchedAt !== null && <span>updated={formatTime(lastTouchedAt)}</span>}
      </div>

      {/* Metrics */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <Metric label="Run" value={meta ? `#${meta.voiceRunNumber}` : '—'} />
        <Metric label="Epoch" value={`${epoch}/${totalEpochs || '—'}`} />
        <Metric label="Step" value={String(step)} />
        <Metric label="Loss" value={loss !== null && loss > 0 ? loss.toFixed(4) : '—'} />
      </div>

      {job.status === 'completed' && (durationMs !== null || finalLoss !== null || finalEpoch !== null) && (
        <div className="mt-3 grid grid-cols-3 gap-3">
          <Metric label="Duration" value={formatDurationMs(durationMs)} />
          <Metric label="Final Loss" value={finalLoss !== null ? finalLoss.toFixed(4) : '—'} />
          <Metric label="Epochs" value={finalEpoch !== null ? `${finalEpoch}/${summaryTotalEpochs ?? '—'}` : '—'} />
        </div>
      )}

      {(validationChecked || selectedCheckpointEpoch !== null) && (
        <div className="mt-3 grid grid-cols-2 gap-3">
          <Metric
            label="Validation"
            value={
              validationPassed
                ? 'passed'
                : validationRejected
                  ? 'rejected'
                  : validationChecked
                    ? 'failed'
                  : 'pending'
            }
          />
          <Metric
            label="Champion"
            value={selectedCheckpointEpoch !== null ? `epoch ${selectedCheckpointEpoch}` : '—'}
          />
        </div>
      )}

      {(validationMessage || lastMessage) && (
        <div className="mt-3 rounded-lg border border-edge px-3 py-2 text-[11px] text-subtle">
          {validationMessage ?? lastMessage}
        </div>
      )}

      {actionError && (
        <div className="mt-3 rounded-lg border border-error/20 bg-error-dim px-3 py-2 text-error text-xs">
          {actionError}
        </div>
      )}

      {(validationPassed || canCompareCheckpoints) && (
        <div className="mt-3 flex flex-wrap gap-2">
          {validationPassed && (
            <Link
              to={`/voices/${job.voice_id}`}
              className="inline-flex items-center rounded-lg bg-accent px-3 py-2 text-[11px] font-semibold text-void transition-colors hover:bg-accent-light"
            >
              Open Voice
            </Link>
          )}
          <Link
            to={`/voices/${job.voice_id}/compare?job=${job.job_id}`}
            className="inline-flex items-center rounded-lg border border-edge px-3 py-2 text-[11px] font-semibold text-primary transition-colors hover:border-accent hover:text-accent"
          >
            {validationRejected ? 'Review Rejected Checkpoints' : 'Compare Checkpoints'}
          </Link>
          {validationPassed && (
            <Link
              to={`/playground?voice=${job.voice_id}`}
              className="inline-flex items-center rounded-lg border border-edge px-3 py-2 text-[11px] font-semibold text-primary transition-colors hover:border-accent hover:text-accent"
            >
              Generate Sample
            </Link>
          )}
        </div>
      )}

      <div className="mt-3 flex flex-wrap gap-2">
        <button
          onClick={() => { void runAction('refresh', () => onRefresh(job.job_id)) }}
          disabled={busyAction !== ''}
          className="inline-flex items-center rounded-lg border border-edge px-3 py-2 text-[11px] font-semibold text-primary transition-colors hover:border-accent hover:text-accent disabled:opacity-50"
          type="button"
        >
          {busyAction === 'refresh' ? 'Refreshing…' : 'Refresh'}
        </button>
        {isPolling && (
          <button
            onClick={() => { void runAction('reconcile', () => onReconcile(job.job_id)) }}
            disabled={busyAction !== ''}
            className="inline-flex items-center rounded-lg border border-edge px-3 py-2 text-[11px] font-semibold text-primary transition-colors hover:border-accent hover:text-accent disabled:opacity-50"
            type="button"
          >
            {busyAction === 'reconcile' ? 'Advancing…' : 'Advance Status'}
          </button>
        )}
        {(job.status === 'completed' || job.status === 'failed') && (
          <button
            onClick={() => { void runAction('revalidate', () => onRevalidate(job.job_id)) }}
            disabled={busyAction !== ''}
            className="inline-flex items-center rounded-lg border border-edge px-3 py-2 text-[11px] font-semibold text-primary transition-colors hover:border-accent hover:text-accent disabled:opacity-50"
            type="button"
          >
            {busyAction === 'revalidate' ? 'Restarting…' : 'Revalidate'}
          </button>
        )}
        <button
          onClick={() => setShowLogs(true)}
          className="inline-flex items-center rounded-lg border border-edge px-3 py-2 text-[11px] font-semibold text-primary transition-colors hover:border-accent hover:text-accent"
          type="button"
        >
          View Logs
        </button>
      </div>

      {/* Config */}
      <div className="mt-3 pt-3 border-t border-edge flex gap-4 text-muted text-[10px] font-mono">
        <span>model={job.config.model_size ?? '1.7B'}</span>
        <span>batch={job.config.batch_size}</span>
        <span>epochs={job.config.num_epochs}</span>
        <span>lr={job.config.learning_rate}</span>
        {typeof job.config.seed === 'number' && <span>seed={job.config.seed}</span>}
        {job.config.gpu_type_id && <span>gpu={job.config.gpu_type_id}</span>}
      </div>

      {showLogs && (
        <JobLogsModal
          job={job}
          onClose={() => setShowLogs(false)}
        />
      )}
    </div>
  )
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-muted text-[10px] font-mono uppercase tracking-wider mb-0.5">
        {label}
      </div>
      <div className="text-primary text-sm font-mono tabular-nums">{value}</div>
    </div>
  )
}

function JobLogsModal({
  job,
  onClose,
}: {
  job: TrainingJob
  onClose: () => void
}) {
  const [chunks, setChunks] = useState<TrainingLogChunk[]>([])
  const [selectedSeq, setSelectedSeq] = useState<number | null>(null)
  const [content, setContent] = useState('')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  async function loadLogs(options?: { keepSelection?: boolean }) {
    setLoading(true)
    setError('')
    try {
      const response = await fetchTrainingLogs(job.job_id, 20)
      setChunks(response.chunks)
      const targetSeq =
        options?.keepSelection && selectedSeq !== null && response.chunks.some((chunk) => chunk.seq === selectedSeq)
          ? selectedSeq
          : response.chunks[0]?.seq ?? null
      setSelectedSeq(targetSeq)
      if (targetSeq !== null) {
        setContent(await fetchTrainingLogChunkText(job.job_id, targetSeq))
      } else {
        setContent('')
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load logs')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    void loadLogs()
  }, [job.job_id])

  useEffect(() => {
    if (!shouldPollJob(job)) return
    const interval = setInterval(() => {
      void loadLogs({ keepSelection: true })
    }, 10000)
    return () => clearInterval(interval)
  }, [job])

  async function handleSelectSeq(seq: number) {
    setSelectedSeq(seq)
    setLoading(true)
    setError('')
    try {
      setContent(await fetchTrainingLogChunkText(job.job_id, seq))
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load log chunk')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4">
      <div className="grid h-[80vh] w-full max-w-6xl grid-cols-[280px_1fr] gap-4 rounded-2xl border border-edge bg-surface p-4">
        <div className="flex min-h-0 flex-col rounded-xl border border-edge bg-raised">
          <div className="flex items-center justify-between border-b border-edge px-4 py-3">
            <div>
              <div className="text-heading text-sm font-semibold">Logs</div>
              <div className="text-[10px] font-mono text-muted">{job.job_id.slice(0, 12)}</div>
            </div>
            <button
              onClick={onClose}
              className="text-muted hover:text-primary"
              type="button"
            >
              Close
            </button>
          </div>
          <div className="flex-1 space-y-2 overflow-y-auto p-3">
            <button
              onClick={() => { void loadLogs({ keepSelection: true }) }}
              className="w-full rounded-lg border border-edge px-3 py-2 text-left text-[11px] font-semibold text-primary transition-colors hover:border-accent hover:text-accent"
              type="button"
            >
              Refresh Log Index
            </button>
            {chunks.length === 0 ? (
              <div className="rounded-lg border border-dashed border-edge bg-surface px-3 py-6 text-center text-sm text-subtle">
                No log chunks yet.
              </div>
            ) : (
              chunks.map((chunk) => (
                <button
                  key={chunk.seq}
                  onClick={() => { void handleSelectSeq(chunk.seq) }}
                  className={`w-full rounded-lg border px-3 py-2 text-left transition-colors ${
                    selectedSeq === chunk.seq
                      ? 'border-accent bg-accent-dim/20'
                      : 'border-edge bg-surface hover:border-accent/40'
                  }`}
                  type="button"
                >
                  <div className="text-[11px] font-mono text-primary">chunk #{chunk.seq}</div>
                  <div className="mt-1 text-[10px] font-mono text-muted">
                    {formatDateTime(chunk.created_at)}
                  </div>
                </button>
              ))
            )}
          </div>
        </div>

        <div className="flex min-h-0 flex-col rounded-xl border border-edge bg-raised">
          <div className="flex items-center justify-between border-b border-edge px-4 py-3">
            <div className="text-heading text-sm font-semibold">Chunk Content</div>
            {selectedSeq !== null && (
              <div className="text-[10px] font-mono text-muted">seq={selectedSeq}</div>
            )}
          </div>
          <div className="min-h-0 flex-1 overflow-auto p-4">
            {error ? (
              <div className="rounded-lg border border-error/20 bg-error-dim px-3 py-2 text-error text-sm">
                {error}
              </div>
            ) : loading ? (
              <div className="text-sm text-muted">Loading…</div>
            ) : (
              <pre className="whitespace-pre-wrap break-words text-[12px] leading-relaxed text-primary">{content || 'No log content.'}</pre>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
