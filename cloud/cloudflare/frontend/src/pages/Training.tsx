import { useState, useEffect, useRef } from 'react'
import { Link, useSearchParams } from 'react-router'
import {
  fetchVoices,
  fetchVoiceDatasets,
  fetchTrainingJobs,
  fetchTrainingAdvice,
  fetchTrainingRounds,
  fetchDatasetSnapshots,
  startTraining,
  fetchTrainingJob,
  cancelTrainingJob,
  reconcileTrainingJob,
  revalidateTrainingJob,
  fetchTrainingLogs,
  fetchTrainingLogChunkText,
  fetchTrainingPreprocessCache,
  fetchTrainingCheckoutLedger,
  type DatasetInfo,
  type DatasetSnapshot,
  type DatasetPreprocessCacheEntry,
  type TrainingCheckoutLedgerEntry,
  type TrainingRound,
  type TrainingLogChunk,
  type TrainingPreprocessCacheResponse,
  type Voice,
  type TrainingJob,
  type TrainingConfig,
  type TrainingAdvice,
  updateTrainingPreprocessCache,
  updateTrainingPreprocessEntry,
  formatDateTime,
  formatDurationMs,
  formatTime,
} from '../lib/api'
import { TrainingAdviceCard } from '../components/TrainingAdviceCard'
import {
  getTrainingCheckoutSearch,
  getTrainingJobDisplayStatus,
  isActiveTrainingJobStatus,
  needsTrainingValidationFollowup,
  shouldWatchTrainingJob,
} from '../lib/trainingCheckout'
import { buildTrainingAdvice } from '../lib/trainingAdvisor'

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
  if (!refAudioKey || !/\/ref_audio\.[^/]+$/i.test(refAudioKey)) {
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
    if (isActiveTrainingJobStatus(job.status) && startedAt !== null) {
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
  const [showAdvancedConfig, setShowAdvancedConfig] = useState(false)
  const [availableDatasets, setAvailableDatasets] = useState<DatasetInfo[]>([])
  const [loadingDatasets, setLoadingDatasets] = useState(false)
  const [selectedDatasetName, setSelectedDatasetName] = useState('')
  const [starting, setStarting] = useState(false)
  const [formError, setFormError] = useState('')

  // Jobs
  const [jobs, setJobs] = useState<TrainingJob[]>([])
  const [rounds, setRounds] = useState<TrainingRound[]>([])
  const [snapshots, setSnapshots] = useState<DatasetSnapshot[]>([])
  const [serverTrainingAdvice, setServerTrainingAdvice] = useState<TrainingAdvice | null>(null)
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
  const hasActiveJobs = jobs.some((j) => shouldWatchTrainingJob(j))
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
  const selectedVoiceAdviceSignature = selectedVoiceJobs
    .map((job) => `${job.job_id}:${job.status}:${job.updated_at ?? job.created_at}:${getTrainingCheckoutSearch(job).status}`)
    .join('|')
  const selectedVoiceRounds = rounds.filter((round) => round.voice_id === selectedVoiceId)
  const selectedVoiceSnapshots = snapshots.filter((snapshot) => snapshot.voice_id === selectedVoiceId)
  const activeRound = selectedVoiceRounds.find((round) => round.round_id === selectedVoice?.active_round_id) ?? selectedVoiceRounds[0] ?? null
  const productionEpoch = activeRound?.production_epoch ?? selectedVoice?.epoch ?? null
  const productionScore = activeRound?.production_score ?? selectedVoice?.checkpoint_score ?? null
  const productionPreset = activeRound?.production_preset ?? selectedVoice?.checkpoint_preset ?? null
  const candidateEpoch = activeRound?.candidate_epoch ?? selectedVoice?.candidate_epoch ?? null
  const candidateScore = activeRound?.candidate_score ?? selectedVoice?.candidate_score ?? null
  const candidatePreset = selectedVoice?.candidate_preset ?? null
  const selectedDatasetSnapshot =
    selectedVoiceSnapshots.find((snapshot) => snapshot.dataset_name === effectiveDatasetName) ?? null
  const localTrainingAdvice = buildTrainingAdvice(selectedVoice ?? null, selectedVoiceJobs)
  const trainingAdvice = serverTrainingAdvice ?? localTrainingAdvice

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
      setServerTrainingAdvice(null)
      return
    }

    const voiceId = selectedVoice.voice_id
    let cancelled = false

    async function loadTrainingAdvice() {
      try {
        const response = await fetchTrainingAdvice(voiceId, 100)
        if (!cancelled) {
          setServerTrainingAdvice(response.advice)
        }
      } catch {
        if (!cancelled) {
          setServerTrainingAdvice(null)
        }
      }
    }

    void loadTrainingAdvice()
    return () => {
      cancelled = true
    }
  }, [selectedVoice?.voice_id, selectedVoiceAdviceSignature])

  useEffect(() => {
    if (!selectedVoice) {
      setAvailableDatasets([])
      setSelectedDatasetName('')
      setRounds([])
      setSnapshots([])
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
    if (!selectedVoiceId) return
    let cancelled = false

    async function loadWorkflow() {
      try {
        const [roundsData, snapshotsData] = await Promise.all([
          fetchTrainingRounds(selectedVoiceId, 20),
          fetchDatasetSnapshots(selectedVoiceId, 20),
        ])
        if (cancelled) return
        setRounds(roundsData.rounds)
        setSnapshots(snapshotsData.snapshots)
      } catch {
        if (!cancelled) {
          setRounds([])
          setSnapshots([])
        }
      }
    }

    void loadWorkflow()
    return () => {
      cancelled = true
    }
  }, [selectedVoiceId])

  useEffect(() => {
    if (!hasActiveJobs) return

    const interval = setInterval(async () => {
      const currentJobs = jobsRef.current
      const activeJobIds = currentJobs
        .filter((j) => shouldWatchTrainingJob(j))
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
        round_id: result.round_id,
        dataset_snapshot_id: result.dataset_snapshot_id,
        created_at: Date.now(),
        last_heartbeat_at: null,
        summary: {},
        metrics: {},
        supervisor: { phase: 'pending' },
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
      const [roundsData, snapshotsData] = await Promise.all([
        fetchTrainingRounds(selectedVoiceId, 20),
        fetchDatasetSnapshots(selectedVoiceId, 20),
      ])
      setRounds(roundsData.rounds)
      setSnapshots(snapshotsData.snapshots)
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
    if (selectedVoiceId) {
      const [roundsData, snapshotsData] = await Promise.all([
        fetchTrainingRounds(selectedVoiceId, 20),
        fetchDatasetSnapshots(selectedVoiceId, 20),
      ])
      setRounds(roundsData.rounds)
      setSnapshots(snapshotsData.snapshots)
    }
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
    const fallbackPreset = getRecommendedTrainingPreset(config.model_size ?? selectedVoice?.model_size ?? '1.7B')
    setBatchSize(config.batch_size ?? fallbackPreset.batchSize)
    setEpochs(config.num_epochs ?? fallbackPreset.epochs)
    setLearningRate(config.learning_rate ?? fallbackPreset.learningRate)
    setTrainingSeed(config.seed ?? fallbackPreset.seed)
    setGradientAccumulationSteps(config.gradient_accumulation_steps ?? fallbackPreset.gradientAccumulationSteps)
    setSubtalkerLossWeight(
      config.subtalker_loss_weight ?? fallbackPreset.subtalkerLossWeight,
    )
    setSaveEveryNEpochs(config.save_every_n_epochs ?? fallbackPreset.saveEveryNEpochs)
    setTrainingLanguage(config.whisper_language ?? 'ko')
    setGpuTypeId(config.gpu_type_id ?? fallbackPreset.gpuTypeId)
  }

  const jobsByRecency = [...jobs].sort(
    (a, b) => (readTimestamp(b.created_at) ?? 0) - (readTimestamp(a.created_at) ?? 0),
  )
  const jobCardMeta = buildJobCardMeta(jobsByRecency, voices, Date.now())
  const activeJobs = jobsByRecency.filter((j) => shouldWatchTrainingJob(j))
  const completedJobs = jobsByRecency.filter(
    (j) => !shouldWatchTrainingJob(j) && (j.status === 'completed' || j.status === 'failed' || j.status === 'cancelled'),
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
            <div className="mb-4 rounded-lg border border-edge bg-surface px-3 py-3">
              <div className="flex items-center justify-between gap-3">
                <div>
                  <div className="text-heading text-xs font-semibold">Workflow State</div>
                  <p className="text-subtle text-[11px] mt-1">
                    Production and candidate are now tracked separately. New runs attach to a round and a frozen dataset snapshot.
                  </p>
                </div>
                {activeRound && (
                  <span className="rounded-full bg-raised px-2 py-1 text-[10px] font-mono text-muted">
                    round #{activeRound.round_index}
                  </span>
                )}
              </div>
              <div className="mt-3 flex flex-wrap gap-2 text-[10px] font-mono text-muted">
                <span>production={selectedVoice.run_name ?? 'none'}</span>
                <span>epoch={selectedVoice.epoch ?? 'n/a'}</span>
                <span>candidate={selectedVoice.candidate_run_name ?? 'none'}</span>
                <span>candidate_epoch={selectedVoice.candidate_epoch ?? 'n/a'}</span>
                <span>active_round={selectedVoice.active_round_id?.slice(0, 8) ?? 'none'}</span>
              </div>
              <div className="mt-3 flex flex-wrap gap-2 text-[10px] font-mono text-muted">
                <span>snapshot={selectedDatasetSnapshot?.status ?? 'not_frozen'}</span>
                <span>dataset={selectedDatasetSnapshot?.dataset_name ?? (effectiveDatasetName || 'none')}</span>
                <span>signature={selectedDatasetSnapshot?.dataset_signature?.slice(0, 10) ?? 'pending'}</span>
                <span>segments={selectedDatasetSnapshot?.segments_accepted ?? 'n/a'}</span>
              </div>
              <div className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-4">
                <Metric label="Production" value={formatCheckpointMetric(productionEpoch, productionScore, productionPreset)} />
                <Metric label="Champion" value={formatCheckpointMetric(activeRound?.champion_epoch, activeRound?.champion_score, activeRound?.champion_preset)} />
                <Metric label="Selected" value={formatCheckpointMetric(activeRound?.selected_epoch, activeRound?.selected_score, activeRound?.selected_preset)} />
                <Metric label="Candidate" value={formatCheckpointMetric(candidateEpoch, candidateScore, candidatePreset)} />
              </div>
              <div className="mt-3 flex flex-wrap gap-2 text-[10px] font-mono text-muted">
                <span className="rounded-full bg-raised px-2 py-1">round_status={activeRound?.status ?? 'none'}</span>
                <span className="rounded-full bg-raised px-2 py-1">adoption={activeRound?.adoption_mode ?? 'n/a'}</span>
                <span className="rounded-full bg-raised px-2 py-1">production_job={activeRound?.production_job_id?.slice(0, 8) ?? 'n/a'}</span>
                <span className="rounded-full bg-raised px-2 py-1">selected_job={activeRound?.selected_job_id?.slice(0, 8) ?? 'n/a'}</span>
              </div>
              {selectedVoice.candidate_checkpoint_r2_prefix && (
                <p className="mt-3 text-[11px] text-warning">
                  A validated candidate is waiting. It will not replace production until you promote it from Compare.
                </p>
              )}
            </div>
          )}

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

            <div className="rounded-lg border border-edge bg-surface px-3 py-2.5">
              <button
                type="button"
                onClick={() => setShowAdvancedConfig((previous) => !previous)}
                className="flex w-full items-center justify-between text-left"
              >
                <span className="text-primary text-xs font-semibold">Advanced training settings</span>
                <span className="text-muted text-[10px] font-mono">{showAdvancedConfig ? 'Hide' : 'Show'}</span>
              </button>

              {showAdvancedConfig && (
                <div className="mt-3 space-y-4">
                  <div>
                    <label htmlFor="training-batch-size" className="text-subtle text-xs font-medium mb-1.5 block">Batch Size</label>
                    <input
                      id="training-batch-size"
                      type="number"
                      value={batchSize}
                      onChange={(e) => setBatchSize(parseInt(e.target.value) || 1)}
                      min={1}
                      max={32}
                      className="w-full bg-raised border border-edge rounded-lg px-3 py-2.5 text-sm text-primary font-mono focus:border-accent transition-colors"
                    />
                  </div>

                  <div>
                    <label htmlFor="training-epochs" className="text-subtle text-xs font-medium mb-1.5 block">Epochs</label>
                    <input
                      id="training-epochs"
                      type="number"
                      value={epochs}
                      onChange={(e) => setEpochs(parseInt(e.target.value) || 1)}
                      min={1}
                      max={50}
                      className="w-full bg-raised border border-edge rounded-lg px-3 py-2.5 text-sm text-primary font-mono focus:border-accent transition-colors"
                    />
                    <p className="text-muted text-[10px] font-mono mt-1">Recommended: 5–15 (more epochs = higher similarity, risk of overfitting)</p>
                  </div>

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
                      className="w-full bg-raised border border-edge rounded-lg px-3 py-2.5 text-sm text-primary font-mono focus:border-accent transition-colors"
                    />
                    <p className="text-muted text-[10px] font-mono mt-1">Recommended: 1e-5 to 5e-5</p>
                  </div>

                  <div>
                    <label htmlFor="training-seed" className="text-subtle text-xs font-medium mb-1.5 block">Seed</label>
                    <input
                      id="training-seed"
                      type="number"
                      value={trainingSeed}
                      onChange={(e) => setTrainingSeed(parseInt(e.target.value, 10) || 1)}
                      className="w-full bg-raised border border-edge rounded-lg px-3 py-2.5 text-sm text-primary font-mono focus:border-accent transition-colors"
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
                      className="w-full bg-raised border border-edge rounded-lg px-3 py-2.5 text-sm text-primary font-mono focus:border-accent transition-colors"
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
                      className="w-full bg-raised border border-edge rounded-lg px-3 py-2.5 text-sm text-primary font-mono focus:border-accent transition-colors"
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
                      className="w-full bg-raised border border-edge rounded-lg px-3 py-2.5 text-sm text-primary font-mono focus:border-accent transition-colors"
                    />
                  </div>

                  <div>
                    <label htmlFor="training-gpu" className="text-subtle text-xs font-medium mb-1.5 block">GPU Type</label>
                    <input
                      id="training-gpu"
                      type="text"
                      value={gpuTypeId}
                      onChange={(e) => setGpuTypeId(e.target.value)}
                      className="w-full bg-raised border border-edge rounded-lg px-3 py-2.5 text-sm text-primary font-mono focus:border-accent transition-colors"
                    />
                  </div>
                </div>
              )}
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
  const checkout = getTrainingCheckoutSearch(job)
  const isActive = isActiveTrainingJobStatus(job.status)
  const isValidationPending = needsTrainingValidationFollowup(job)
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
  const championCheckpointEpoch = checkout.champion?.epoch ?? null
  const championCheckpointScore = checkout.champion?.score ?? null
  const selectedCheckpointEpoch = checkout.selected?.epoch ?? null
  const selectedCheckpointScore = checkout.selected?.score ?? null
  const canCompareCheckpoints = checkout.compare_ready
  const validationPassed = checkout.validation_passed
  const validationChecked = checkout.validation_checked
  const validationRejected = checkout.status === 'rejected'
  const statusLabel = getTrainingJobDisplayStatus(job)
  const validationMessage = checkout.message
  const lastMessage = checkout.last_message
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
    promoted: { bg: 'bg-accent-dim', text: 'text-accent' },
    manual_promoted: { bg: 'bg-accent-dim', text: 'text-accent' },
    candidate_ready: { bg: 'bg-warning-dim', text: 'text-warning' },
    kept_current: { bg: 'bg-raised', text: 'text-muted' },
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

      <div className="mt-3 flex flex-wrap gap-2 text-[10px] font-mono text-muted">
        <span className="rounded-full bg-raised px-2 py-1">round={job.round_id?.slice(0, 8) ?? 'n/a'}</span>
        <span className="rounded-full bg-raised px-2 py-1">snapshot={job.dataset_snapshot_id?.slice(0, 8) ?? 'n/a'}</span>
        <span className="rounded-full bg-raised px-2 py-1">phase={String(job.supervisor?.phase ?? job.status)}</span>
        {checkout.adoption_mode && (
          <span className="rounded-full bg-raised px-2 py-1">adoption={checkout.adoption_mode}</span>
        )}
      </div>

      {job.status === 'completed' && (durationMs !== null || finalLoss !== null || finalEpoch !== null) && (
        <div className="mt-3 grid grid-cols-3 gap-3">
          <Metric label="Duration" value={formatDurationMs(durationMs)} />
          <Metric label="Final Loss" value={finalLoss !== null ? finalLoss.toFixed(4) : '—'} />
          <Metric label="Epochs" value={finalEpoch !== null ? `${finalEpoch}/${summaryTotalEpochs ?? '—'}` : '—'} />
        </div>
      )}

      {(validationChecked || championCheckpointEpoch !== null || selectedCheckpointEpoch !== null) && (
        <div className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-3">
          <Metric
            label="Validation"
            value={validationChecked ? checkout.status : 'pending'}
          />
          <Metric
            label="Champion"
            value={
              championCheckpointEpoch !== null
                ? `e${championCheckpointEpoch}${championCheckpointScore !== null ? ` · ${championCheckpointScore.toFixed(3)}` : ''}`
                : '—'
            }
          />
          <Metric
            label="Selected"
            value={
              selectedCheckpointEpoch !== null
                ? `e${selectedCheckpointEpoch}${selectedCheckpointScore !== null ? ` · ${selectedCheckpointScore.toFixed(3)}` : ''}`
                : '—'
            }
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
          Inspect Run
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
        {typeof job.summary?.preprocess_cache_lookup === 'string' && (
          <span>cache={job.summary.preprocess_cache_lookup}</span>
        )}
        {typeof job.summary?.preprocess_cache_segments_accepted === 'number' && (
          <span>segments={job.summary.preprocess_cache_segments_accepted}</span>
        )}
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

function formatBytes(value: number): string {
  if (!Number.isFinite(value) || value <= 0) return '0 B'
  const units = ['B', 'KB', 'MB', 'GB']
  let size = value
  let unitIndex = 0
  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024
    unitIndex += 1
  }
  return `${size >= 10 || unitIndex === 0 ? size.toFixed(0) : size.toFixed(1)} ${units[unitIndex]}`
}

function getPathLabel(path: string): string {
  const parts = path.split('/').filter(Boolean)
  return parts[parts.length - 1] ?? path
}

function formatCheckpointMetric(
  epoch: number | null | undefined,
  score: number | null | undefined,
  preset?: string | null,
): string {
  if (typeof epoch !== 'number') return '—'
  const scoreLabel = typeof score === 'number' ? ` · ${score.toFixed(3)}` : ''
  const presetLabel = preset ? ` · ${preset}` : ''
  return `e${epoch}${scoreLabel}${presetLabel}`
}

function JobLogsModal({
  job,
  onClose,
}: {
  job: TrainingJob
  onClose: () => void
}) {
  const [activeTab, setActiveTab] = useState<'logs' | 'checkout' | 'transcripts'>('logs')
  const [chunks, setChunks] = useState<TrainingLogChunk[]>([])
  const [selectedSeq, setSelectedSeq] = useState<number | null>(null)
  const [content, setContent] = useState('')
  const [loadingLogs, setLoadingLogs] = useState(true)
  const [logError, setLogError] = useState('')
  const [checkoutLedger, setCheckoutLedger] = useState<TrainingCheckoutLedgerEntry[]>([])
  const [loadingLedger, setLoadingLedger] = useState(true)
  const [ledgerError, setLedgerError] = useState('')
  const [preprocess, setPreprocess] = useState<TrainingPreprocessCacheResponse | null>(null)
  const [loadingPreprocess, setLoadingPreprocess] = useState(true)
  const [preprocessError, setPreprocessError] = useState('')
  const [selectedEntryId, setSelectedEntryId] = useState<string | null>(null)
  const [entryDraft, setEntryDraft] = useState('')
  const [entryIncluded, setEntryIncluded] = useState(true)
  const [referenceDraft, setReferenceDraft] = useState('')
  const [transcriptQuery, setTranscriptQuery] = useState('')
  const [saving, setSaving] = useState<'entry' | 'reference' | ''>('')
  const [saveMessage, setSaveMessage] = useState('')

  async function loadLogs(options?: { keepSelection?: boolean }) {
    setLoadingLogs(true)
    setLogError('')
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
      setLogError(err instanceof Error ? err.message : 'Failed to load logs')
    } finally {
      setLoadingLogs(false)
    }
  }

  async function loadPreprocess(options?: { keepSelection?: boolean }) {
    setLoadingPreprocess(true)
    setPreprocessError('')
    try {
      const response = await fetchTrainingPreprocessCache(job.job_id)
      setPreprocess(response)
      const keepSelection =
        options?.keepSelection &&
        selectedEntryId !== null &&
        response.entries.some((entry) => entry.entry_id === selectedEntryId)
      const targetEntry = keepSelection
        ? response.entries.find((entry) => entry.entry_id === selectedEntryId) ?? null
        : response.entries[0] ?? null
      setSelectedEntryId(targetEntry?.entry_id ?? null)
      setReferenceDraft(response.reference_text ?? '')
    } catch (err) {
      setPreprocessError(err instanceof Error ? err.message : 'Failed to load cached transcripts')
    } finally {
      setLoadingPreprocess(false)
    }
  }

  async function loadCheckoutLedger() {
    setLoadingLedger(true)
    setLedgerError('')
    try {
      const response = await fetchTrainingCheckoutLedger(job.job_id)
      setCheckoutLedger(response.entries)
    } catch (err) {
      setLedgerError(err instanceof Error ? err.message : 'Failed to load checkout ledger')
    } finally {
      setLoadingLedger(false)
    }
  }

  useEffect(() => {
    void Promise.all([
      loadLogs(),
      loadPreprocess(),
      loadCheckoutLedger(),
    ])
  }, [job.job_id])

  useEffect(() => {
    if (!shouldWatchTrainingJob(job)) return
    const interval = setInterval(() => {
      void loadLogs({ keepSelection: true })
      void loadPreprocess({ keepSelection: true })
      void loadCheckoutLedger()
    }, 10000)
    return () => clearInterval(interval)
  }, [job])

  async function handleSelectSeq(seq: number) {
    setSelectedSeq(seq)
    setLoadingLogs(true)
    setLogError('')
    try {
      setContent(await fetchTrainingLogChunkText(job.job_id, seq))
    } catch (err) {
      setLogError(err instanceof Error ? err.message : 'Failed to load log chunk')
    } finally {
      setLoadingLogs(false)
    }
  }

  const entries = preprocess?.entries ?? []
  const includedEntries = entries.filter((entry) => entry.included)
  const filteredEntries = entries.filter((entry) => {
    const query = transcriptQuery.trim().toLowerCase()
    if (!query) return true
    return (
      entry.text.toLowerCase().includes(query) ||
      entry.audio_path.toLowerCase().includes(query)
    )
  })
  const selectedEntry =
    entries.find((entry) => entry.entry_id === selectedEntryId) ??
    filteredEntries[0] ??
    null
  const totalLogBytes = chunks.reduce((sum, chunk) => sum + (chunk.bytes ?? 0), 0)
  const totalLogLines = chunks.reduce((sum, chunk) => sum + (chunk.lines ?? 0), 0)
  const championLedgerCount = checkoutLedger.filter((entry) => entry.role === 'champion').length
  const firstChunkAt = chunks.length > 0 ? chunks[chunks.length - 1]?.created_at ?? null : null
  const lastChunkAt = chunks[0]?.created_at ?? null
  const cacheLookup =
    typeof job.summary?.preprocess_cache_lookup === 'string'
      ? job.summary.preprocess_cache_lookup
      : preprocess?.cache
        ? 'hit'
        : 'miss'

  useEffect(() => {
    if (!selectedEntry) {
      setEntryDraft('')
      setEntryIncluded(true)
      return
    }
    setEntryDraft(selectedEntry.text)
    setEntryIncluded(selectedEntry.included)
  }, [selectedEntry?.entry_id, selectedEntry?.text, selectedEntry?.included])

  async function handleSaveEntry() {
    if (!selectedEntry) return
    setSaving('entry')
    setSaveMessage('')
    try {
      const response = await updateTrainingPreprocessEntry(job.job_id, selectedEntry.entry_id, {
        text: entryDraft,
        included: entryIncluded,
      })
      setPreprocess((previous) => {
        if (!previous) return previous
        return {
          ...previous,
          entries: previous.entries.map((entry) =>
            entry.entry_id === response.entry.entry_id ? response.entry : entry,
          ),
        }
      })
      setSaveMessage('Transcript entry saved to the reusable cache.')
    } catch (err) {
      setPreprocessError(err instanceof Error ? err.message : 'Failed to save transcript entry')
    } finally {
      setSaving('')
    }
  }

  async function handleSaveReference() {
    setSaving('reference')
    setSaveMessage('')
    try {
      const response = await updateTrainingPreprocessCache(job.job_id, {
        reference_text: referenceDraft,
      })
      setPreprocess((previous) => (
        previous
          ? {
              ...previous,
              reference_text: response.reference_text,
            }
          : previous
      ))
      setSaveMessage('Reference text updated.')
    } catch (err) {
      setPreprocessError(err instanceof Error ? err.message : 'Failed to save reference text')
    } finally {
      setSaving('')
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4">
      <div className="flex h-[88vh] w-full max-w-7xl flex-col rounded-2xl border border-edge bg-surface p-4">
        <div className="mb-4 flex items-start justify-between gap-4">
          <div>
            <div className="text-heading text-sm font-semibold">Run Inspector</div>
            <div className="text-[10px] font-mono text-muted">{job.job_id.slice(0, 12)}</div>
            <div className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-5">
              <Metric label="Status" value={job.status} />
              <Metric label="Cache" value={String(cacheLookup)} />
              <Metric label="Logs" value={String(chunks.length)} />
              <Metric label="Entries" value={preprocess?.cache ? `${includedEntries.length}/${entries.length}` : '—'} />
              <Metric label="Ledger" value={String(checkoutLedger.length)} />
            </div>
            <div className="mt-3 flex flex-wrap gap-4 text-[10px] font-mono text-muted">
              {lastChunkAt !== null && <span>last_log={formatDateTime(lastChunkAt)}</span>}
              {firstChunkAt !== null && <span>first_log={formatDateTime(firstChunkAt)}</span>}
              {preprocess?.cache?.updated_at ? <span>cache_updated={formatDateTime(preprocess.cache.updated_at)}</span> : null}
              {preprocess?.hydrated_from_r2 ? <span>cache_sync=refreshed</span> : null}
              {championLedgerCount > 0 ? <span>champion_entries={championLedgerCount}</span> : null}
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => {
                void loadLogs({ keepSelection: true })
                void loadPreprocess({ keepSelection: true })
                void loadCheckoutLedger()
              }}
              className="rounded-lg border border-edge px-3 py-2 text-[11px] font-semibold text-primary transition-colors hover:border-accent hover:text-accent"
              type="button"
            >
              Refresh
            </button>
            <button
              onClick={onClose}
              className="text-muted hover:text-primary"
              type="button"
            >
              Close
            </button>
          </div>
        </div>

        <div className="mb-4 flex gap-2">
          <button
            onClick={() => setActiveTab('logs')}
            className={`rounded-lg px-3 py-2 text-[11px] font-semibold transition-colors ${
              activeTab === 'logs'
                ? 'bg-accent text-void'
                : 'border border-edge text-primary hover:border-accent hover:text-accent'
            }`}
            type="button"
          >
            Logs
          </button>
          <button
            onClick={() => setActiveTab('checkout')}
            className={`rounded-lg px-3 py-2 text-[11px] font-semibold transition-colors ${
              activeTab === 'checkout'
                ? 'bg-accent text-void'
                : 'border border-edge text-primary hover:border-accent hover:text-accent'
            }`}
            type="button"
          >
            Checkout Ledger
          </button>
          <button
            onClick={() => setActiveTab('transcripts')}
            className={`rounded-lg px-3 py-2 text-[11px] font-semibold transition-colors ${
              activeTab === 'transcripts'
                ? 'bg-accent text-void'
                : 'border border-edge text-primary hover:border-accent hover:text-accent'
            }`}
            type="button"
          >
            Cached Transcripts
          </button>
        </div>

        {activeTab === 'logs' ? (
          <div className="grid min-h-0 flex-1 grid-cols-[280px_1fr] gap-4">
            <div className="flex min-h-0 flex-col rounded-xl border border-edge bg-raised">
              <div className="border-b border-edge px-4 py-3">
                <div className="text-heading text-sm font-semibold">Log Index</div>
                <div className="mt-1 flex flex-wrap gap-3 text-[10px] font-mono text-muted">
                  <span>bytes={formatBytes(totalLogBytes)}</span>
                  <span>lines={totalLogLines}</span>
                </div>
              </div>
              <div className="flex-1 space-y-2 overflow-y-auto p-3">
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
                      <div className="flex items-center justify-between gap-2">
                        <div className="text-[11px] font-mono text-primary">chunk #{chunk.seq}</div>
                        {typeof chunk.lines === 'number' && (
                          <div className="text-[10px] font-mono text-muted">{chunk.lines} lines</div>
                        )}
                      </div>
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
                {logError ? (
                  <div className="rounded-lg border border-error/20 bg-error-dim px-3 py-2 text-error text-sm">
                    {logError}
                  </div>
                ) : loadingLogs ? (
                  <div className="text-sm text-muted">Loading…</div>
                ) : (
                  <pre className="whitespace-pre-wrap break-words text-[12px] leading-relaxed text-primary">{content || 'No log content.'}</pre>
                )}
              </div>
            </div>
          </div>
        ) : activeTab === 'checkout' ? (
          <div className="min-h-0 flex-1 overflow-y-auto rounded-xl border border-edge bg-raised p-4">
            {ledgerError ? (
              <div className="rounded-lg border border-error/20 bg-error-dim px-3 py-2 text-error text-sm">
                {ledgerError}
              </div>
            ) : loadingLedger ? (
              <div className="text-sm text-muted">Loading checkout ledger…</div>
            ) : checkoutLedger.length === 0 ? (
              <div className="rounded-lg border border-dashed border-edge bg-surface px-3 py-6 text-center text-sm text-subtle">
                No persisted checkout entries for this run yet.
              </div>
            ) : (
              <div className="space-y-3">
                {checkoutLedger.map((entry) => (
                  <div key={entry.entry_id} className="rounded-xl border border-edge bg-surface p-4">
                    <div className="flex items-start justify-between gap-3">
                      <div>
                        <div className="text-primary text-sm font-semibold">{entry.role.replaceAll('_', ' ')}</div>
                        <div className="mt-1 text-[10px] font-mono text-muted">
                          {entry.run_name ?? 'checkpoint'} · epoch={entry.epoch ?? 'n/a'} · score={typeof entry.score === 'number' ? entry.score.toFixed(3) : 'n/a'}
                        </div>
                      </div>
                      <div className="text-right text-[10px] font-mono">
                        <div className={entry.ok === true ? 'text-accent' : entry.ok === false ? 'text-error' : 'text-muted'}>
                          {entry.ok === true ? 'passed' : entry.ok === false ? 'failed' : 'n/a'}
                        </div>
                        <div className="text-muted">{formatDateTime(entry.created_at)}</div>
                      </div>
                    </div>

                    <div className="mt-3 flex flex-wrap gap-2 text-[10px] font-mono text-muted">
                      <span className="rounded-full bg-raised px-2 py-0.5">preset={entry.preset ?? 'n/a'}</span>
                      <span className="rounded-full bg-raised px-2 py-0.5">source={entry.source}</span>
                      {entry.adoption_mode && (
                        <span className="rounded-full bg-raised px-2 py-0.5">adoption={entry.adoption_mode}</span>
                      )}
                      {typeof entry.passed_samples === 'number' && typeof entry.total_samples === 'number' && (
                        <span className="rounded-full bg-raised px-2 py-0.5">samples={entry.passed_samples}/{entry.total_samples}</span>
                      )}
                    </div>

                    {entry.message && (
                      <div className="mt-3 text-[11px] leading-relaxed text-subtle">{entry.message}</div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        ) : (
          <div className="grid min-h-0 flex-1 grid-cols-[320px_1fr] gap-4">
            <div className="flex min-h-0 flex-col rounded-xl border border-edge bg-raised">
              <div className="border-b border-edge px-4 py-3">
                <div className="text-heading text-sm font-semibold">Transcript Cache</div>
                <div className="mt-2 grid grid-cols-2 gap-3 text-[10px] font-mono text-muted">
                  <span>accepted={preprocess?.cache?.segments_accepted ?? '—'}</span>
                  <span>minutes={preprocess?.cache?.accepted_duration_min ?? '—'}</span>
                  <span>source_files={preprocess?.cache?.source_file_count ?? '—'}</span>
                  <span>signature={preprocess?.cache?.dataset_signature?.slice(0, 10) ?? '—'}</span>
                </div>
                <input
                  value={transcriptQuery}
                  onChange={(event) => setTranscriptQuery(event.target.value)}
                  className="mt-3 w-full rounded-lg border border-edge bg-surface px-3 py-2 text-sm text-primary outline-none transition-colors focus:border-accent"
                  placeholder="Search transcripts or segment names"
                />
              </div>
              <div className="flex-1 space-y-2 overflow-y-auto p-3">
                {loadingPreprocess ? (
                  <div className="text-sm text-muted">Loading cached transcripts…</div>
                ) : preprocessError ? (
                  <div className="rounded-lg border border-error/20 bg-error-dim px-3 py-2 text-error text-sm">
                    {preprocessError}
                  </div>
                ) : !preprocess?.cache ? (
                  <div className="rounded-lg border border-dashed border-edge bg-surface px-3 py-6 text-center text-sm text-subtle">
                    This run has no reusable preprocess cache yet.
                  </div>
                ) : filteredEntries.length === 0 ? (
                  <div className="rounded-lg border border-dashed border-edge bg-surface px-3 py-6 text-center text-sm text-subtle">
                    No transcript entries match this filter.
                  </div>
                ) : (
                  filteredEntries.map((entry) => (
                    <button
                      key={entry.entry_id}
                      onClick={() => setSelectedEntryId(entry.entry_id)}
                      className={`w-full rounded-lg border px-3 py-2 text-left transition-colors ${
                        selectedEntry?.entry_id === entry.entry_id
                          ? 'border-accent bg-accent-dim/20'
                          : 'border-edge bg-surface hover:border-accent/40'
                      }`}
                      type="button"
                    >
                      <div className="flex items-center justify-between gap-2">
                        <div className="text-[11px] font-mono text-primary">#{entry.seq}</div>
                        <div className={`text-[10px] font-mono ${entry.included ? 'text-accent' : 'text-muted'}`}>
                          {entry.included ? 'included' : 'excluded'}
                        </div>
                      </div>
                      <div className="mt-1 truncate text-[11px] text-primary">{getPathLabel(entry.audio_path)}</div>
                      <div className="mt-1 line-clamp-2 text-[11px] text-subtle">{entry.text}</div>
                    </button>
                  ))
                )}
              </div>
            </div>

            <div className="flex min-h-0 flex-col rounded-xl border border-edge bg-raised">
              <div className="border-b border-edge px-4 py-3">
                <div className="text-heading text-sm font-semibold">Transcript Editor</div>
                {saveMessage && (
                  <div className="mt-2 rounded-lg border border-accent/20 bg-accent-dim/20 px-3 py-2 text-[11px] text-primary">
                    {saveMessage}
                  </div>
                )}
              </div>
              <div className="min-h-0 flex-1 overflow-y-auto p-4">
                {!preprocess?.cache ? (
                  <div className="text-sm text-muted">
                    Long-form raw-media runs will populate a reusable transcript cache here after preprocessing finishes.
                  </div>
                ) : (
                  <div className="space-y-6">
                    <section className="rounded-xl border border-edge bg-surface p-4">
                      <div className="flex items-center justify-between gap-3">
                        <div>
                          <div className="text-primary text-sm font-semibold">Reference Text</div>
                          <div className="text-[11px] text-subtle">Used with the cached reference audio for follow-up retrains.</div>
                        </div>
                        <button
                          onClick={() => { void handleSaveReference() }}
                          disabled={saving !== ''}
                          className="rounded-lg border border-edge px-3 py-2 text-[11px] font-semibold text-primary transition-colors hover:border-accent hover:text-accent disabled:opacity-50"
                          type="button"
                        >
                          {saving === 'reference' ? 'Saving…' : 'Save Reference'}
                        </button>
                      </div>
                      <textarea
                        value={referenceDraft}
                        onChange={(event) => setReferenceDraft(event.target.value)}
                        className="mt-3 min-h-24 w-full rounded-lg border border-edge bg-raised px-3 py-2 text-sm text-primary outline-none transition-colors focus:border-accent"
                      />
                    </section>

                    <section className="rounded-xl border border-edge bg-surface p-4">
                      {!selectedEntry ? (
                        <div className="text-sm text-muted">Select a transcript entry to review and edit it.</div>
                      ) : (
                        <>
                          <div className="flex flex-wrap items-start justify-between gap-3">
                            <div>
                              <div className="text-primary text-sm font-semibold">
                                Segment #{selectedEntry.seq}
                              </div>
                              <div className="mt-1 text-[11px] font-mono text-muted">
                                {selectedEntry.audio_path}
                              </div>
                            </div>
                            <label className="inline-flex items-center gap-2 text-[11px] font-semibold text-primary">
                              <input
                                type="checkbox"
                                checked={entryIncluded}
                                onChange={(event) => setEntryIncluded(event.target.checked)}
                                className="h-4 w-4 rounded border-edge bg-surface"
                              />
                              Include In Future Retrains
                            </label>
                          </div>

                          <textarea
                            value={entryDraft}
                            onChange={(event) => setEntryDraft(event.target.value)}
                            className="mt-4 min-h-40 w-full rounded-lg border border-edge bg-raised px-3 py-2 text-sm text-primary outline-none transition-colors focus:border-accent"
                          />

                          <div className="mt-4 flex items-center justify-between gap-3">
                            <div className="text-[11px] text-subtle">
                              Editing here rewrites the cached <code>train_raw.jsonl</code> used by the next retrain.
                            </div>
                            <button
                              onClick={() => { void handleSaveEntry() }}
                              disabled={saving !== ''}
                              className="rounded-lg bg-accent px-3 py-2 text-[11px] font-semibold text-void transition-colors hover:bg-accent-light disabled:opacity-50"
                              type="button"
                            >
                              {saving === 'entry' ? 'Saving…' : 'Save Entry'}
                            </button>
                          </div>
                        </>
                      )}
                    </section>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
