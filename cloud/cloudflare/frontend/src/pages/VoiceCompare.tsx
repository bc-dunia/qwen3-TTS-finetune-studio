import { useEffect, useState } from 'react'
import { useNavigate, useParams, useSearchParams } from 'react-router'
import { AudioPlayer } from '../components/AudioPlayer'
import { TrainingAdviceCard } from '../components/TrainingAdviceCard'
import {
  DEFAULT_VOICE_SETTINGS,
  fetchTrainingAdvice,
  fetchTrainingJobs,
  fetchVoice,
  formatDate,
  formatDateTime,
  formatDurationMs,
  formatTime,
  pollSpeechGeneration,
  promoteTrainingCheckpoint,
  startSpeechGenerationAsync,
  type SpeechGenerationOptions,
  type TrainingAdvice,
  type TrainingJob,
  type Voice,
  type VoiceSettings,
} from '../lib/api'
import {
  getTrainingCheckoutSearch,
  getTrainingJobDisplayStatus,
  shouldWatchTrainingJob,
} from '../lib/trainingCheckout'
import { buildTrainingAdvice } from '../lib/trainingAdvisor'

const MAX_COMPARE_CANDIDATES = 4

type CheckpointCandidate = {
  id: string
  prefix: string
  epoch: number | null
  score: number | null
  preset: string | null
  message: string | null
  jobId: string | null
  createdAt: number
  completedAt: number | null
  attemptNumber: number | null
  runName: string | null
  isCurrentProduction: boolean
  isStoredCandidate: boolean
  isJobRecommendation: boolean
  validationPassed: boolean
}

type RunSummary = {
  jobId: string
  attemptNumber: number | null
  createdAt: number
  startedAt: number | null
  completedAt: number | null
  durationMs: number | null
  status: string
  championScore: number | null
  championEpoch: number | null
  championPreset: string | null
  validationMessage: string | null
  hasCandidates: boolean
  validationPassed: boolean
  validationRejected: boolean
}

type CompareResult = {
  status: string
  blob?: Blob
  error?: string
}

function normalizeVoiceSettings(value: Partial<VoiceSettings> | null | undefined): VoiceSettings {
  const src = value ?? {}
  return {
    stability: typeof src.stability === 'number' ? src.stability : DEFAULT_VOICE_SETTINGS.stability,
    similarity_boost: typeof src.similarity_boost === 'number' ? src.similarity_boost : DEFAULT_VOICE_SETTINGS.similarity_boost,
    style: typeof src.style === 'number' ? src.style : DEFAULT_VOICE_SETTINGS.style,
    speed: typeof src.speed === 'number' ? src.speed : DEFAULT_VOICE_SETTINGS.speed,
  }
}

function getDefaultCompareText(language: string | undefined): string {
  switch ((language ?? '').toLowerCase()) {
    case 'en':
      return 'Hello. This sample compares checkpoint quality, tone preservation, and speaker similarity.'
    case 'ja':
      return 'こんにちは。この 샘플은 체크포인트별 화자 유사도와 말투 보존을 비교하기 위한 문장입니다。'
    case 'zh':
      return '你好。这段样例用于比较各个检查点的音色、语气和整体稳定性。'
    case 'ko':
    default:
      return '안녕하세요. 이 샘플은 체크포인트별 화자 유사도와 말투 보존을 비교하기 위한 문장입니다.'
  }
}

function parseRunNameFromPrefix(prefix: string | null | undefined): string | null {
  if (!prefix) return null
  const parts = prefix.split('/')
  if (parts.length < 4 || parts[0] !== 'checkpoints') {
    return null
  }
  return parts[2] || null
}

function toFiniteNumber(value: unknown): number | null {
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

function getJobStartedAt(job: TrainingJob): number | null {
  return readTimestamp(job.started_at) ?? readTimestamp(job.created_at)
}

function getJobCompletedAt(job: TrainingJob): number | null {
  return readTimestamp(job.completed_at) ?? readTimestamp(job.summary?.completed_at)
}

function getJobDurationMs(job: TrainingJob): number | null {
  const summaryDuration = toFiniteNumber(job.summary?.duration_ms)
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

function buildAttemptNumbers(jobs: TrainingJob[]): Map<string, number> {
  const counts = new Map<string, number>()
  const result = new Map<string, number>()
  const byCreatedAt = [...jobs].sort((a, b) => (readTimestamp(a.created_at) ?? 0) - (readTimestamp(b.created_at) ?? 0))

  for (const job of byCreatedAt) {
    const next = (counts.get(job.voice_id) ?? 0) + 1
    counts.set(job.voice_id, next)
    result.set(job.job_id, next)
  }

  return result
}

function buildRunSummaries(jobs: TrainingJob[], attemptNumbers: Map<string, number>): RunSummary[] {
  return jobs
    .map((job) => {
      const checkout = getTrainingCheckoutSearch(job)
      const validationPassed = checkout.validation_passed
      const validationRejected = checkout.status === 'rejected'
      return {
        jobId: job.job_id,
        createdAt: job.created_at,
        attemptNumber: attemptNumbers.get(job.job_id) ?? null,
        startedAt: getJobStartedAt(job),
        completedAt: getJobCompletedAt(job),
        durationMs: getJobDurationMs(job),
        status: getTrainingJobDisplayStatus(job),
        championScore: checkout.champion?.score ?? checkout.selected?.score ?? null,
        championEpoch: checkout.champion?.epoch ?? checkout.selected?.epoch ?? null,
        championPreset: checkout.champion?.preset ?? checkout.selected?.preset ?? null,
        validationMessage: checkout.message,
        hasCandidates: checkout.has_candidates,
        validationPassed,
        validationRejected,
      }
    })
    .filter((run) => run.hasCandidates || run.championScore !== null)
}

function getTrainingResetAt(voice: Voice | null): number | null {
  if (!voice) return null
  const raw = voice.labels?.training_reset_at
  if (!raw) return null
  const parsed = Number(raw)
  return Number.isFinite(parsed) && parsed > 0 ? parsed : null
}

function getRunOutcomeLabel(run: RunSummary): string {
  return run.status
}

function getCandidateTitle(candidate: CheckpointCandidate): string {
  if (candidate.isCurrentProduction && candidate.validationPassed) return 'Trusted Current'
  if (candidate.isCurrentProduction) return 'Current Production (Invalidated)'
  if (candidate.isStoredCandidate) return 'Stored Candidate'
  if (candidate.validationPassed) return 'Recommended Candidate'
  if (!candidate.validationPassed) return 'Rejected Candidate'
  return candidate.runName ?? 'Checkpoint'
}

function buildCheckpointCandidates(
  voice: Voice | null,
  jobs: TrainingJob[],
  attemptNumbers: Map<string, number>,
): CheckpointCandidate[] {
  const byPrefix = new Map<string, CheckpointCandidate>()
  const currentPrefix = voice?.checkpoint_r2_prefix ?? null
  const storedCandidatePrefix = voice?.candidate_checkpoint_r2_prefix ?? null

  for (const job of jobs) {
    const checkout = getTrainingCheckoutSearch(job)
    const champion = checkout.champion
    const selected = checkout.selected
    const validationMessage = checkout.message
    const validationPassed = checkout.validation_passed

    const registerCandidate = (input: {
      prefix: string
      epoch: number | null
      score: number | null
      preset: string | null
      message: string | null
      isJobRecommendation: boolean
      validationPassed: boolean
    }) => {
      const existing = byPrefix.get(input.prefix)
      const next: CheckpointCandidate = {
        id: input.prefix,
        prefix: input.prefix,
        epoch: input.epoch,
        score: input.score,
        preset: input.preset,
        message: input.message,
        jobId: job.job_id,
        createdAt: job.created_at,
        completedAt: getJobCompletedAt(job),
        attemptNumber: attemptNumbers.get(job.job_id) ?? null,
        runName: parseRunNameFromPrefix(input.prefix),
        isCurrentProduction: currentPrefix === input.prefix,
        isStoredCandidate: storedCandidatePrefix === input.prefix,
        isJobRecommendation: input.isJobRecommendation,
        validationPassed: input.validationPassed,
      }

      if (!existing) {
        byPrefix.set(input.prefix, next)
        return
      }

      byPrefix.set(input.prefix, {
        ...existing,
        epoch: next.epoch ?? existing.epoch,
        score: next.score ?? existing.score,
        preset: next.preset ?? existing.preset,
        message: next.message ?? existing.message,
        isCurrentProduction: existing.isCurrentProduction || next.isCurrentProduction,
        isStoredCandidate: existing.isStoredCandidate || next.isStoredCandidate,
        isJobRecommendation: existing.isJobRecommendation || next.isJobRecommendation,
        validationPassed: existing.validationPassed || next.validationPassed,
        createdAt: Math.max(existing.createdAt, next.createdAt),
        completedAt: existing.completedAt ?? next.completedAt,
        attemptNumber: existing.attemptNumber ?? next.attemptNumber,
        jobId: existing.jobId ?? next.jobId,
      })
    }

    for (const candidate of checkout.evaluated) {
      registerCandidate({
        prefix: candidate.prefix,
        epoch: candidate.epoch,
        score: candidate.score,
        preset: candidate.preset,
        message: candidate.message,
        isJobRecommendation: champion?.prefix === candidate.prefix,
        validationPassed: candidate.ok,
      })
    }

    if (champion) {
      registerCandidate({
        prefix: champion.prefix,
        epoch: champion.epoch,
        score: champion.score,
        preset: champion.preset,
        message: validationMessage,
        isJobRecommendation: true,
        validationPassed,
      })
    }

    if (selected && selected.prefix !== champion?.prefix) {
      registerCandidate({
        prefix: selected.prefix,
        epoch: selected.epoch,
        score: selected.score,
        preset: selected.preset,
        message: validationMessage,
        isJobRecommendation: false,
        validationPassed,
      })
    }
  }

  if (voice?.checkpoint_r2_prefix && !byPrefix.has(voice.checkpoint_r2_prefix)) {
    byPrefix.set(voice.checkpoint_r2_prefix, {
      id: voice.checkpoint_r2_prefix,
      prefix: voice.checkpoint_r2_prefix,
      epoch: typeof voice.epoch === 'number' ? voice.epoch : null,
      score: typeof voice.checkpoint_score === 'number' ? voice.checkpoint_score : null,
      preset: voice.checkpoint_preset ?? null,
      message:
        typeof voice.checkpoint_score === 'number'
          ? 'Current production checkpoint from a previous validated cycle.'
          : 'Current production checkpoint has no validated result in this cycle',
      jobId: voice.checkpoint_job_id ?? null,
      createdAt: Number(new Date(voice.updated_at ?? voice.created_at).getTime()),
      completedAt: readTimestamp(voice.updated_at ?? voice.created_at),
      attemptNumber: null,
      runName: voice.run_name ?? parseRunNameFromPrefix(voice.checkpoint_r2_prefix),
      isCurrentProduction: true,
      isStoredCandidate: false,
      isJobRecommendation: false,
      validationPassed: false,
    })
  }

  if (voice?.candidate_checkpoint_r2_prefix && !byPrefix.has(voice.candidate_checkpoint_r2_prefix)) {
    byPrefix.set(voice.candidate_checkpoint_r2_prefix, {
      id: voice.candidate_checkpoint_r2_prefix,
      prefix: voice.candidate_checkpoint_r2_prefix,
      epoch: typeof voice.candidate_epoch === 'number' ? voice.candidate_epoch : null,
      score: typeof voice.candidate_score === 'number' ? voice.candidate_score : null,
      preset: voice.candidate_preset ?? null,
      message: 'Validated candidate currently staged for manual promotion.',
      jobId: voice.candidate_job_id ?? null,
      createdAt: Number(new Date(voice.updated_at ?? voice.created_at).getTime()),
      completedAt: readTimestamp(voice.updated_at ?? voice.created_at),
      attemptNumber: null,
      runName: voice.candidate_run_name ?? parseRunNameFromPrefix(voice.candidate_checkpoint_r2_prefix),
      isCurrentProduction: false,
      isStoredCandidate: true,
      isJobRecommendation: true,
      validationPassed: true,
    })
  }

  return [...byPrefix.values()].sort((a, b) => {
    if (a.isCurrentProduction !== b.isCurrentProduction) {
      return a.isCurrentProduction ? -1 : 1
    }
    const aScore = a.score ?? -1
    const bScore = b.score ?? -1
    if (aScore !== bScore) {
      return bScore - aScore
    }
    return b.createdAt - a.createdAt
  })
}

function SettingSlider({
  label,
  value,
  onChange,
  min,
  max,
  step,
}: {
  label: string
  value: number
  onChange: (value: number) => void
  min: number
  max: number
  step: number
}) {
  return (
    <div>
      <div className="flex items-center justify-between mb-1.5">
        <label className="text-subtle text-xs font-medium">{label}</label>
        <span className="text-muted text-[10px] font-mono">{value.toFixed(2)}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full accent-accent"
      />
    </div>
  )
}

function SummaryCard({
  title,
  subtitle,
  detail,
  badge,
  onSelect,
  actionLabel,
}: {
  title: string
  subtitle: string
  detail: string
  badge: string
  onSelect?: () => void
  actionLabel?: string
}) {
  return (
    <div className="rounded-xl border border-edge bg-raised p-4">
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="text-heading text-sm font-semibold">{title}</div>
          <div className="mt-1 text-primary text-sm">{subtitle}</div>
        </div>
        <span className="rounded-full bg-surface px-2 py-0.5 text-[10px] font-mono text-muted">
          {badge}
        </span>
      </div>
      <p className="mt-3 text-[12px] leading-relaxed text-subtle">{detail}</p>
      {onSelect && actionLabel && (
        <button
          onClick={onSelect}
          className="mt-4 inline-flex items-center rounded-lg border border-edge px-3 py-2 text-[11px] font-semibold text-primary transition-colors hover:border-accent hover:text-accent"
          type="button"
        >
          {actionLabel}
        </button>
      )}
    </div>
  )
}

export function VoiceCompare() {
  const { voiceId = '' } = useParams()
  const [searchParams] = useSearchParams()
  const navigate = useNavigate()

  const [voice, setVoice] = useState<Voice | null>(null)
  const [jobs, setJobs] = useState<TrainingJob[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [actionMessage, setActionMessage] = useState('')

  const [text, setText] = useState('')
  const [stylePrompt, setStylePrompt] = useState('')
  const [instruct, setInstruct] = useState('')
  const [seed, setSeed] = useState(123456)
  const [settings, setSettings] = useState<VoiceSettings>(DEFAULT_VOICE_SETTINGS)
  const [selectedCandidateIds, setSelectedCandidateIds] = useState<Set<string>>(new Set())
  const [selectionError, setSelectionError] = useState('')
  const [results, setResults] = useState<Record<string, CompareResult>>({})
  const [generating, setGenerating] = useState(false)
  const [refreshing, setRefreshing] = useState(false)
  const [applyingCandidateId, setApplyingCandidateId] = useState('')
  const [serverTrainingAdvice, setServerTrainingAdvice] = useState<TrainingAdvice | null>(null)

  const requestedJobId = searchParams.get('job') ?? ''
  const trainingResetAt = getTrainingResetAt(voice)
  const cycleJobs = trainingResetAt === null
    ? jobs
    : jobs.filter((job) => job.created_at >= trainingResetAt)
  const cycleJobsAdviceSignature = cycleJobs
    .map((job) => `${job.job_id}:${job.status}:${job.updated_at ?? job.created_at}:${getTrainingCheckoutSearch(job).status}`)
    .join('|')
  const archivedJobsCount = Math.max(0, jobs.length - cycleJobs.length)
  const attemptNumbers = buildAttemptNumbers(cycleJobs)
  const runSummaries = buildRunSummaries(cycleJobs, attemptNumbers)
  const candidates = buildCheckpointCandidates(voice, cycleJobs, attemptNumbers)
  const selectedCandidates = candidates.filter((candidate) => selectedCandidateIds.has(candidate.id))
  const supportsPromptControls = Boolean(voice?.model_size?.includes('1.7'))
  const trustedCurrent = candidates.find((candidate) => candidate.isCurrentProduction && candidate.validationPassed) ?? null
  const storedCandidate = candidates.find((candidate) => candidate.isStoredCandidate) ?? null
  const currentProductionCandidate = candidates.find((candidate) => candidate.isCurrentProduction) ?? null
  const recommendedCandidate =
    storedCandidate ??
    candidates.find((candidate) => candidate.validationPassed && !candidate.isCurrentProduction) ??
    candidates.find((candidate) => candidate.isJobRecommendation && !candidate.isCurrentProduction) ??
    candidates.find((candidate) => !candidate.isCurrentProduction) ??
    null
  const latestRejectedCandidate =
    [...candidates]
      .filter((candidate) => !candidate.validationPassed && !candidate.isCurrentProduction)
      .sort((a, b) => b.createdAt - a.createdAt)[0] ?? null
  const hasWatchableJobs = cycleJobs.some((job) => shouldWatchTrainingJob(job))
  const localTrainingAdvice = buildTrainingAdvice(voice, cycleJobs)
  const trainingAdvice = serverTrainingAdvice ?? localTrainingAdvice

  async function loadData(options?: { silent?: boolean }) {
    if (!voiceId) return
    const silent = options?.silent === true
    if (silent) {
      setRefreshing(true)
    } else {
      setLoading(true)
    }
    setError('')
    try {
      const [voiceData, jobsData] = await Promise.all([
        fetchVoice(voiceId),
        fetchTrainingJobs(voiceId, 100),
      ])
      setVoice(voiceData)
      setSettings(normalizeVoiceSettings(voiceData.settings))
      setJobs(jobsData.jobs)
      setText((prev) => prev || getDefaultCompareText(voiceData.labels.language))
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load checkpoint comparison data')
    } finally {
      setLoading(false)
      setRefreshing(false)
    }
  }

  useEffect(() => {
    void loadData()
  }, [voiceId])

  useEffect(() => {
    if (!voiceId || !hasWatchableJobs) return

    const interval = setInterval(() => {
      void loadData({ silent: true })
    }, 8000)

    return () => clearInterval(interval)
  }, [voiceId, hasWatchableJobs])

  useEffect(() => {
    if (!voice) {
      setServerTrainingAdvice(null)
      return
    }

    const voiceId = voice.voice_id
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
  }, [voice?.voice_id, cycleJobsAdviceSignature])

  useEffect(() => {
    if (candidates.length === 0) return
    setSelectedCandidateIds((prev) => {
      if (prev.size > 0) return prev

      const defaults: string[] = []
      const current = candidates.find((candidate) => candidate.isCurrentProduction)
      if (current) {
        defaults.push(current.id)
      }

      if (requestedJobId) {
        const requested = candidates.filter((candidate) => candidate.jobId === requestedJobId)
        const requestedRecommended = requested.find((candidate) => candidate.isJobRecommendation) ?? requested[0]
        if (requestedRecommended && !defaults.includes(requestedRecommended.id)) {
          defaults.push(requestedRecommended.id)
        }
      }

      if (defaults.length < 2 && recommendedCandidate && !defaults.includes(recommendedCandidate.id)) {
        defaults.push(recommendedCandidate.id)
      }

      if (defaults.length < 2 && latestRejectedCandidate && !defaults.includes(latestRejectedCandidate.id)) {
        defaults.push(latestRejectedCandidate.id)
      }

      for (const candidate of candidates) {
        if (defaults.length >= 2) break
        if (!defaults.includes(candidate.id)) {
          defaults.push(candidate.id)
        }
      }

      return new Set(defaults.slice(0, MAX_COMPARE_CANDIDATES))
    })
  }, [candidates, latestRejectedCandidate, recommendedCandidate, requestedJobId])

  function updateSetting(key: keyof VoiceSettings, value: number) {
    setSettings((prev) => ({ ...prev, [key]: value }))
  }

  function toggleCandidate(candidateId: string) {
    setSelectionError('')
    setSelectedCandidateIds((prev) => {
      const next = new Set(prev)
      if (next.has(candidateId)) {
        next.delete(candidateId)
        return next
      }
      if (next.size >= MAX_COMPARE_CANDIDATES) {
        setSelectionError(`Select up to ${MAX_COMPARE_CANDIDATES} checkpoints at a time.`)
        return prev
      }
      next.add(candidateId)
      return next
    })
  }

  async function generateCandidate(candidate: CheckpointCandidate) {
    const options: SpeechGenerationOptions = {
      stylePrompt: supportsPromptControls ? stylePrompt.trim() || undefined : undefined,
      instruct: supportsPromptControls ? instruct.trim() || undefined : undefined,
      checkpointPrefix: candidate.prefix,
      checkpointEpoch: candidate.epoch ?? undefined,
      seed,
    }

    setResults((prev) => ({
      ...prev,
      [candidate.id]: { status: 'Queued...' },
    }))

    try {
      const asyncJob = await startSpeechGenerationAsync(voiceId, text.trim(), settings, options)
      const result = await pollSpeechGeneration(asyncJob.job_id, (status) => {
        setResults((prev) => ({
          ...prev,
          [candidate.id]: { status },
        }))
      })
      if (result.audio) {
        const bytes = Uint8Array.from(atob(result.audio), (value) => value.charCodeAt(0))
        setResults((prev) => ({
          ...prev,
          [candidate.id]: {
            status: 'Completed',
            blob: new Blob([bytes], { type: 'audio/wav' }),
          },
        }))
      }
    } catch (err) {
      setResults((prev) => ({
        ...prev,
        [candidate.id]: {
          status: 'Failed',
          error: err instanceof Error ? err.message : 'Generation failed',
        },
      }))
    }
  }

  async function handleGenerateSelected() {
    if (!voiceId || !text.trim() || selectedCandidates.length === 0) return

    setGenerating(true)
    await Promise.all(selectedCandidates.map((candidate) => generateCandidate(candidate)))
    setGenerating(false)
  }

  async function handleApplyCandidate(candidate: CheckpointCandidate) {
    if (!candidate.jobId || candidate.isCurrentProduction) return
    setApplyingCandidateId(candidate.id)
    setError('')
    setActionMessage('')
    try {
      await promoteTrainingCheckpoint(candidate.jobId, candidate.prefix)
      setActionMessage(`${candidate.runName ?? 'checkpoint'} epoch ${candidate.epoch ?? 'n/a'} is now active.`)
      await loadData({ silent: true })
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to apply checkpoint')
    } finally {
      setApplyingCandidateId('')
    }
  }

  function selectCurrentAndBest() {
    const next = new Set<string>()
    const current = trustedCurrent
    const bestAlternative = recommendedCandidate
    if (current) next.add(current.id)
    if (bestAlternative) next.add(bestAlternative.id)
    setSelectionError('')
    setSelectedCandidateIds(next)
  }

  function selectRecommendedAndRejected() {
    const next = new Set<string>()
    if (recommendedCandidate) next.add(recommendedCandidate.id)
    if (latestRejectedCandidate) next.add(latestRejectedCandidate.id)
    setSelectionError('')
    setSelectedCandidateIds(next)
  }

  if (loading) {
    return (
      <div className="space-y-6 animate-pulse">
        <div className="h-7 bg-raised rounded w-48" />
        <div className="h-20 bg-raised rounded-xl" />
        <div className="grid lg:grid-cols-[320px_1fr] gap-6">
          <div className="h-96 bg-raised rounded-xl" />
          <div className="h-96 bg-raised rounded-xl" />
        </div>
      </div>
    )
  }

  if (error || !voice) {
    return (
      <div className="text-center py-16 space-y-4">
        <div className="text-error text-sm">{error || 'Voice not found'}</div>
        <button
          onClick={() => navigate(voiceId ? `/voices/${voiceId}` : '/voices')}
          className="text-accent text-sm hover:text-accent-light"
          type="button"
        >
          Back
        </button>
      </div>
    )
  }

  return (
    <div className="space-y-8">
      <button
        onClick={() => navigate(`/voices/${voice.voice_id}`)}
        className="text-subtle text-sm hover:text-accent transition-colors inline-flex items-center gap-1"
        type="button"
      >
        <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
          <polyline points="15 18 9 12 15 6" />
        </svg>
        Back to Voice
      </button>

      <div className="flex flex-col gap-2">
        <h1 className="text-heading text-2xl font-bold">Checkpoint Compare</h1>
        <p className="text-subtle text-sm">
          {voice.name} · listen to trusted, recommended, and rejected checkpoints side by side before accepting a new version.
        </p>
        <div className="flex flex-wrap gap-3 text-[11px] font-mono text-muted">
          <span>trusted={voice.run_name ?? 'none'}</span>
          <span>epoch={typeof voice.epoch === 'number' ? voice.epoch : 'none'}</span>
          <span>cycle_runs={runSummaries.length}</span>
          <span>candidate_checkpoints={candidates.length}</span>
          {refreshing && <span>refreshing=on</span>}
          {trainingResetAt !== null && <span>fresh_cycle=on</span>}
        </div>
        <div className="pt-1">
          <button
            onClick={() => void loadData({ silent: true })}
            className="inline-flex items-center rounded-lg border border-edge px-3 py-2 text-[11px] font-semibold text-primary transition-colors hover:border-accent hover:text-accent"
            type="button"
          >
            Refresh Compare Data
          </button>
        </div>
      </div>

      {actionMessage && (
        <div className="rounded-lg border border-accent/20 bg-accent-dim px-4 py-3 text-accent text-sm">
          {actionMessage}
        </div>
      )}

      <div className="grid md:grid-cols-3 gap-4">
        <SummaryCard
          title="Trusted Now"
          subtitle={trustedCurrent ? `${trustedCurrent.runName ?? 'current'} · run ${trustedCurrent.attemptNumber ?? 'n/a'} · epoch ${trustedCurrent.epoch ?? 'n/a'}` : currentProductionCandidate ? `${currentProductionCandidate.runName ?? 'current'} · invalidated` : 'No trusted checkpoint'}
          detail={trustedCurrent?.message ?? currentProductionCandidate?.message ?? 'A fresh cycle is in progress. Existing discarded checkpoints are not treated as trusted.'}
          badge={trustedCurrent ? 'current' : currentProductionCandidate ? 'invalidated' : 'empty'}
          onSelect={trustedCurrent ? () => setSelectedCandidateIds(new Set([trustedCurrent.id])) : undefined}
          actionLabel={trustedCurrent ? 'Listen' : undefined}
        />
        <SummaryCard
          title={storedCandidate ? 'Candidate Slot' : 'Recommended'}
          subtitle={recommendedCandidate ? `${recommendedCandidate.runName ?? 'candidate'} · run ${recommendedCandidate.attemptNumber ?? 'n/a'} · epoch ${recommendedCandidate.epoch ?? 'n/a'}` : 'No recommendation yet'}
          detail={recommendedCandidate?.message ?? 'A recommendation appears here once the current cycle produces candidates.'}
          badge={storedCandidate ? 'candidate' : recommendedCandidate?.validationPassed ? 'validated' : recommendedCandidate ? 'candidate' : 'waiting'}
          onSelect={recommendedCandidate ? () => setSelectedCandidateIds(new Set([recommendedCandidate.id])) : undefined}
          actionLabel={recommendedCandidate ? 'Listen' : undefined}
        />
        <SummaryCard
          title="Latest Rejected"
          subtitle={latestRejectedCandidate ? `${latestRejectedCandidate.runName ?? 'rejected'} · run ${latestRejectedCandidate.attemptNumber ?? 'n/a'} · epoch ${latestRejectedCandidate.epoch ?? 'n/a'}` : 'No rejected checkpoint in this cycle'}
          detail={latestRejectedCandidate?.message ?? 'If a checkpoint is trained but rejected by validation, it will still show up here for manual listening.'}
          badge={latestRejectedCandidate ? 'rejected' : 'waiting'}
          onSelect={latestRejectedCandidate ? () => setSelectedCandidateIds(new Set([latestRejectedCandidate.id])) : undefined}
          actionLabel={latestRejectedCandidate ? 'Listen' : undefined}
        />
      </div>

      <TrainingAdviceCard
        voiceId={voice.voice_id}
        advice={trainingAdvice}
        compact
        showCompareLink={false}
      />

      <div className="grid lg:grid-cols-[340px_1fr] gap-6">
        <div className="space-y-6">
          <div className="bg-raised border border-edge rounded-xl p-5 space-y-4">
            <div>
              <h2 className="text-heading font-semibold text-sm">Audition Controls</h2>
              <p className="text-subtle text-xs mt-1">
                Generate the same script across multiple checkpoints to compare tone retention, pacing, and speaker stability.
              </p>
              {trainingResetAt !== null && (
                <p className="text-warning text-[11px] mt-2">
                  Fresh-cycle mode is active. Older discarded 0.6B runs are hidden from this page by default.
                </p>
              )}
              {archivedJobsCount > 0 && (
                <p className="text-muted text-[11px] mt-2">
                  Hidden older runs: {archivedJobsCount}
                </p>
              )}
            </div>

            <div>
              <label className="text-subtle text-xs font-medium mb-1.5 block">Comparison Text</label>
              <textarea
                value={text}
                onChange={(e) => setText(e.target.value)}
                rows={6}
                className="w-full bg-surface border border-edge rounded-lg px-3 py-2 text-sm text-primary placeholder:text-muted focus:border-accent transition-colors resize-none"
              />
            </div>

            <div>
              <label className="text-subtle text-xs font-medium mb-1.5 block">Seed</label>
              <input
                type="number"
                value={seed}
                onChange={(e) => setSeed(parseInt(e.target.value, 10) || 1)}
                className="w-full bg-surface border border-edge rounded-lg px-3 py-2 text-sm text-primary font-mono focus:border-accent transition-colors"
              />
            </div>

            <div className="space-y-4 pt-1">
              <SettingSlider label="Stability" value={settings.stability} onChange={(value) => updateSetting('stability', value)} min={0} max={1} step={0.01} />
              <SettingSlider label="Similarity Boost" value={settings.similarity_boost} onChange={(value) => updateSetting('similarity_boost', value)} min={0} max={1} step={0.01} />
              <SettingSlider label="Style" value={settings.style} onChange={(value) => updateSetting('style', value)} min={0} max={1} step={0.01} />
              <SettingSlider label="Speed" value={settings.speed} onChange={(value) => updateSetting('speed', value)} min={0.5} max={2} step={0.05} />
            </div>

            <div className="pt-1 border-t border-edge space-y-3">
              <div>
                <label className="text-subtle text-xs font-medium mb-1.5 block">Style Prompt</label>
                <textarea
                  value={stylePrompt}
                  onChange={(e) => setStylePrompt(e.target.value)}
                  disabled={!supportsPromptControls}
                  rows={3}
                  placeholder="Preserve the speaker's distinctive phrasing, measured pauses, and sentence-ending emphasis."
                  className="w-full bg-surface border border-edge rounded-lg px-3 py-2 text-sm text-primary placeholder:text-muted disabled:opacity-50 disabled:cursor-not-allowed focus:border-accent transition-colors resize-none"
                />
              </div>
              <div>
                <label className="text-subtle text-xs font-medium mb-1.5 block">Instruct</label>
                <textarea
                  value={instruct}
                  onChange={(e) => setInstruct(e.target.value)}
                  disabled={!supportsPromptControls}
                  rows={3}
                  placeholder="Keep the original speaking habit and intonation instead of smoothing it into a generic narration tone."
                  className="w-full bg-surface border border-edge rounded-lg px-3 py-2 text-sm text-primary placeholder:text-muted disabled:opacity-50 disabled:cursor-not-allowed focus:border-accent transition-colors resize-none"
                />
              </div>
            </div>

            <div className="flex flex-wrap gap-2">
              <button
                onClick={selectCurrentAndBest}
                className="inline-flex items-center rounded-lg border border-edge px-3 py-2 text-[11px] font-semibold text-primary transition-colors hover:border-accent hover:text-accent"
                type="button"
              >
                Select Trusted + Recommended
              </button>
              <button
                onClick={selectRecommendedAndRejected}
                className="inline-flex items-center rounded-lg border border-edge px-3 py-2 text-[11px] font-semibold text-primary transition-colors hover:border-accent hover:text-accent"
                type="button"
              >
                Select Recommended + Rejected
              </button>
              <button
                onClick={() => setSelectedCandidateIds(new Set())}
                className="inline-flex items-center rounded-lg border border-edge px-3 py-2 text-[11px] font-semibold text-muted transition-colors hover:text-primary"
                type="button"
              >
                Clear Selection
              </button>
            </div>

            {selectionError && (
              <div className="rounded-lg border border-warning/20 bg-warning-dim px-3 py-2 text-warning text-xs">
                {selectionError}
              </div>
            )}

            <button
              onClick={handleGenerateSelected}
              disabled={generating || selectedCandidates.length === 0 || !text.trim()}
              className="w-full bg-accent hover:bg-accent-light text-void font-semibold text-sm py-2.5 rounded-lg disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
              type="button"
            >
              {generating ? 'Generating…' : `Generate ${selectedCandidates.length} Selected Checkpoint${selectedCandidates.length === 1 ? '' : 's'}`}
            </button>
          </div>

          <div className="bg-raised border border-edge rounded-xl p-5">
            <h2 className="text-heading font-semibold text-sm mb-4">Recent Attempts</h2>
            <div className="space-y-3">
              {runSummaries.map((run) => (
                <div key={run.jobId} className="rounded-lg border border-edge bg-surface px-3 py-3">
                  <div className="flex items-center justify-between gap-3">
                    <div>
                      <div className="text-primary text-xs font-mono">
                        run #{run.attemptNumber ?? '—'} · {run.jobId.slice(0, 12)}
                      </div>
                      <div className="mt-1 text-[10px] font-mono text-muted">
                        created={formatDateTime(run.createdAt)}
                      </div>
                    </div>
                    <div className="text-right text-[10px] font-mono text-muted">
                      {run.completedAt !== null ? `done ${formatTime(run.completedAt)}` : run.startedAt !== null ? `started ${formatTime(run.startedAt)}` : 'pending'}
                    </div>
                  </div>
                  <div className="mt-2 flex flex-wrap gap-2 text-[10px] font-mono text-muted">
                    <span>status={getRunOutcomeLabel(run)}</span>
                    <span>score={run.championScore !== null ? run.championScore.toFixed(3) : 'n/a'}</span>
                    <span>epoch={run.championEpoch ?? 'n/a'}</span>
                    <span>preset={run.championPreset ?? 'n/a'}</span>
                    {run.durationMs !== null && <span>duration={formatDurationMs(run.durationMs)}</span>}
                  </div>
                  {run.validationMessage && (
                    <div className="mt-2 text-[11px] text-subtle">{run.validationMessage}</div>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="space-y-6">
          <div className="bg-raised border border-edge rounded-xl p-5">
            <div className="flex items-center justify-between gap-4 mb-4">
              <div>
                <h2 className="text-heading font-semibold text-sm">Checkpoint Candidates</h2>
                <p className="text-subtle text-xs mt-1">Recommended versions are highlighted, but rejected checkpoints are also kept here for manual listening.</p>
              </div>
              <div className="text-muted text-[10px] font-mono">max_select={MAX_COMPARE_CANDIDATES}</div>
            </div>

            {candidates.length === 0 ? (
              <div className="rounded-xl border border-dashed border-edge bg-surface px-4 py-8 text-center">
                <div className="text-primary text-sm font-semibold">No checkpoints in the current training cycle yet</div>
                <p className="mt-2 text-subtle text-sm">
                  Start a fresh training run first. As soon as checkpoints are saved, this page will surface trusted, recommended, and rejected candidates here.
                </p>
                <button
                  onClick={() => navigate(`/voices/${voiceId}/training`)}
                  className="mt-4 inline-flex items-center rounded-lg bg-accent px-3 py-2 text-[11px] font-semibold text-void transition-colors hover:bg-accent-light"
                  type="button"
                >
                  Open Training
                </button>
              </div>
            ) : (
            <div className="grid xl:grid-cols-2 gap-3">
              {candidates.map((candidate) => {
                const selected = selectedCandidateIds.has(candidate.id)
                return (
                  <button
                    key={candidate.id}
                    onClick={() => toggleCandidate(candidate.id)}
                    className={`rounded-xl border p-4 text-left transition-colors ${selected ? 'border-accent bg-accent-dim/20' : 'border-edge bg-surface hover:border-accent/40'}`}
                    type="button"
                  >
                    <div className="flex items-start justify-between gap-3">
                      <div>
                        <div className="text-primary text-sm font-semibold">
                          {getCandidateTitle(candidate)}
                        </div>
                        <div className="text-subtle text-[11px] mt-1">
                          {candidate.runName ?? candidate.jobId?.slice(0, 8) ?? 'checkpoint'}
                        </div>
                        <div className="text-muted text-[10px] font-mono mt-1">
                          run={candidate.attemptNumber ?? 'n/a'} epoch={candidate.epoch ?? 'n/a'} score={candidate.score !== null ? candidate.score.toFixed(3) : 'n/a'}
                        </div>
                        <div className="text-muted text-[10px] font-mono mt-1">
                          {formatDate(candidate.createdAt)} {formatTime(candidate.createdAt)}
                          {candidate.completedAt !== null ? ` · done ${formatTime(candidate.completedAt)}` : ''}
                        </div>
                      </div>
                      <input
                        type="checkbox"
                        checked={selected}
                        onChange={() => toggleCandidate(candidate.id)}
                        className="accent-accent mt-1"
                      />
                    </div>

                    <div className="mt-3 flex flex-wrap gap-2 text-[10px] font-mono">
                      {candidate.isCurrentProduction && (
                        <span className={`rounded-full px-2 py-0.5 ${candidate.validationPassed ? 'bg-accent-dim text-accent' : 'bg-error-dim text-error'}`}>
                          {candidate.validationPassed ? 'current' : 'invalidated'}
                        </span>
                      )}
                      {candidate.isJobRecommendation && (
                        <span className="rounded-full bg-warning-dim px-2 py-0.5 text-warning">recommended</span>
                      )}
                      {candidate.isStoredCandidate && (
                        <span className="rounded-full bg-accent-dim px-2 py-0.5 text-accent">candidate</span>
                      )}
                      {candidate.validationPassed && (
                        <span className="rounded-full bg-accent-dim px-2 py-0.5 text-accent">passed</span>
                      )}
                      {!candidate.validationPassed && !candidate.isCurrentProduction && (
                        <span className="rounded-full bg-error-dim px-2 py-0.5 text-error">rejected</span>
                      )}
                      {candidate.preset && (
                        <span className="rounded-full bg-raised px-2 py-0.5 text-muted">{candidate.preset}</span>
                      )}
                    </div>

                    {candidate.message && (
                      <p className="mt-3 text-[11px] leading-relaxed text-subtle">
                        {candidate.message}
                      </p>
                    )}
                  </button>
                )
              })}
            </div>
            )}
          </div>

          <div className="bg-raised border border-edge rounded-xl p-5">
            <h2 className="text-heading font-semibold text-sm mb-4">Side-by-Side Listening</h2>
            {selectedCandidates.length === 0 ? (
              <div className="text-subtle text-sm py-8 text-center">Select checkpoints above to audition them together.</div>
            ) : (
              <div className="grid xl:grid-cols-2 gap-4">
                {selectedCandidates.map((candidate) => {
                  const result = results[candidate.id]
                  return (
                    <div key={candidate.id} className="rounded-xl border border-edge bg-surface p-4">
                      <div className="flex items-start justify-between gap-3 mb-3">
                        <div>
                          <div className="text-primary font-semibold text-sm">
                            {candidate.isCurrentProduction && !candidate.validationPassed ? 'Current Production (Invalidated)' : candidate.isCurrentProduction ? 'Current Production' : candidate.runName ?? 'Candidate'}
                          </div>
                          <div className="text-muted text-[10px] font-mono mt-1">
                            run={candidate.attemptNumber ?? 'n/a'} epoch={candidate.epoch ?? 'n/a'} score={candidate.score !== null ? candidate.score.toFixed(3) : 'n/a'} preset={candidate.preset ?? 'n/a'}
                          </div>
                          <div className="text-muted text-[10px] font-mono mt-1">
                            {formatDate(candidate.createdAt)} {formatTime(candidate.createdAt)}
                          </div>
                        </div>
                        <div className="text-[10px] font-mono text-muted">{result?.status ?? 'Idle'}</div>
                      </div>

                      <AudioPlayer blob={result?.blob} generating={generating && !result?.blob && result?.status !== 'Failed'} />

                      {result?.error && (
                        <div className="mt-3 rounded-lg border border-error/20 bg-error-dim px-3 py-2 text-error text-xs">
                          {result.error}
                        </div>
                      )}

                      {candidate.message && (
                        <div className="mt-3 text-[11px] text-subtle">
                          {candidate.message}
                        </div>
                      )}

                      {candidate.jobId && !candidate.isCurrentProduction && (
                        <button
                          onClick={() => void handleApplyCandidate(candidate)}
                          disabled={applyingCandidateId === candidate.id}
                          className="mt-3 inline-flex items-center rounded-lg border border-edge px-3 py-2 text-[11px] font-semibold text-primary transition-colors hover:border-accent hover:text-accent disabled:opacity-50"
                          type="button"
                        >
                          {applyingCandidateId === candidate.id ? 'Applying…' : 'Apply This Checkpoint'}
                        </button>
                      )}
                    </div>
                  )
                })}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
