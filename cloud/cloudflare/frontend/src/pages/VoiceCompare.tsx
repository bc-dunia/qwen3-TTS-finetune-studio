import { useEffect, useState } from 'react'
import { useNavigate, useParams, useSearchParams } from 'react-router'
import { TrainingAdviceCard } from '../components/TrainingAdviceCard'
import { DecisionHeader } from '../components/compare/DecisionHeader'
import { ComparisonSetupForm } from '../components/compare/ComparisonSetupForm'
import { CandidateGrid } from '../components/compare/CandidateGrid'
import { CompareTray } from '../components/compare/CompareTray'
import { RecommendationBanner } from '../components/compare/RecommendationBanner'
import { MAX_COMPARE_CANDIDATES } from '../components/compare/types'
import type { CheckpointCandidate, RunSummary, CompareResult } from '../components/compare/types'
import {
  DEFAULT_VOICE_SETTINGS,
  fetchTrainingAdvice,
  fetchTrainingJobs,
  fetchVoice,
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
import { parseRunNameFromCheckpointPrefix, readNumber } from '../lib/training-domain'


// ── Utility functions ─────────────────────────────────────────────────────────

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
      return 'こんにちは。このサンプルは、チェックポイントごとの話者類似度と話し方の保持を比較するための文章です。'
    case 'zh':
      return '你好。这段样例用于比较各个检查点的音色、语气和整体稳定性。'
    case 'ko':
    default:
      return '안녕하세요. 이 샘플은 체크포인트별 화자 유사도와 말투 보존을 비교하기 위한 문장입니다.'
  }
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
      toneScore: number | null
      speedScore: number | null
      styleScore: number | null
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
        runName: input.prefix ? parseRunNameFromCheckpointPrefix(input.prefix) : null,
        isCurrentProduction: currentPrefix === input.prefix,
        isStoredCandidate: storedCandidatePrefix === input.prefix,
        isJobRecommendation: input.isJobRecommendation,
        validationPassed: input.validationPassed,
        toneScore: input.toneScore,
        speedScore: input.speedScore,
        styleScore: input.styleScore,
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
        toneScore: next.toneScore ?? existing.toneScore,
        speedScore: next.speedScore ?? existing.speedScore,
        styleScore: next.styleScore ?? existing.styleScore,
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
        toneScore: candidate.tone_score ?? null,
        speedScore: candidate.speed_score ?? null,
        styleScore: candidate.style_score ?? null,
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
        toneScore: null,
        speedScore: null,
        styleScore: null,
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
        toneScore: null,
        speedScore: null,
        styleScore: null,
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
      runName: voice.run_name ?? (voice.checkpoint_r2_prefix ? parseRunNameFromCheckpointPrefix(voice.checkpoint_r2_prefix) : null),
      isCurrentProduction: true,
      isStoredCandidate: false,
      isJobRecommendation: false,
      validationPassed: typeof voice.checkpoint_score === 'number',
      toneScore: null,
      speedScore: null,
      styleScore: null,
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
      runName: voice.candidate_run_name ?? (voice.candidate_checkpoint_r2_prefix ? parseRunNameFromCheckpointPrefix(voice.candidate_checkpoint_r2_prefix) : null),
      isCurrentProduction: false,
      isStoredCandidate: true,
      isJobRecommendation: true,
      validationPassed: true,
      toneScore: null,
      speedScore: null,
      styleScore: null,
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

// ── Recent Attempts Disclosure ────────────────────────────────────────────────

function RecentAttemptsDisclosure({ runSummaries }: { runSummaries: RunSummary[] }) {
  const [open, setOpen] = useState(false)

  if (runSummaries.length === 0) return null

  return (
    <div className="rounded-xl border border-edge bg-raised p-5">
      <button
        onClick={() => setOpen((prev) => !prev)}
        className="flex items-center gap-2 w-full"
        type="button"
      >
        <svg
          className={`w-3.5 h-3.5 text-muted transition-transform duration-200 ${open ? 'rotate-90' : ''}`}
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2.5"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <polyline points="9 18 15 12 9 6" />
        </svg>
        <h2 className="text-heading font-semibold text-sm">
          Recent Attempts ({runSummaries.length})
        </h2>
      </button>

      {open && (
        <div className="mt-4 space-y-3 animate-slide-up">
          {runSummaries.map((run) => (
            <div key={run.jobId} className="rounded-lg border border-edge bg-surface px-3 py-3">
              <div className="flex items-center justify-between gap-3">
                <div>
                  <div className="text-primary text-xs font-mono">
                    run #{run.attemptNumber ?? '\u2014'} \u00B7 {run.jobId.slice(0, 12)}
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
                <span>status={run.status}</span>
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
      )}
    </div>
  )
}

// ── Main Component ────────────────────────────────────────────────────────────

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

  function handleSelectPreset(preset: 'trusted-recommended' | 'recommended-rejected' | 'clear') {
    setSelectionError('')
    if (preset === 'clear') {
      setSelectedCandidateIds(new Set())
      return
    }
    const next = new Set<string>()
    if (preset === 'trusted-recommended') {
      if (trustedCurrent) next.add(trustedCurrent.id)
      if (recommendedCandidate) next.add(recommendedCandidate.id)
    } else {
      if (recommendedCandidate) next.add(recommendedCandidate.id)
      if (latestRejectedCandidate) next.add(latestRejectedCandidate.id)
    }
    setSelectedCandidateIds(next)
  }

  // ── Loading / Error states ──────────────────────────────────────────────────

  if (loading) {
    return (
      <div className="space-y-6 animate-pulse">
        <div className="h-7 bg-raised rounded w-48" />
        <div className="h-20 bg-raised rounded-xl" />
        <div className="h-96 bg-raised rounded-xl" />
      </div>
    )
  }

  if (!voice) {
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

  // ── Render ──────────────────────────────────────────────────────────────────

  return (
    <div className="space-y-6 pb-28 lg:pb-24">
      {error && (
        <div className="rounded-lg border border-error/20 bg-error-dim px-3 py-2 text-sm text-error">{error}</div>
      )}

      {actionMessage && (
        <div className="rounded-lg border border-accent/20 bg-accent-dim px-4 py-3 text-accent text-sm">
          {actionMessage}
        </div>
      )}

      <DecisionHeader
        trustedCandidate={trustedCurrent}
        recommendedCandidate={recommendedCandidate}
        currentProductionCandidate={currentProductionCandidate}
        refreshing={refreshing}
        onRefresh={() => void loadData({ silent: true })}
      />

      <TrainingAdviceCard
        voiceId={voice.voice_id}
        advice={trainingAdvice}
        compact
        showCompareLink={false}
      />

      <ComparisonSetupForm
        text={text}
        onTextChange={setText}
        seed={seed}
        onSeedChange={setSeed}
        settings={settings}
        onSettingChange={updateSetting}
        stylePrompt={stylePrompt}
        onStylePromptChange={setStylePrompt}
        instruct={instruct}
        onInstructChange={setInstruct}
        supportsPromptControls={supportsPromptControls}
        trainingResetAt={trainingResetAt}
        archivedJobsCount={archivedJobsCount}
      />

      <CandidateGrid
        candidates={candidates}
        selectedIds={selectedCandidateIds}
        onToggle={toggleCandidate}
        results={results}
        generating={generating}
        applyingCandidateId={applyingCandidateId}
        onApply={handleApplyCandidate}
        selectionError={selectionError}
        onSelectPreset={handleSelectPreset}
      />

      <RecentAttemptsDisclosure runSummaries={runSummaries} />

      <RecommendationBanner
        trustedCandidate={trustedCurrent}
        recommendedCandidate={recommendedCandidate}
        results={results}
        onApply={handleApplyCandidate}
        applyingCandidateId={applyingCandidateId}
      />

      <CompareTray
        selectedCandidates={selectedCandidates}
        onRemove={(id) => {
          setSelectedCandidateIds((prev) => {
            const next = new Set(prev)
            next.delete(id)
            return next
          })
        }}
        onGenerate={handleGenerateSelected}
        generating={generating}
        hasText={Boolean(text.trim())}
        maxSelect={MAX_COMPARE_CANDIDATES}
      />
    </div>
  )
}
