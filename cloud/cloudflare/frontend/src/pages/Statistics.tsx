import { useEffect, useMemo, useRef, useState } from 'react'
import { Link } from 'react-router'
import {
  fetchVoices,
  fetchAllTrainingJobs,
  type Voice,
  type TrainingJob,
  formatDate,
  formatDurationMs,
} from '../lib/api'
import { formatScore, scoreColor } from '../lib/voiceScoreUi'
import { LineChart } from '../components/charts/LineChart'
import { BarChart } from '../components/charts/BarChart'
import { MiniSparkline } from '../components/charts/MiniSparkline'

type SortKey = 'name' | 'score' | 'delta' | 'jobs' | 'duration' | 'last'
type SortDirection = 'asc' | 'desc'

type VoicePerformanceRow = {
  voiceId: string
  name: string
  modelSize: string
  currentScore: number | null
  firstScore: number | null
  bestScore: number | null
  scoreDelta: number | null
  completedJobs: number
  totalJobs: number
  avgDurationMs: number | null
  lastTrainedAt: number | null
  sparkline: number[]
}

function readNumber(value: unknown): number | null {
  if (typeof value === 'number' && Number.isFinite(value)) return value
  if (typeof value === 'string' && value.trim()) {
    const parsed = Number(value)
    if (Number.isFinite(parsed)) return parsed
  }
  return null
}

function median(values: number[]): number | null {
  if (values.length === 0) return null
  const sorted = [...values].sort((a, b) => a - b)
  const mid = Math.floor(sorted.length / 2)
  if (sorted.length % 2 === 0) {
    return (sorted[mid - 1] + sorted[mid]) / 2
  }
  return sorted[mid]
}

function jobDuration(job: TrainingJob): number | null {
  if (typeof job.started_at !== 'number' || typeof job.completed_at !== 'number') return null
  const ms = job.completed_at - job.started_at
  return Number.isFinite(ms) && ms >= 0 ? ms : null
}

function jobScore(job: TrainingJob): number | null {
  return readNumber(job.summary?.candidate_score)
}

function parseEvaluatedCheckpointOk(summary: Record<string, unknown> | undefined): boolean[] {
  const value = summary?.evaluated_checkpoints
  if (!Array.isArray(value)) return []
  return value
    .map((item) => (typeof item === 'object' && item !== null ? item as Record<string, unknown> : null))
    .filter((item): item is Record<string, unknown> => item !== null)
    .map((item) => item.ok === true)
}

function parseSelectedEpoch(summary: Record<string, unknown> | undefined): number | null {
  const direct = readNumber(summary?.selected_epoch)
    ?? readNumber(summary?.champion_epoch)
    ?? readNumber(summary?.best_epoch)
    ?? readNumber(summary?.epoch)
  if (direct !== null) return Math.max(0, Math.round(direct))

  const prefix = typeof summary?.selected_checkpoint_r2_prefix === 'string'
    ? summary.selected_checkpoint_r2_prefix
    : typeof summary?.champion_checkpoint_r2_prefix === 'string'
      ? summary.champion_checkpoint_r2_prefix
      : null
  if (!prefix) return null
  const match = prefix.match(/epoch-(\d+)/)
  if (!match) return null
  return Number(match[1])
}

function relativeTime(timestamp: number | null): string {
  if (timestamp === null) return '—'
  const delta = Date.now() - timestamp
  const minute = 60_000
  const hour = 60 * minute
  const day = 24 * hour
  if (delta < minute) return 'just now'
  if (delta < hour) return `${Math.floor(delta / minute)}m ago`
  if (delta < day) return `${Math.floor(delta / hour)}h ago`
  return `${Math.floor(delta / day)}d ago`
}

const SERIES_COLORS = [
  'var(--color-accent)',
  'var(--color-warning)',
  'var(--color-error)',
  '#3b82f6',
  '#14b8a6',
  '#f97316',
  '#22c55e',
  '#eab308',
]

export function Statistics() {
  const [voices, setVoices] = useState<Voice[]>([])
  const [allJobs, setAllJobs] = useState<TrainingJob[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [selectedVoice, setSelectedVoice] = useState('')
  const selectedVoiceRef = useRef(selectedVoice)
  selectedVoiceRef.current = selectedVoice
  const [refreshing, setRefreshing] = useState(false)
  const [sortKey, setSortKey] = useState<SortKey>('last')
  const [sortDirection, setSortDirection] = useState<SortDirection>('desc')

  useEffect(() => {
    let cancelled = false
    async function load() {
      try {
        const [voicesData, jobsData] = await Promise.all([
          fetchVoices(),
          fetchAllTrainingJobs(2000),
        ])
        if (!cancelled) {
          setVoices(voicesData.voices)
          setAllJobs(jobsData.jobs)
          setError('')
          if (selectedVoiceRef.current && !voicesData.voices.some((v) => v.voice_id === selectedVoiceRef.current)) {
            setSelectedVoice('')
          }
        }
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : 'Failed to load')
      } finally {
        if (!cancelled) {
          setLoading(false)
          setRefreshing(false)
        }
      }
    }
    void load()
    return () => { cancelled = true }
  }, [])

  const voicesById = useMemo(() => new Map(voices.map((voice) => [voice.voice_id, voice])), [voices])

  const completedJobs = useMemo(
    () => allJobs.filter((job) => job.status === 'completed'),
    [allJobs],
  )

  const filteredAllJobs = useMemo(
    () => (selectedVoice ? allJobs.filter((job) => job.voice_id === selectedVoice) : allJobs),
    [allJobs, selectedVoice],
  )

  const filteredJobs = useMemo(
    () => (selectedVoice ? completedJobs.filter((job) => job.voice_id === selectedVoice) : completedJobs),
    [completedJobs, selectedVoice],
  )

  const voicesInScope = useMemo(() => {
    if (!selectedVoice) return voices
    return voices.filter((voice) => voice.voice_id === selectedVoice)
  }, [voices, selectedVoice])

  const voicesTrained = useMemo(() => {
    const completedByVoice = new Set(filteredAllJobs.filter((job) => job.status === 'completed').map((job) => job.voice_id))
    return voicesInScope.filter((voice) => voice.status === 'ready' || completedByVoice.has(voice.voice_id)).length
  }, [filteredAllJobs, voicesInScope])

  const completedCount = useMemo(
    () => filteredAllJobs.filter((job) => job.status === 'completed').length,
    [filteredAllJobs],
  )

  const totalCount = useMemo(() => filteredAllJobs.length, [filteredAllJobs])

  const completionRate = useMemo(
    () => (totalCount > 0 ? (completedCount / totalCount) * 100 : 0),
    [completedCount, totalCount],
  )

  const medianTrainingTime = useMemo(() => {
    const durations = filteredJobs
      .map((job) => jobDuration(job))
      .filter((duration): duration is number => duration !== null)
    return median(durations)
  }, [filteredJobs])

  const bestScore = useMemo(() => {
    const voiceScores = voicesInScope
      .map((voice) => voice.checkpoint_score)
      .filter((score): score is number => typeof score === 'number' && Number.isFinite(score))
    const jobScores = filteredJobs
      .map((job) => jobScore(job))
      .filter((score): score is number => score !== null)
    const all = [...voiceScores, ...jobScores]
    return all.length > 0 ? Math.max(...all) : null
  }, [voicesInScope, filteredJobs])

  const scoreImprovement = useMemo(() => {
    const voiceMap = new Map<string, number[]>()
    filteredJobs
      .slice()
      .sort((a, b) => a.created_at - b.created_at)
      .forEach((job) => {
        const score = jobScore(job)
        if (score === null) return
        const list = voiceMap.get(job.voice_id) ?? []
        list.push(score)
        voiceMap.set(job.voice_id, list)
      })
    const deltas = Array.from(voiceMap.values())
      .filter((scores) => scores.length >= 2)
      .map((scores) => Math.max(...scores) - scores[0])
    return median(deltas)
  }, [filteredJobs])

  const scoreProgressionData = useMemo(() => {
    function addRunningBest(rawValues: (number | null)[]): (number | null)[] {
      let best: number | null = null
      return rawValues.map((v) => {
        if (v !== null && (best === null || v > best)) best = v
        return best
      })
    }

    if (selectedVoice) {
      const rawValues = filteredJobs
        .slice()
        .sort((a, b) => (a.attempt_index ?? a.created_at) - (b.attempt_index ?? b.created_at))
        .map((job) => jobScore(job))
      return {
        label: 'Attempt index',
        series: [
          {
            name: voicesById.get(selectedVoice)?.name ?? 'Selected voice',
            color: 'var(--color-muted)',
            values: rawValues,
          },
          {
            name: 'Best so far',
            color: 'var(--color-accent)',
            values: addRunningBest(rawValues),
          },
        ],
      }
    }

    const byVoice = new Map<string, TrainingJob[]>()
    filteredJobs.forEach((job) => {
      const list = byVoice.get(job.voice_id) ?? []
      list.push(job)
      byVoice.set(job.voice_id, list)
    })

    const series = Array.from(byVoice.entries())
      .sort((a, b) => a[0].localeCompare(b[0]))
      .flatMap(([voiceId, jobs], index) => {
        const rawValues = jobs
          .slice()
          .sort((a, b) => (a.attempt_index ?? a.created_at) - (b.attempt_index ?? b.created_at))
          .map((job) => jobScore(job))
        const name = voicesById.get(voiceId)?.name ?? voiceId.slice(0, 8)
        return [
          {
            name,
            color: SERIES_COLORS[index % SERIES_COLORS.length],
            values: addRunningBest(rawValues),
          },
        ]
      })

    return {
      label: 'Attempt index',
      series,
    }
  }, [filteredJobs, selectedVoice, voicesById])

  const checkpointPassRateData = useMemo(() => {
    if (selectedVoice) {
      const checks = filteredJobs.flatMap((job) => parseEvaluatedCheckpointOk(job.summary))
      if (checks.length === 0) return []
      const passed = checks.filter(Boolean).length
      const rate = passed / checks.length
      return [{ label: 'Pass rate', value: rate, color: rate >= 0.8 ? 'var(--color-accent)' : 'var(--color-warning)' }]
    }

    return voicesInScope
      .map((voice) => {
        const checks = filteredJobs
          .filter((job) => job.voice_id === voice.voice_id)
          .flatMap((job) => parseEvaluatedCheckpointOk(job.summary))
        if (checks.length === 0) return null
        const passed = checks.filter(Boolean).length
        const rate = passed / checks.length
        return {
          label: voice.name,
          value: rate,
          color: rate >= 0.8 ? 'var(--color-accent)' : rate >= 0.6 ? 'var(--color-warning)' : 'var(--color-error)',
        }
      })
      .filter((item): item is { label: string; value: number; color: string } => item !== null)
  }, [filteredJobs, selectedVoice, voicesInScope])

  const durationDistributionData = useMemo(() => {
    const buckets = [
      { label: '<30m', min: 0, max: 30 * 60 * 1000, count: 0 },
      { label: '30-60m', min: 30 * 60 * 1000, max: 60 * 60 * 1000, count: 0 },
      { label: '1-2h', min: 60 * 60 * 1000, max: 2 * 60 * 60 * 1000, count: 0 },
      { label: '2-4h', min: 2 * 60 * 60 * 1000, max: 4 * 60 * 60 * 1000, count: 0 },
      { label: '4h+', min: 4 * 60 * 60 * 1000, max: Number.POSITIVE_INFINITY, count: 0 },
    ]

    filteredJobs.forEach((job) => {
      const duration = jobDuration(job)
      if (duration === null) return
      const bucket = buckets.find((item) => duration >= item.min && duration < item.max)
      if (bucket) bucket.count += 1
    })

    return buckets.map((bucket) => ({ label: bucket.label, value: bucket.count }))
  }, [filteredJobs])

  const scoreVsDurationPoints = useMemo(() => {
    return filteredJobs
      .map((job) => {
        const duration = jobDuration(job)
        const score = jobScore(job)
        if (duration === null || score === null) return null
        return {
          durationMin: duration / 60_000,
          score,
          pass: score >= 0.8,
          id: job.job_id,
        }
      })
      .filter((point): point is { durationMin: number; score: number; pass: boolean; id: string } => point !== null)
  }, [filteredJobs])

  const bestEpochHistogram = useMemo(() => {
    const epochCounts = new Map<number, number>()
    filteredJobs.forEach((job) => {
      const epoch = parseSelectedEpoch(job.summary)
      if (epoch === null) return
      epochCounts.set(epoch, (epochCounts.get(epoch) ?? 0) + 1)
    })
    return Array.from(epochCounts.entries())
      .sort((a, b) => a[0] - b[0])
      .map(([epoch, count]) => ({ label: `E${epoch}`, value: count }))
  }, [filteredJobs])

  const tableRows = useMemo(() => {
    const rows: VoicePerformanceRow[] = voicesInScope.map((voice) => {
      const voiceJobs = filteredAllJobs.filter((job) => job.voice_id === voice.voice_id)
      const voiceCompletedJobs = voiceJobs.filter((job) => job.status === 'completed')
      const durations = voiceCompletedJobs
        .map((job) => jobDuration(job))
        .filter((duration): duration is number => duration !== null)
      const sortedScores = voiceCompletedJobs
        .slice()
        .sort((a, b) => a.created_at - b.created_at)
        .map((job) => jobScore(job))
        .filter((score): score is number => score !== null)
      const firstScore = sortedScores.length > 0 ? sortedScores[0] : null
      const bestScoreForVoice = sortedScores.length > 0 ? Math.max(...sortedScores) : null
      const delta = firstScore !== null && bestScoreForVoice !== null ? bestScoreForVoice - firstScore : null
      const completedTimestamps = voiceCompletedJobs
        .map((job) => job.completed_at)
        .filter((ts): ts is number => typeof ts === 'number' && Number.isFinite(ts))
      const lastTrainedAt = completedTimestamps.length > 0
        ? Math.max(...completedTimestamps)
        : null

      return {
        voiceId: voice.voice_id,
        name: voice.name,
        modelSize: voice.model_size,
        currentScore: typeof voice.checkpoint_score === 'number' ? voice.checkpoint_score : null,
        firstScore,
        bestScore: bestScoreForVoice,
        scoreDelta: delta,
        completedJobs: voiceCompletedJobs.length,
        totalJobs: voiceJobs.length,
        avgDurationMs: durations.length > 0 ? durations.reduce((sum, value) => sum + value, 0) / durations.length : null,
        lastTrainedAt,
        sparkline: sortedScores,
      }
    })

    return rows.sort((a, b) => {
      const direction = sortDirection === 'asc' ? 1 : -1
      let cmp = 0
      if (sortKey === 'name') cmp = a.name.localeCompare(b.name) * direction
      else if (sortKey === 'score') cmp = ((a.currentScore ?? -1) - (b.currentScore ?? -1)) * direction
      else if (sortKey === 'delta') cmp = ((a.scoreDelta ?? -1) - (b.scoreDelta ?? -1)) * direction
      else if (sortKey === 'jobs') cmp = (a.completedJobs - b.completedJobs) * direction
      else if (sortKey === 'duration') cmp = ((a.avgDurationMs ?? Number.MAX_SAFE_INTEGER) - (b.avgDurationMs ?? Number.MAX_SAFE_INTEGER)) * direction
      else cmp = ((a.lastTrainedAt ?? 0) - (b.lastTrainedAt ?? 0)) * direction
      return cmp || a.voiceId.localeCompare(b.voiceId)
    })
  }, [filteredAllJobs, sortDirection, sortKey, voicesInScope])

  const completionRateColor = useMemo(() => {
    if (completionRate >= 80) return 'text-accent'
    if (completionRate >= 60) return 'text-warning'
    return 'text-error'
  }, [completionRate])

  const scatterDomain = useMemo(() => {
    const xMax = Math.max(30, ...scoreVsDurationPoints.map((point) => point.durationMin))
    return { xMin: 0, xMax, yMin: 0, yMax: 1 }
  }, [scoreVsDurationPoints])

  const handleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortDirection((current) => (current === 'asc' ? 'desc' : 'asc'))
      return
    }
    setSortKey(key)
    setSortDirection('desc')
  }

  const sortArrow = (key: SortKey) => sortKey === key ? (sortDirection === 'asc' ? ' \u25B2' : ' \u25BC') : ''

  return (
    <div className="space-y-8">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h1 className="text-heading text-2xl font-bold">Statistics</h1>
          <p className="text-subtle text-sm mt-1">Training performance and quality metrics</p>
        </div>
        <div className="flex items-center gap-2">
          <select
            value={selectedVoice}
            onChange={(event) => setSelectedVoice(event.target.value)}
            className="rounded-lg border border-edge bg-surface px-3 py-2 text-sm text-primary"
          >
            <option value="">All voices</option>
            {voices.map((voice) => (
              <option key={voice.voice_id} value={voice.voice_id}>{voice.name}</option>
            ))}
          </select>
          <button
            type="button"
            className="rounded-lg border border-edge bg-surface px-3 py-2 text-sm text-primary hover:border-accent hover:text-accent disabled:opacity-50"
            disabled={refreshing || loading}
            onClick={() => {
              setRefreshing(true)
              Promise.all([fetchVoices(), fetchAllTrainingJobs(2000)])
                .then(([voicesData, jobsData]) => {
                  setVoices(voicesData.voices)
                  setAllJobs(jobsData.jobs)
                  setError('')
                  if (selectedVoiceRef.current && !voicesData.voices.some((v) => v.voice_id === selectedVoiceRef.current)) {
                    setSelectedVoice('')
                  }
                })
                .catch((err) => setError(err instanceof Error ? err.message : 'Refresh failed'))
                .finally(() => setRefreshing(false))
            }}
          >
            {refreshing ? 'Refreshing...' : 'Refresh'}
          </button>
        </div>
      </div>

      {error && (
        <div className="bg-error-dim border border-error/20 rounded-lg px-4 py-3 text-error text-sm">
          {error}
        </div>
      )}

      <section className="space-y-3">
        <h2 className="text-heading font-semibold text-sm">Overview KPIs</h2>
        <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-6 gap-4">
          <StatCard label="Voices Trained" value={String(voicesTrained)} loading={loading} />
          <StatCard label="Completed / Total" value={`${completedCount} / ${totalCount}`} loading={loading} />
          <StatCard label="Completion Rate" value={`${completionRate.toFixed(1)}%`} valueClass={completionRateColor} loading={loading} />
          <StatCard label="Median Training Time" value={formatDurationMs(medianTrainingTime)} loading={loading} />
          <StatCard label="Best Score" value={formatScore(bestScore)} loading={loading} />
          <StatCard label="Score Improvement" value={scoreImprovement === null ? '—' : `${scoreImprovement >= 0 ? '+' : ''}${scoreImprovement.toFixed(3)}`} loading={loading} />
        </div>
      </section>

      <section className="space-y-4">
        <h2 className="text-heading font-semibold text-sm">Quality Trends</h2>
        <div className="grid lg:grid-cols-2 gap-4">
          <div className="bg-raised border border-edge rounded-xl p-5 space-y-3">
            <h3 className="text-heading font-semibold text-sm">Score Progression</h3>
            {scoreProgressionData.series.length === 0 || scoreProgressionData.series.every((s) => s.values.every((v) => v === null)) ? (
              <EmptyChart message="No scored training jobs yet" />
            ) : (
              <>
                <LineChart
                  data={scoreProgressionData}
                  yMin={0}
                  yMax={1}
                  yLabel="score"
                  xLabel="attempt"
                  height={260}
                />
                {!selectedVoice && scoreProgressionData.series.length > 1 && (
                  <div className="flex flex-wrap gap-x-4 gap-y-1 pt-1">
                    {scoreProgressionData.series.map((s) => (
                      <span key={s.name} className="flex items-center gap-1.5 text-[10px] text-muted">
                        <span className="inline-block w-3 h-0.5 rounded-full" style={{ backgroundColor: s.color }} />
                        {s.name}
                      </span>
                    ))}
                  </div>
                )}
              </>
            )}
          </div>
          <div className="bg-raised border border-edge rounded-xl p-5 space-y-3">
            <h3 className="text-heading font-semibold text-sm">Checkpoint Pass Rate</h3>
            {checkpointPassRateData.length === 0 ? (
              <EmptyChart message="No checkpoint evaluations yet" />
            ) : (
              <BarChart data={checkpointPassRateData} yLabel="pass rate" height={260} />
            )}
          </div>
        </div>
      </section>

      <section className="space-y-4">
        <h2 className="text-heading font-semibold text-sm">Training Efficiency</h2>
        <div className="grid lg:grid-cols-3 gap-4">
          <div className="bg-raised border border-edge rounded-xl p-5 space-y-3">
            <h3 className="text-heading font-semibold text-sm">Duration Distribution</h3>
            {durationDistributionData.every((b) => b.value === 0) ? (
              <EmptyChart message="No completed jobs with duration data" />
            ) : (
              <BarChart data={durationDistributionData} yLabel="jobs" height={240} />
            )}
          </div>

          <div className="bg-raised border border-edge rounded-xl p-5 space-y-3 lg:col-span-2">
            <h3 className="text-heading font-semibold text-sm">Score vs Duration</h3>
            {scoreVsDurationPoints.length === 0 ? (
              <EmptyChart message="No completed jobs with score and duration data" />
            ) : (
            <div className="w-full rounded-lg border border-edge bg-surface p-3">
              <svg viewBox="0 0 760 250" className="w-full h-auto" role="img" aria-label="Score versus duration scatter plot">
                <line x1="42" y1="18" x2="42" y2="216" stroke="var(--color-edge)" />
                <line x1="42" y1="216" x2="744" y2="216" stroke="var(--color-edge)" />
                {Array.from({ length: 5 }).map((_, tick) => {
                  const ratio = tick / 4
                  const y = 18 + ratio * 198
                  const value = (1 - ratio).toFixed(2)
                  return (
                    <g key={`scatter-y-${value}`}>
                      <line x1="42" y1={y} x2="744" y2={y} stroke="var(--color-edge)" strokeWidth="1" />
                      <text x="36" y={y + 4} textAnchor="end" fontSize="10" fill="var(--color-muted)">{value}</text>
                    </g>
                  )
                })}
                {scoreVsDurationPoints.map((point) => {
                  const xRatio = (point.durationMin - scatterDomain.xMin) / Math.max(1, scatterDomain.xMax - scatterDomain.xMin)
                  const yRatio = (point.score - scatterDomain.yMin) / Math.max(0.0001, scatterDomain.yMax - scatterDomain.yMin)
                  const x = 42 + xRatio * 702
                  const y = 216 - yRatio * 198
                  return (
                    <circle
                      key={point.id}
                      cx={x}
                      cy={y}
                      r="3.2"
                      fill={point.pass ? 'var(--color-accent)' : 'var(--color-muted)'}
                      fillOpacity="0.9"
                    />
                  )
                })}
                <text x="42" y="240" fontSize="10" fill="var(--color-muted)">0m</text>
                <text x="744" y="240" textAnchor="end" fontSize="10" fill="var(--color-muted)">{Math.round(scatterDomain.xMax)}m</text>
                <text x="393" y="240" textAnchor="middle" fontSize="10" fill="var(--color-muted)">duration (minutes)</text>
                <text x="12" y="12" fontSize="10" fill="var(--color-muted)">score</text>
              </svg>
            </div>
            )}
          </div>

          <div className="bg-raised border border-edge rounded-xl p-5 space-y-3 lg:col-span-3">
            <h3 className="text-heading font-semibold text-sm">Best Epoch Selected</h3>
            {bestEpochHistogram.length === 0 ? (
              <EmptyChart message="No epoch selection data yet" />
            ) : (
              <BarChart data={bestEpochHistogram} yLabel="champion picks" height={220} />
            )}
          </div>
        </div>
      </section>

      <section className="space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-heading font-semibold text-sm">Recent Performance</h2>
          <span className="text-muted text-[10px] font-mono uppercase tracking-widest">sortable</span>
        </div>

        <div className="bg-raised border border-edge rounded-xl p-5 space-y-3">
          <div className="hidden md:grid grid-cols-[1.4fr_72px_96px_170px_110px_100px_95px_90px] gap-x-3 text-[10px] font-mono uppercase tracking-widest text-muted border-b border-edge pb-2">
            <button type="button" className={`text-left ${sortKey === 'name' ? 'text-primary' : ''}`} onClick={() => handleSort('name')}>Voice{sortArrow('name')}</button>
            <span>Model</span>
            <button type="button" className={`text-right ${sortKey === 'score' ? 'text-primary' : ''}`} onClick={() => handleSort('score')}>Current{sortArrow('score')}</button>
            <button type="button" className={`text-right ${sortKey === 'delta' ? 'text-primary' : ''}`} onClick={() => handleSort('delta')}>First → Best{sortArrow('delta')}</button>
            <button type="button" className={`text-right ${sortKey === 'jobs' ? 'text-primary' : ''}`} onClick={() => handleSort('jobs')}>Jobs{sortArrow('jobs')}</button>
            <button type="button" className={`text-right ${sortKey === 'duration' ? 'text-primary' : ''}`} onClick={() => handleSort('duration')}>Avg Dur{sortArrow('duration')}</button>
            <button type="button" className={`text-right ${sortKey === 'last' ? 'text-primary' : ''}`} onClick={() => handleSort('last')}>Last{sortArrow('last')}</button>
            <span className="text-right">Trend</span>
          </div>

          <div className="space-y-2">
            {tableRows.map((row) => (
              <div key={row.voiceId}>
                <div className="hidden md:grid grid-cols-[1.4fr_72px_96px_170px_110px_100px_95px_90px] gap-x-3 items-center py-2.5 border-b border-edge/40 last:border-b-0">
                  <Link to={`/voices/${row.voiceId}/training`} className="text-accent text-xs hover:text-accent-light truncate">{row.name}</Link>
                  <span className="text-muted text-[11px] font-mono">{row.modelSize}</span>
                  <span className={`text-right text-xs font-mono font-bold ${scoreColor(row.currentScore)}`}>{formatScore(row.currentScore)}</span>
                  <span className="text-right text-xs font-mono">
                    <span className="text-muted">{formatScore(row.firstScore)}</span>
                    <span className="text-muted mx-1">→</span>
                    <span className={scoreColor(row.bestScore)}>{formatScore(row.bestScore)}</span>
                    <span className={`ml-1 ${row.scoreDelta !== null && row.scoreDelta >= 0 ? 'text-accent' : 'text-error'}`}>
                      {row.scoreDelta === null ? '' : `${row.scoreDelta >= 0 ? '+' : ''}${row.scoreDelta.toFixed(3)}`}
                    </span>
                  </span>
                  <span className="text-right text-[11px] font-mono text-primary">{row.completedJobs} / {row.totalJobs}</span>
                  <span className="text-right text-[11px] font-mono text-primary">{formatDurationMs(row.avgDurationMs)}</span>
                  <span className="text-right text-[11px] font-mono text-muted" title={row.lastTrainedAt !== null ? formatDate(row.lastTrainedAt) : ''}>
                    {relativeTime(row.lastTrainedAt)}
                  </span>
                  <div className="flex justify-end">
                    <MiniSparkline values={row.sparkline} />
                  </div>
                </div>

                <div className="md:hidden rounded-lg border border-edge bg-surface p-3 space-y-2">
                  <div className="flex items-center justify-between">
                    <Link to={`/voices/${row.voiceId}/training`} className="text-accent text-xs hover:text-accent-light">{row.name}</Link>
                    <span className="text-muted text-[10px] font-mono">{row.modelSize}</span>
                  </div>
                  <div className="grid grid-cols-2 gap-2 text-[11px] font-mono">
                    <div className="text-muted">Current <span className={scoreColor(row.currentScore)}>{formatScore(row.currentScore)}</span></div>
                    <div className="text-muted text-right">Jobs {row.completedJobs}/{row.totalJobs}</div>
                    <div className="text-muted">Avg {formatDurationMs(row.avgDurationMs)}</div>
                    <div className="text-muted text-right">{relativeTime(row.lastTrainedAt)}</div>
                  </div>
                  <div className="text-[11px] font-mono">
                    <span className="text-muted">{formatScore(row.firstScore)} → </span>
                    <span className={scoreColor(row.bestScore)}>{formatScore(row.bestScore)}</span>
                    {row.scoreDelta !== null && (
                      <span className={`ml-1 ${row.scoreDelta >= 0 ? 'text-accent' : 'text-error'}`}>{row.scoreDelta >= 0 ? '+' : ''}{row.scoreDelta.toFixed(3)}</span>
                    )}
                  </div>
                  <MiniSparkline values={row.sparkline} width={140} height={28} />
                </div>
              </div>
            ))}

            {tableRows.length === 0 && !loading && (
              <div className="rounded-lg border border-edge bg-surface px-4 py-8 text-center text-sm text-muted">
                No voice performance data yet.
              </div>
            )}
          </div>
        </div>
      </section>

    </div>
  )
}

function EmptyChart({ message }: { message: string }) {
  return (
    <div className="flex items-center justify-center rounded-lg border border-dashed border-edge bg-surface px-4 py-12 text-sm text-muted">
      {message}
    </div>
  )
}

function StatCard({
  label,
  value,
  loading,
  valueClass,
}: {
  label: string
  value: string
  loading: boolean
  valueClass?: string
}) {
  return (
    <div className="bg-raised border border-edge rounded-xl p-5">
      <p className="text-muted text-[10px] font-mono uppercase tracking-widest">{label}</p>
      <p className={`text-3xl font-bold font-mono text-heading mt-2 ${valueClass ?? ''} ${loading ? 'animate-pulse' : ''}`}>
        {loading ? '—' : value}
      </p>
    </div>
  )
}
