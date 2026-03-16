import { useState, useEffect, useMemo } from 'react'
import { Link } from 'react-router'
import { fetchVoices, fetchAllTrainingJobs, type Voice, type TrainingJob, formatDate, formatTime } from '../lib/api'
import { shouldWatchTrainingJob } from '../lib/trainingCheckout'
import { formatScore, scoreColor } from '../lib/voiceScoreUi'

interface GenerationRecord {
  id: string
  text: string
  voiceName: string
  createdAt: string
}

export function Dashboard() {
  const [voices, setVoices] = useState<Voice[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  // Load from localStorage
  const [recentGenerations] = useState<GenerationRecord[]>(() => {
    try {
      const stored = localStorage.getItem('tts-generation-history')
      if (stored) {
        const parsed = JSON.parse(stored) as GenerationRecord[]
        return parsed.slice(0, 5)
      }
    } catch {
      // ignore
    }
    return []
  })

  const [allJobs, setAllJobs] = useState<TrainingJob[]>([])

  useEffect(() => {
    let cancelled = false

    async function load() {
      try {
        const [voicesData, jobsData] = await Promise.all([
          fetchVoices(),
          fetchAllTrainingJobs(200),
        ])
        if (!cancelled) {
          setVoices(voicesData.voices)
          setAllJobs(jobsData.jobs)
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : 'Failed to load dashboard data')
        }
      } finally {
        if (!cancelled) setLoading(false)
      }
    }

    load()
    const interval = setInterval(() => { void load() }, 10_000)
    return () => { cancelled = true; clearInterval(interval) }
  }, [])

  const activeJobs = useMemo(() => allJobs.filter((job) => shouldWatchTrainingJob(job)), [allJobs])
  const queuedJobCount = useMemo(() => allJobs.filter((job) => job.status === 'queued' || job.status === 'pending').length, [allJobs])

  const readyCount = voices.filter((v) => v.status === 'ready').length
  const trainingCount = voices.filter((v) => v.status === 'training').length
  const createdCount = voices.filter((v) => v.status === 'created').length
  const voiceNames = useMemo(() => new Map(voices.map((v) => [v.voice_id, v.name])), [voices])

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-heading text-2xl font-bold">Dashboard</h1>
          <p className="text-subtle text-sm mt-1">Voice cloning overview</p>
        </div>
        <Link
          to="/playground"
          className="inline-flex items-center gap-2 bg-accent hover:bg-accent-light text-void font-semibold text-sm px-5 py-2.5 rounded-lg transition-colors"
        >
          <svg className="w-4 h-4" viewBox="0 0 24 24" fill="currentColor">
            <title>Generate speech</title>
            <path d="M8 5.14v13.72a1 1 0 0 0 1.5.86l11.24-7.36a1 1 0 0 0 0-1.72L9.5 4.28A1 1 0 0 0 8 5.14z" />
          </svg>
          Generate Speech
        </Link>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 lg:grid-cols-5 gap-4">
        <StatCard
          label="Total Voices"
          value={loading ? '—' : String(voices.length)}
          loading={loading}
        />
        <StatCard
          label="Ready"
          value={loading ? '—' : String(readyCount)}
          accent
          loading={loading}
        />
        <StatCard
          label="Training"
          value={loading ? '—' : String(trainingCount)}
          warning={trainingCount > 0}
          loading={loading}
        />
        <StatCard
          label="Active Jobs"
          value={loading ? '—' : String(activeJobs.length)}
          warning={activeJobs.length > 0}
          loading={loading}
        />
        <StatCard
          label="Queued"
          value={loading ? '—' : String(queuedJobCount)}
          loading={loading}
        />
      </div>

      {/* Error */}
      {error && (
        <div className="bg-error-dim border border-error/20 rounded-lg px-4 py-3 text-error text-sm">
          {error}
        </div>
      )}

      {/* Voice Overview */}
      {!loading && voices.length > 0 && (
        <VoiceOverviewTable voices={voices} />
      )}

      <div className="grid lg:grid-cols-2 gap-6">
        {/* Recent Generations */}
        <div className="bg-raised border border-edge rounded-xl p-5">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-heading font-semibold text-sm">Recent Generations</h2>
            <Link to="/playground" className="text-accent text-xs hover:text-accent-light">
              View all →
            </Link>
          </div>

          {recentGenerations.length === 0 ? (
            <div className="text-center py-8">
              <div className="text-muted text-sm">No generations yet</div>
              <Link
                to="/playground"
                className="text-accent text-xs mt-2 inline-block hover:text-accent-light"
              >
                Go to Playground →
              </Link>
            </div>
          ) : (
            <div className="space-y-2">
              {recentGenerations.map((gen) => (
                <div
                  key={gen.id}
                  className="flex items-center gap-3 px-3 py-2.5 rounded-lg bg-surface"
                >
                  <div className="w-7 h-7 rounded-full bg-accent-dim flex items-center justify-center shrink-0">
                    <svg className="w-3 h-3 text-accent" viewBox="0 0 24 24" fill="currentColor">
                      <title>Recent generation</title>
                      <path d="M8 5.14v13.72a1 1 0 0 0 1.5.86l11.24-7.36a1 1 0 0 0 0-1.72L9.5 4.28A1 1 0 0 0 8 5.14z" />
                    </svg>
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-primary text-xs truncate">{gen.text}</p>
                    <p className="text-muted text-[10px] font-mono mt-0.5">
                      {gen.voiceName} · {formatTime(gen.createdAt)}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Active Training */}
        <div className="bg-raised border border-edge rounded-xl p-5">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-heading font-semibold text-sm">Active Training</h2>
            <Link to="/queue" className="text-accent text-xs hover:text-accent-light">
              Manage →
            </Link>
          </div>

          {activeJobs.length === 0 ? (
            <div className="text-center py-8">
              <div className="text-muted text-sm">No active training jobs</div>
              <Link
                to="/queue"
                className="text-accent text-xs mt-2 inline-block hover:text-accent-light"
              >
                View Queue →
              </Link>
            </div>
          ) : (
            <div className="space-y-3">
              {activeJobs.map((job) => (
                <div key={job.job_id} className="px-3 py-3 rounded-lg bg-surface">
                  {(() => {
                    const epoch = typeof job.progress.epoch === 'number' ? job.progress.epoch : 0
                    const totalEpochs = typeof job.progress.total_epochs === 'number' ? job.progress.total_epochs : 0
                    const loss = typeof job.progress.loss === 'number' ? job.progress.loss : null

                    return (
                      <>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-primary text-xs font-medium">
                      {voiceNames.get(job.voice_id) ?? job.voice_id.slice(0, 8)} · {job.job_id.slice(0, 8)}
                    </span>
                    <span className="text-warning text-[10px] font-mono uppercase">
                      {job.status}
                    </span>
                  </div>
                  <div className="w-full h-1.5 bg-edge rounded-full overflow-hidden">
                    <div
                      className="h-full bg-accent rounded-full transition-[width] duration-500"
                      style={{
                        width: totalEpochs > 0
                          ? `${(epoch / totalEpochs) * 100}%`
                          : '0%',
                      }}
                    />
                  </div>
                  <div className="flex justify-between mt-1.5 text-muted text-[10px] font-mono">
                    <span>Epoch {epoch}/{totalEpochs || '—'}</span>
                    <span>Loss: {loss !== null ? loss.toFixed(4) : '—'}</span>
                  </div>
                      </>
                    )
                  })()}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Quick Links */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <QuickLink to="/voices" label="Manage Voices" icon="mic" />
        <QuickLink to="/playground" label="TTS Playground" icon="play" />
        <QuickLink to="/queue" label="Training Queue" icon="training" />
        <QuickLink to="/voices" label="Create Voice" icon="plus" />
      </div>
    </div>
  )
}

// ── Sub-components ─────────────────────────────────────────────────────────────

function sortVoicesForOverview(voices: Voice[]): Voice[] {
  const statusOrder: Record<string, number> = { training: 0, ready: 1, created: 2 }
  return [...voices].sort((a, b) => {
    const oa = statusOrder[a.status] ?? 3
    const ob = statusOrder[b.status] ?? 3
    if (oa !== ob) return oa - ob
    const sa = a.checkpoint_score ?? -1
    const sb = b.checkpoint_score ?? -1
    return sb - sa
  })
}

function VoiceOverviewTable({ voices }: { voices: Voice[] }) {
  const sorted = sortVoicesForOverview(voices)
  const statusDot: Record<string, string> = {
    ready: 'bg-accent',
    training: 'bg-warning animate-pulse',
    created: 'bg-muted',
  }

  return (
    <div className="bg-raised border border-edge rounded-xl p-5">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-heading font-semibold text-sm">Voice Overview</h2>
        <Link to="/voices" className="text-accent text-xs hover:text-accent-light">
          Manage →
        </Link>
      </div>

      {/* Desktop table */}
      <div className="hidden sm:block">
        <div className="grid grid-cols-[1fr_60px_70px_56px_80px_90px] gap-x-3 px-3 pb-2 text-[10px] font-mono text-muted uppercase tracking-widest border-b border-edge">
          <span>Name</span>
          <span>Model</span>
          <span className="text-right">Score</span>
          <span className="text-right">Epoch</span>
          <span>Status</span>
          <span className="text-right">Action</span>
        </div>
        <div className="divide-y divide-edge/50">
          {sorted.map((v) => {
            const isImproving =
              v.candidate_score != null &&
              v.checkpoint_score != null &&
              Number.isFinite(v.candidate_score) &&
              Number.isFinite(v.checkpoint_score) &&
              v.candidate_score > v.checkpoint_score

            const href = v.status === 'ready'
              ? `/voices/${v.voice_id}/generate`
              : `/voices/${v.voice_id}/training`

            return (
              <Link
                key={v.voice_id}
                to={href}
                className="grid grid-cols-[1fr_60px_70px_56px_80px_90px] gap-x-3 px-3 py-2.5 items-center hover:bg-surface transition-colors rounded"
              >
                <span className="text-primary text-xs font-medium truncate">{v.name}</span>
                <span className="text-muted text-[11px] font-mono">{v.model_size || '—'}</span>
                <span className={`text-right text-xs font-mono font-bold ${scoreColor(v.checkpoint_score)}`}>
                  {formatScore(v.checkpoint_score)}
                  {isImproving && (
                    <span className="text-accent text-[9px] ml-0.5">+</span>
                  )}
                </span>
                <span className="text-right text-muted text-[11px] font-mono">
                  {typeof v.epoch === 'number' ? v.epoch : '—'}
                </span>
                <span className="flex items-center gap-1.5 text-[11px] font-mono text-muted">
                  <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${statusDot[v.status] ?? 'bg-muted'}`} />
                  {v.status}
                </span>
                <span className="text-right text-accent text-[11px] font-mono hover:text-accent-light">
                  {v.status === 'ready' ? 'Generate →' : v.status === 'training' ? 'View →' : 'Train →'}
                </span>
              </Link>
            )
          })}
        </div>
      </div>

      {/* Mobile cards */}
      <div className="sm:hidden space-y-2">
        {sorted.map((v) => {
          const isImproving =
            v.candidate_score != null &&
            v.checkpoint_score != null &&
            Number.isFinite(v.candidate_score) &&
            Number.isFinite(v.checkpoint_score) &&
            v.candidate_score > v.checkpoint_score

          const mobileHref = v.status === 'ready'
            ? `/voices/${v.voice_id}/generate`
            : `/voices/${v.voice_id}/training`

          return (
            <Link
              key={v.voice_id}
              to={mobileHref}
              className="block px-3 py-3 rounded-lg bg-surface hover:bg-elevated transition-colors"
            >
              <div className="flex items-center justify-between mb-2">
                <span className="text-primary text-xs font-medium truncate">{v.name}</span>
                <span className="flex items-center gap-1.5 text-[10px] font-mono text-muted">
                  <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${statusDot[v.status] ?? 'bg-muted'}`} />
                  {v.status}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3 text-[11px] font-mono text-muted">
                  <span>{v.model_size || '—'}</span>
                  {typeof v.epoch === 'number' && <span>ep {v.epoch}</span>}
                </div>
                <div className="flex items-center gap-2">
                  <span className={`text-sm font-bold font-mono ${scoreColor(v.checkpoint_score)}`}>
                    {formatScore(v.checkpoint_score)}
                    {isImproving && <span className="text-accent text-[9px] ml-0.5">+</span>}
                  </span>
                  <span className="text-accent text-[10px] font-mono">
                    {v.status === 'ready' ? 'Generate →' : v.status === 'training' ? 'View →' : 'Train →'}
                  </span>
                </div>
              </div>
            </Link>
          )
        })}
      </div>
    </div>
  )
}

function StatCard({
  label,
  value,
  accent,
  warning,
  loading,
}: {
  label: string
  value: string
  accent?: boolean
  warning?: boolean
  loading?: boolean
}) {
  let valueColor = 'text-heading'
  if (accent) valueColor = 'text-accent'
  if (warning) valueColor = 'text-warning'

  return (
    <div className="bg-raised border border-edge rounded-xl p-4">
      <p className="text-muted text-[10px] font-mono uppercase tracking-widest mb-2">{label}</p>
      <p className={`text-3xl font-bold font-mono ${valueColor} ${loading ? 'animate-pulse' : ''}`}>
        {value}
      </p>
    </div>
  )
}

function QuickLink({ to, label, icon }: { to: string; label: string; icon: string }) {
  return (
    <Link
      to={to}
      className="flex items-center gap-2.5 bg-raised border border-edge rounded-lg px-4 py-3 text-subtle text-xs font-medium hover:text-accent hover:border-accent/30 transition-colors"
    >
      <QuickIcon name={icon} />
      {label}
    </Link>
  )
}

function QuickIcon({ name }: { name: string }) {
  const cls = "w-4 h-4 shrink-0"
  switch (name) {
    case 'mic':
      return (
        <svg className={cls} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
          <title>Microphone</title>
          <rect x="9" y="2" width="6" height="11" rx="3" fill="currentColor" stroke="none" />
          <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
        </svg>
      )
    case 'play':
      return (
        <svg className={cls} viewBox="0 0 24 24" fill="currentColor">
          <title>Play</title>
          <path d="M8 5.14v13.72a1 1 0 0 0 1.5.86l11.24-7.36a1 1 0 0 0 0-1.72L9.5 4.28A1 1 0 0 0 8 5.14z" />
        </svg>
      )
    case 'training':
      return (
        <svg className={cls} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <title>Training</title>
          <polyline points="22,12 18,12 15,21 9,3 6,12 2,12" />
        </svg>
      )
    case 'plus':
      return (
        <svg className={cls} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
          <title>Add</title>
          <line x1="12" y1="5" x2="12" y2="19" />
          <line x1="5" y1="12" x2="19" y2="12" />
        </svg>
      )
    default:
      return <div className={cls} />
  }
}
