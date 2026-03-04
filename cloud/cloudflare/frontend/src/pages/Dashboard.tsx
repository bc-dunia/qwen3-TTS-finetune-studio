import { useState, useEffect } from 'react'
import { Link } from 'react-router'
import { fetchVoices, type Voice, type TrainingJob, formatDate, formatTime } from '../lib/api'

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

  const [activeJobs] = useState<TrainingJob[]>([])

  useEffect(() => {
    let cancelled = false

    async function load() {
      try {
        const data = await fetchVoices()
        if (!cancelled) setVoices(data.voices)
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : 'Failed to load voices')
        }
      } finally {
        if (!cancelled) setLoading(false)
      }
    }

    load()
    return () => { cancelled = true }
  }, [])

  const readyCount = voices.filter((v) => v.status === 'ready').length
  const trainingCount = voices.filter((v) => v.status === 'training').length
  const createdCount = voices.filter((v) => v.status === 'created').length

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
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
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
          label="Created"
          value={loading ? '—' : String(createdCount)}
          loading={loading}
        />
      </div>

      {/* Error */}
      {error && (
        <div className="bg-error-dim border border-error/20 rounded-lg px-4 py-3 text-error text-sm">
          {error}
        </div>
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
            <Link to="/training" className="text-accent text-xs hover:text-accent-light">
              Manage →
            </Link>
          </div>

          {activeJobs.length === 0 ? (
            <div className="text-center py-8">
              <div className="text-muted text-sm">No active training jobs</div>
              <Link
                to="/training"
                className="text-accent text-xs mt-2 inline-block hover:text-accent-light"
              >
                Start Training →
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
                      Job {job.job_id.slice(0, 8)}
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
        <QuickLink to="/training" label="Training Jobs" icon="training" />
        <QuickLink to="/voices" label="Create Voice" icon="plus" />
      </div>
    </div>
  )
}

// ── Sub-components ─────────────────────────────────────────────────────────────

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
