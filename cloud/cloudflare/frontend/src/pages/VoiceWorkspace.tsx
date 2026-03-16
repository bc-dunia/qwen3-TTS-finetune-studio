import { useEffect, useState } from 'react'
import { Link, NavLink, Navigate, Outlet, useOutletContext, useParams } from 'react-router'
import { fetchVoice, type Voice } from '../lib/api'
import { formatScore, scoreColor } from '../lib/voiceScoreUi'

const TABS = [
  { to: 'generate', label: 'Generate' },
  { to: 'training', label: 'Training' },
  { to: 'dataset', label: 'Dataset' },
  { to: 'compare', label: 'Compare' },
]

function getStatusClass(status: Voice['status']) {
  if (status === 'ready') return 'bg-accent-dim text-accent'
  if (status === 'training') return 'bg-warning-dim text-warning'
  return 'bg-raised text-muted'
}

export function VoiceWorkspace() {
  const { voiceId = '' } = useParams()
  const [voice, setVoice] = useState<Voice | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    if (!voiceId) return
    let cancelled = false

    async function load() {
      setLoading(true)
      try {
        const data = await fetchVoice(voiceId)
        if (!cancelled) {
          setVoice(data)
        }
      } catch {
        if (!cancelled) {
          setVoice(null)
        }
      } finally {
        if (!cancelled) {
          setLoading(false)
        }
      }
    }

    void load()
    return () => {
      cancelled = true
    }
  }, [voiceId])

  const statusText = voice?.status ?? 'unknown'

  return (
    <div className="space-y-5">
      <div className="rounded-xl border border-edge bg-raised px-5 py-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <Link to="/voices" className="text-muted text-xs font-mono uppercase tracking-wider hover:text-accent transition-colors">Voices</Link>
            <h1 className="mt-1 text-heading text-xl font-bold">
              {loading ? 'Loading voice...' : (voice?.name ?? 'Voice not found')}
            </h1>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            {voice?.model_size && (
              <span className="rounded-full bg-surface px-2 py-0.5 text-[10px] font-mono text-muted">
                {voice.model_size}
              </span>
            )}
            {voice && typeof voice.checkpoint_score === 'number' && (
              <span className={`text-sm font-bold font-mono ${scoreColor(voice.checkpoint_score)}`}>
                {formatScore(voice.checkpoint_score)}
                {voice.candidate_score != null &&
                  Number.isFinite(voice.candidate_score) &&
                  voice.checkpoint_score != null &&
                  Number.isFinite(voice.checkpoint_score) &&
                  voice.candidate_score > voice.checkpoint_score && (
                  <span className="text-accent text-[9px] ml-0.5">+</span>
                )}
              </span>
            )}
            <span
              className={`inline-flex items-center rounded-full px-2.5 py-1 text-[10px] font-mono uppercase tracking-wider ${
                getStatusClass(voice?.status ?? 'created')
              }`}
            >
              {statusText}
            </span>
          </div>
        </div>
        <div className="mt-4 flex flex-wrap gap-2 border-t border-edge pt-4">
          {TABS.map((tab) => (
            <NavLink
              key={tab.to}
              to={tab.to}
              className={({ isActive }) =>
                `rounded-lg border px-3 py-1.5 text-xs font-semibold transition-colors ${
                  isActive
                    ? 'border-accent bg-accent-dim text-accent'
                    : 'border-edge text-subtle hover:border-accent hover:text-accent'
                }`
              }
            >
              {tab.label}
            </NavLink>
          ))}
        </div>
      </div>

      <Outlet context={{ voice, loading }} />
    </div>
  )
}

export function VoiceDefaultRedirect() {
  const { voice, loading } = useOutletContext<{ voice: Voice | null; loading: boolean }>()

  if (loading) {
    return (
      <div className="rounded-xl border border-edge bg-raised p-6 text-sm text-muted">
        Loading...
      </div>
    )
  }

  const hasCheckpoint =
    Boolean(voice?.run_name) || Boolean(voice?.checkpoint_r2_prefix)

  return <Navigate to={hasCheckpoint ? 'generate' : 'training'} replace />
}
