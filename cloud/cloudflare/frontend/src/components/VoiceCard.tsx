import { useNavigate } from 'react-router'
import type { Voice } from '../lib/api'
import { DEFAULT_VOICE_SETTINGS, formatDate } from '../lib/api'

interface VoiceCardProps {
  voice: Voice
}

const STATUS_STYLES: Record<string, { bg: string; text: string; dot: string }> = {
  ready: {
    bg: 'bg-accent-dim',
    text: 'text-accent',
    dot: 'bg-accent shadow-[0_0_4px_rgba(16,185,129,0.6)]',
  },
  training: {
    bg: 'bg-warning-dim',
    text: 'text-warning',
    dot: 'bg-warning shadow-[0_0_4px_rgba(245,158,11,0.6)]',
  },
  created: {
    bg: 'bg-raised',
    text: 'text-muted',
    dot: 'bg-muted',
  },
}

export function VoiceCard({ voice }: VoiceCardProps) {
  const navigate = useNavigate()
  const status = STATUS_STYLES[voice.status] ?? STATUS_STYLES.created
  const lastUpdated = voice.updated_at ?? voice.created_at
  const src = voice.settings ?? {}
  const settings = {
    stability: typeof src.stability === 'number' ? src.stability : DEFAULT_VOICE_SETTINGS.stability,
    similarity_boost:
      typeof src.similarity_boost === 'number' ? src.similarity_boost : DEFAULT_VOICE_SETTINGS.similarity_boost,
    style: typeof src.style === 'number' ? src.style : DEFAULT_VOICE_SETTINGS.style,
    speed: typeof src.speed === 'number' ? src.speed : DEFAULT_VOICE_SETTINGS.speed,
  }
  const settingSummary = `stab ${settings.stability.toFixed(2)} · sim ${settings.similarity_boost.toFixed(2)} · style ${settings.style.toFixed(2)} · speed ${settings.speed.toFixed(2)}`

  return (
    <button
      type="button"
      onClick={() => navigate(`/voices/${voice.voice_id}`)}
      className="bg-raised border border-edge rounded-xl p-5 text-left hover:border-accent/30 hover:bg-elevated transition-all duration-150 group w-full"
    >
      {/* Header */}
      <div className="flex items-start justify-between gap-3 mb-3">
        <h3 className="text-heading font-semibold text-sm truncate group-hover:text-accent transition-colors">
          {voice.name}
        </h3>
        <span className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[10px] font-mono uppercase tracking-wider shrink-0 ${status.bg} ${status.text}`}>
          <span className={`w-1.5 h-1.5 rounded-full ${status.dot}`} />
          {voice.status}
        </span>
      </div>

      {/* Description */}
      {voice.description && (
        <p className="text-subtle text-xs leading-relaxed line-clamp-2 mb-4">
          {voice.description}
        </p>
      )}

      {/* Footer */}
      <div className="space-y-2 text-[11px] font-mono text-muted">
        <div className="flex items-center justify-between gap-2">
          <div className="flex items-center gap-2 min-w-0">
            <span className="bg-surface px-2 py-0.5 rounded shrink-0">
              {voice.model_size || 'base'}
            </span>
            {typeof voice.epoch === 'number' && (
              <span className="bg-surface px-2 py-0.5 rounded shrink-0">
                epoch {voice.epoch}
              </span>
            )}
          </div>
          <span className="shrink-0">Updated {formatDate(lastUpdated)}</span>
        </div>
        <div className="text-[10px] leading-relaxed text-subtle">
          {settingSummary}
        </div>
      </div>
    </button>
  )
}
