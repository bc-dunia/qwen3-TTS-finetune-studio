import { useNavigate } from 'react-router'
import type { Voice } from '../lib/api'
import { formatDate } from '../lib/api'

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
      <div className="flex items-center justify-between text-[11px] font-mono text-muted">
        <span className="bg-surface px-2 py-0.5 rounded">
          {voice.model_size || 'base'}
        </span>
        <span>{formatDate(voice.created_at)}</span>
      </div>
    </button>
  )
}
