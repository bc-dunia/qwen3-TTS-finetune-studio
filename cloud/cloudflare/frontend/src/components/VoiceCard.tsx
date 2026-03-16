import { useNavigate } from 'react-router'
import type { Voice } from '../lib/api'
import { DEFAULT_VOICE_SETTINGS, formatDate } from '../lib/api'
import { formatScore, scoreColor, scoreBgColor } from '../lib/voiceScoreUi'

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

  const hasCheckpoint = voice.checkpoint_r2_prefix != null || voice.run_name != null
  const isImproving =
    voice.candidate_score != null &&
    voice.checkpoint_score != null &&
    Number.isFinite(voice.candidate_score) &&
    Number.isFinite(voice.checkpoint_score) &&
    voice.candidate_score > voice.checkpoint_score

  return (
    <div className="bg-raised border border-edge rounded-xl text-left hover:border-accent/30 hover:bg-elevated transition-all duration-150 group w-full flex flex-col">
      {/* Clickable card body */}
      <button
        type="button"
        onClick={() => navigate(
          voice.status === 'created' || voice.status === 'training'
            ? `/voices/${voice.voice_id}/training`
            : `/voices/${voice.voice_id}`
        )}
        className="p-5 pb-3 text-left w-full flex-1"
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

        {/* Score + Metadata row */}
        <div className="flex items-start gap-4 mb-3">
          {/* Score badge */}
          <div className={`rounded-lg px-3 py-2.5 text-center shrink-0 ${scoreBgColor(voice.checkpoint_score)}`}>
            {hasCheckpoint && voice.checkpoint_score != null && Number.isFinite(voice.checkpoint_score) ? (
              <>
                <div className={`text-lg font-bold font-mono leading-none ${scoreColor(voice.checkpoint_score)}`}>
                  {formatScore(voice.checkpoint_score)}
                </div>
                <div className="text-[9px] font-mono text-muted uppercase tracking-wider mt-1">score</div>
              </>
            ) : hasCheckpoint ? (
              <>
                <div className="text-xs font-mono text-muted leading-none">...</div>
                <div className="text-[9px] font-mono text-muted uppercase tracking-wider mt-1">scoring</div>
              </>
            ) : (
              <>
                <div className="text-xs font-mono text-muted leading-none">—</div>
                <div className="text-[9px] font-mono text-muted uppercase tracking-wider mt-1">no ckpt</div>
              </>
            )}
          </div>

          {/* Metadata column */}
          <div className="space-y-1.5 text-[11px] font-mono text-muted min-w-0 flex-1">
            <div className="flex items-center gap-2">
              <span className="bg-surface px-2 py-0.5 rounded shrink-0">
                {voice.model_size || 'base'}
              </span>
              {typeof voice.epoch === 'number' && (
                <span className="bg-surface px-2 py-0.5 rounded shrink-0">
                  epoch {voice.epoch}
                </span>
              )}
              {isImproving && (
                <span className="text-accent text-[10px] flex items-center gap-0.5 shrink-0">
                  <svg className="w-3 h-3" viewBox="0 0 24 24" fill="currentColor">
                    <title>Score improving</title>
                    <path d="M12 4l-1.41 1.41L16.17 11H4v2h12.17l-5.58 5.59L12 20l8-8-8-8z" />
                  </svg>
                  improving
                </span>
              )}
            </div>
            <div className="truncate">Updated {formatDate(lastUpdated)}</div>
            <div className="text-[10px] text-subtle truncate">{settingSummary}</div>
          </div>
        </div>
      </button>

      {/* Training CTA — separate click target */}
      <div className="px-5 pb-4 pt-1">
        {voice.status === 'created' ? (
          <button
            type="button"
            onClick={() => navigate(`/voices/${voice.voice_id}/training`)}
            className="w-full flex items-center justify-center gap-2 bg-accent hover:bg-accent-light text-void font-semibold text-xs py-2.5 rounded-lg transition-colors"
          >
            Start Training
            <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="currentColor">
              <title>Start training</title>
              <path d="M12 4l-1.41 1.41L16.17 11H4v2h12.17l-5.58 5.59L12 20l8-8-8-8z" />
            </svg>
          </button>
        ) : voice.status === 'training' ? (
          <button
            type="button"
            onClick={() => navigate(`/voices/${voice.voice_id}/training`)}
            className="w-full flex items-center justify-center gap-2 bg-warning-dim hover:bg-warning/20 text-warning font-mono text-[11px] py-2.5 rounded-lg transition-colors cursor-pointer"
          >
            <span className="w-1.5 h-1.5 rounded-full bg-warning animate-pulse" />
            View Training Progress
          </button>
        ) : (
          <button
            type="button"
            onClick={() => navigate(`/voices/${voice.voice_id}/training`)}
            className="w-full flex items-center justify-center gap-2 bg-surface hover:bg-elevated text-muted hover:text-accent font-mono text-[11px] py-2 rounded-lg transition-colors"
          >
            Continue Training
            <svg className="w-3 h-3" viewBox="0 0 24 24" fill="currentColor">
              <title>Continue training</title>
              <path d="M12 4l-1.41 1.41L16.17 11H4v2h12.17l-5.58 5.59L12 20l8-8-8-8z" />
            </svg>
          </button>
        )}
      </div>
    </div>
  )
}
