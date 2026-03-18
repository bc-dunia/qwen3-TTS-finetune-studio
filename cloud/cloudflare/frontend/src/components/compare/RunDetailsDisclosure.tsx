import { useState } from 'react'
import { formatDate, formatTime, formatDateTime, formatDurationMs } from '../../lib/api'
import { formatScore, scoreColor } from '../../lib/voiceScoreUi'

export function RunDetailsDisclosure({
  runName,
  attemptNumber,
  epoch,
  score,
  preset,
  jobId,
  createdAt,
  completedAt,
  message,
  styleScore,
  toneScore,
  speedScore,
}: {
  runName?: string | null
  attemptNumber?: number | null
  epoch?: number | null
  score?: number | null
  preset?: string | null
  jobId?: string | null
  createdAt: number
  completedAt?: number | null
  message?: string | null
  styleScore?: number | null
  toneScore?: number | null
  speedScore?: number | null
}) {
  const [open, setOpen] = useState(false)

  const durationMs =
    completedAt != null ? Math.max(0, completedAt - createdAt) : null

  return (
    <div className="border-t border-edge mt-3 pt-2">
      <button
        onClick={() => setOpen((prev) => !prev)}
        className="flex items-center gap-1.5 text-[10px] font-semibold text-muted hover:text-primary transition-colors"
        type="button"
      >
        <svg
          className={`w-3 h-3 transition-transform duration-200 ${open ? 'rotate-90' : ''}`}
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2.5"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <polyline points="9 18 15 12 9 6" />
        </svg>
        Details
      </button>

      {open && (
        <div className="mt-2 rounded-lg bg-surface border border-edge px-3 py-2 text-[10px] font-mono text-muted space-y-1 animate-slide-up">
          {runName != null && <div>run_name={runName}</div>}
          {attemptNumber != null && <div>attempt={attemptNumber}</div>}
          {epoch != null && <div>epoch={epoch}</div>}
          {score != null && (
            <div>
              score={formatScore(score)}{' '}
              <span className={scoreColor(score)}>
                ({score >= 0.90 ? 'excellent' : score >= 0.85 ? 'strong' : score >= 0.70 ? 'borderline' : 'weak'})
              </span>
            </div>
          )}
          {preset != null && <div>preset={preset}</div>}
          {jobId != null && <div>job_id={jobId.slice(0, 16)}</div>}
          <div>created={formatDateTime(createdAt)}</div>
          {completedAt != null && (
            <div>completed={formatDate(completedAt)} {formatTime(completedAt)}</div>
          )}
          {durationMs != null && <div>duration={formatDurationMs(durationMs)}</div>}
          {styleScore != null && (
            <div>style_score=<span className={scoreColor(styleScore)}>{formatScore(styleScore)}</span></div>
          )}
          {toneScore != null && (
            <div>tone_score=<span className={scoreColor(toneScore)}>{formatScore(toneScore)}</span></div>
          )}
          {speedScore != null && (
            <div>speed_score=<span className={scoreColor(speedScore)}>{formatScore(speedScore)}</span></div>
          )}
          {message && <div className="text-subtle pt-1 font-sans text-[11px]">{message}</div>}
        </div>
      )}
    </div>
  )
}
