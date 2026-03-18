import type { CheckpointCandidate, CompareResult } from './types'
import { getCandidateTitle } from './utils'
import { ScoreBand, TraitBars } from './ScoreBand'
import { RunDetailsDisclosure } from './RunDetailsDisclosure'
import { AudioPlayer } from '../AudioPlayer'

export function CandidateCard({
  candidate,
  selected,
  onToggle,
  mode,
  result,
  generating,
  applyingId,
  onApply,
}: {
  candidate: CheckpointCandidate
  selected: boolean
  onToggle: () => void
  mode: 'browse' | 'listening'
  result?: CompareResult
  generating?: boolean
  applyingId?: string
  onApply?: () => void
}) {
  const title = getCandidateTitle(candidate)
  const subtitle = `${candidate.runName ?? candidate.jobId?.slice(0, 8) ?? 'checkpoint'} \u00B7 run ${candidate.attemptNumber ?? 'n/a'} \u00B7 epoch ${candidate.epoch ?? 'n/a'}`

  return (
    <div
      role="button"
      tabIndex={0}
      onClick={(e) => {
        // Don't toggle when clicking interactive elements inside the card
        const target = e.target as HTMLElement
        if (target.closest('button, a, input, audio, canvas')) return
        onToggle()
      }}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          const target = e.target as HTMLElement
          if (target.closest('button, a, input, audio, canvas')) return
          e.preventDefault()
          onToggle()
        }
      }}
      className={`rounded-xl border p-4 transition-colors cursor-pointer ${
        selected
          ? 'border-accent bg-accent-dim/20'
          : 'border-edge bg-surface hover:border-accent/40'
      }`}
    >
      {/* Header */}
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-primary text-sm font-semibold">{title}</span>
            <ScoreBand score={candidate.score} />
          </div>
          <div className="text-subtle text-[11px] mt-1">{subtitle}</div>
        </div>
        <input
          type="checkbox"
          checked={selected}
          onChange={onToggle}
          aria-label={`Select ${title}`}
          className="accent-accent mt-1 shrink-0"
        />
      </div>

      {/* Role badges */}
      <div className="mt-2.5 flex flex-wrap gap-1.5 text-[10px] font-mono">
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

      {mode === 'browse' && candidate.message && (
        <p className="mt-2 text-[11px] leading-relaxed text-subtle line-clamp-2">{candidate.message}</p>
      )}

      {/* LISTENING MODE: AudioPlayer + Traits + Apply */}
      {mode === 'listening' && (
        <div className="mt-3 space-y-3">
          {/* Status */}
          <div className="text-[10px] font-mono text-muted">{result?.status ?? 'Idle'}</div>

          <AudioPlayer
            blob={result?.blob}
            generating={generating && !result?.blob && result?.status !== 'Failed'}
          />

          {(candidate.styleScore != null || candidate.toneScore != null || candidate.speedScore != null) && (
            <TraitBars
              style={candidate.styleScore}
              tone={candidate.toneScore}
              speed={candidate.speedScore}
            />
          )}

          {result?.error && (
            <div className="rounded-lg border border-error/20 bg-error-dim px-3 py-2 text-error text-xs">
              {result.error}
            </div>
          )}

          {candidate.jobId && !candidate.isCurrentProduction && onApply && (
            <button
              onClick={onApply}
              disabled={applyingId === candidate.id}
              className="inline-flex items-center rounded-lg border border-edge px-3 py-2 text-[11px] font-semibold text-primary transition-colors hover:border-accent hover:text-accent disabled:opacity-50"
              type="button"
            >
              {applyingId === candidate.id ? 'Applying\u2026' : 'Apply This Checkpoint'}
            </button>
          )}
        </div>
      )}

      {/* Details disclosure */}
      <RunDetailsDisclosure
        runName={candidate.runName}
        attemptNumber={candidate.attemptNumber}
        epoch={candidate.epoch}
        score={candidate.score}
        preset={candidate.preset}
        jobId={candidate.jobId}
        createdAt={candidate.createdAt}
        completedAt={candidate.completedAt}
        message={candidate.message}
        styleScore={candidate.styleScore}
        toneScore={candidate.toneScore}
        speedScore={candidate.speedScore}
      />
    </div>
  )
}
