import type { CheckpointCandidate } from './types'
import { ScoreBand } from './ScoreBand'
import { getCandidateTitle } from './utils'

function getComparisonSummary(
  trusted: CheckpointCandidate | null,
  recommended: CheckpointCandidate | null,
  currentProduction: CheckpointCandidate | null,
): string {
  if (!trusted && !recommended) return 'No checkpoints to compare yet'
  if (!trusted && currentProduction) return 'Current checkpoint is invalidated'
  if (!trusted) return 'No trusted checkpoint in this cycle'
  if (!recommended) return 'No recommendation yet'
  const tScore = trusted.score ?? -1
  const rScore = recommended.score ?? -1
  if (rScore > tScore + 0.01) return 'Candidate scores higher overall'
  if (tScore > rScore + 0.01) return 'Current checkpoint leads'
  return 'Scores are close \u2014 listen carefully'
}

export function DecisionHeader({
  trustedCandidate,
  recommendedCandidate,
  currentProductionCandidate,
  refreshing,
  onRefresh,
}: {
  trustedCandidate: CheckpointCandidate | null
  recommendedCandidate: CheckpointCandidate | null
  currentProductionCandidate: CheckpointCandidate | null
  refreshing: boolean
  onRefresh: () => void
}) {
  const summary = getComparisonSummary(
    trustedCandidate,
    recommendedCandidate,
    currentProductionCandidate,
  )

  return (
    <div className="rounded-xl border border-edge bg-raised p-5">
      <div className="flex flex-col lg:flex-row lg:items-center gap-4 lg:gap-6">
        {/* LEFT: Trusted */}
        <div className="flex-1 min-w-0">
          <div className="text-[10px] font-mono uppercase tracking-wider text-muted mb-1">Trusted</div>
          {trustedCandidate ? (
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-primary text-sm font-semibold truncate">
                {trustedCandidate.runName ?? 'current'}
              </span>
              <ScoreBand score={trustedCandidate.score} />
              {trustedCandidate.epoch != null && (
                <span className="text-[10px] font-mono text-muted">epoch {trustedCandidate.epoch}</span>
              )}
            </div>
          ) : currentProductionCandidate ? (
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-error text-sm font-semibold">Invalidated</span>
              <span className="text-[10px] font-mono text-muted">
                {currentProductionCandidate.runName ?? 'current'}
              </span>
            </div>
          ) : (
            <span className="text-muted text-sm">None</span>
          )}
        </div>

        {/* CENTER: Comparison summary */}
        <div className="flex-shrink-0 text-center lg:max-w-64">
          <div className="text-subtle text-xs font-medium">{summary}</div>
        </div>

        {/* RIGHT: Recommended + Refresh */}
        <div className="flex-1 min-w-0 flex items-center justify-between gap-3 lg:justify-end">
          <div className="lg:text-right">
            <div className="text-[10px] font-mono uppercase tracking-wider text-muted mb-1">Recommended</div>
            {recommendedCandidate ? (
              <div className="flex flex-wrap items-center gap-2 lg:justify-end">
                <span className="text-primary text-sm font-semibold truncate">
                  {recommendedCandidate.runName ?? getCandidateTitle(recommendedCandidate)}
                </span>
                <ScoreBand score={recommendedCandidate.score} />
                {recommendedCandidate.epoch != null && (
                  <span className="text-[10px] font-mono text-muted">epoch {recommendedCandidate.epoch}</span>
                )}
              </div>
            ) : (
              <span className="text-muted text-sm">Waiting</span>
            )}
          </div>
          <button
            onClick={onRefresh}
            disabled={refreshing}
            className="shrink-0 inline-flex items-center rounded-lg border border-edge px-3 py-1.5 text-[11px] font-semibold text-primary transition-colors hover:border-accent hover:text-accent disabled:opacity-50"
            type="button"
          >
            {refreshing ? (
              <svg className="w-3.5 h-3.5 animate-spin" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M21 12a9 9 0 1 1-6.219-8.56" />
              </svg>
            ) : (
              'Refresh'
            )}
          </button>
        </div>
      </div>
      <p className="mt-3 text-subtle text-xs border-t border-edge pt-3">
        Listen to trusted, recommended, and rejected checkpoints side by side before accepting a new version.
      </p>
    </div>
  )
}
