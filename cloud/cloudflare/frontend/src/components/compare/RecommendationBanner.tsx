import type { CheckpointCandidate, CompareResult } from './types'
import { getCandidateTitle } from './utils'
import { ScoreBand } from './ScoreBand'

function getRecommendationText(
  trusted: CheckpointCandidate | null,
  recommended: CheckpointCandidate | null,
): string {
  if (!trusted && !recommended) return 'No checkpoints available for comparison.'
  if (!trusted && recommended) return 'No trusted baseline. Consider promoting the recommended checkpoint.'
  if (trusted && !recommended) return 'Current checkpoint has no challenger yet.'

  const tScore = trusted!.score ?? -1
  const rScore = recommended!.score ?? -1

  if (rScore > tScore + 0.01) return 'Candidate outperforms current. Consider promoting.'
  if (Math.abs(rScore - tScore) <= 0.01) return 'Scores are close. Listen carefully before deciding.'
  return 'Current checkpoint remains the strongest option.'
}

export function RecommendationBanner({
  trustedCandidate,
  recommendedCandidate,
  results,
  onApply,
  applyingCandidateId,
}: {
  trustedCandidate: CheckpointCandidate | null
  recommendedCandidate: CheckpointCandidate | null
  results: Record<string, CompareResult>
  onApply: (candidate: CheckpointCandidate) => void
  applyingCandidateId: string
}) {
  const trustedGenerated = trustedCandidate ? results[trustedCandidate.id]?.status === 'Completed' : false
  const recommendedGenerated = recommendedCandidate ? results[recommendedCandidate.id]?.status === 'Completed' : false
  if (!trustedGenerated && !recommendedGenerated) return null

  const text = getRecommendationText(trustedCandidate, recommendedCandidate)

  return (
    <div className="rounded-xl border border-edge bg-raised p-5 animate-slide-up">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div className="space-y-2 flex-1">
          <div className="text-heading text-sm font-semibold">Recommendation</div>
          <p className="text-subtle text-sm">{text}</p>

          {/* Score comparison */}
          <div className="flex flex-wrap items-center gap-3 text-xs">
            {trustedCandidate && (
              <div className="flex items-center gap-1.5">
                <span className="text-muted">Trusted:</span>
                <ScoreBand score={trustedCandidate.score} />
              </div>
            )}
            {recommendedCandidate && (
              <div className="flex items-center gap-1.5">
                <span className="text-muted">Recommended:</span>
                <ScoreBand score={recommendedCandidate.score} />
              </div>
            )}
          </div>
        </div>

        {/* Action buttons */}
        <div className="flex flex-wrap gap-2 shrink-0">
          {recommendedGenerated && recommendedCandidate && recommendedCandidate.jobId && !recommendedCandidate.isCurrentProduction && (
            <button
              onClick={() => onApply(recommendedCandidate)}
              disabled={applyingCandidateId === recommendedCandidate.id}
              className="bg-accent hover:bg-accent-light text-void font-semibold text-[11px] py-2 px-4 rounded-lg disabled:opacity-50 transition-colors"
              type="button"
            >
              {applyingCandidateId === recommendedCandidate.id
                ? 'Promoting\u2026'
                : `Promote ${getCandidateTitle(recommendedCandidate)}`}
            </button>
          )}
          {trustedCandidate && (
            <span className="inline-flex items-center border border-edge px-4 py-2 text-[11px] font-semibold text-muted rounded-lg">
              Keep Current
            </span>
          )}
        </div>
      </div>
    </div>
  )
}
