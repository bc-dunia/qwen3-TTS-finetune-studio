import type { CheckpointCandidate } from './types'

export function getCandidateTitle(candidate: CheckpointCandidate): string {
  if (candidate.isCurrentProduction && candidate.validationPassed) return 'Trusted Current'
  if (candidate.isCurrentProduction) return 'Current Production (Invalidated)'
  if (candidate.isStoredCandidate) return 'Stored Candidate'
  if (candidate.validationPassed) return 'Recommended Candidate'
  if (!candidate.validationPassed) return 'Rejected Candidate'
  return candidate.runName ?? 'Checkpoint'
}
