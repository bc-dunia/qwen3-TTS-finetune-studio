export const MAX_COMPARE_CANDIDATES = 4

export type CheckpointCandidate = {
  id: string
  prefix: string
  epoch: number | null
  score: number | null
  preset: string | null
  message: string | null
  jobId: string | null
  createdAt: number
  completedAt: number | null
  attemptNumber: number | null
  runName: string | null
  isCurrentProduction: boolean
  isStoredCandidate: boolean
  isJobRecommendation: boolean
  validationPassed: boolean
  toneScore: number | null
  speedScore: number | null
  styleScore: number | null
}

export type RunSummary = {
  jobId: string
  attemptNumber: number | null
  createdAt: number
  startedAt: number | null
  completedAt: number | null
  durationMs: number | null
  status: string
  championScore: number | null
  championEpoch: number | null
  championPreset: string | null
  validationMessage: string | null
  hasCandidates: boolean
  validationPassed: boolean
  validationRejected: boolean
}

export type CompareResult = {
  status: string
  blob?: Blob
  error?: string
}
