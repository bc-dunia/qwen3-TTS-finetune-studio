import type { CheckpointCandidate } from './types'
import { getCandidateTitle } from './utils'

export function CompareTray({
  selectedCandidates,
  onRemove,
  onGenerate,
  generating,
  hasText,
  maxSelect,
}: {
  selectedCandidates: CheckpointCandidate[]
  onRemove: (id: string) => void
  onGenerate: () => void
  generating: boolean
  hasText: boolean
  maxSelect: number
}) {
  if (selectedCandidates.length === 0) return null

  const disableGenerate = !hasText || generating || selectedCandidates.length === 0

  return (
    <div className="fixed bottom-0 left-0 right-0 z-50 border-t border-edge bg-raised shadow-lg animate-slide-up">
      <div className="max-w-7xl mx-auto px-4 py-3 flex flex-wrap items-center gap-3">
        {/* Selected chips */}
        <div className="flex-1 flex flex-wrap items-center gap-2 min-w-0">
          {selectedCandidates.map((candidate) => (
            <span
              key={candidate.id}
              className="inline-flex items-center gap-1.5 rounded-full bg-surface border border-edge px-2.5 py-1 text-[11px] font-medium text-primary"
            >
              <span className="truncate max-w-32">
                {candidate.runName ?? getCandidateTitle(candidate)}
              </span>
              {candidate.epoch != null && (
                <span className="text-muted font-mono">e{candidate.epoch}</span>
              )}
              <button
                onClick={() => onRemove(candidate.id)}
                className="ml-0.5 text-muted hover:text-error transition-colors"
                type="button"
                aria-label={`Remove ${candidate.runName ?? 'checkpoint'}`}
              >
                <svg className="w-3 h-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
                  <line x1="18" y1="6" x2="6" y2="18" />
                  <line x1="6" y1="6" x2="18" y2="18" />
                </svg>
              </button>
            </span>
          ))}
          <span className="text-[10px] font-mono text-muted">
            {selectedCandidates.length}/{maxSelect}
          </span>
        </div>

        {/* Generate CTA */}
        <button
          onClick={onGenerate}
          disabled={disableGenerate}
          className="shrink-0 bg-accent hover:bg-accent-light text-void font-semibold text-sm py-2.5 px-5 rounded-lg disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          type="button"
        >
          {generating
            ? 'Generating\u2026'
            : `Generate ${selectedCandidates.length} Checkpoint${selectedCandidates.length === 1 ? '' : 's'}`}
        </button>
      </div>
    </div>
  )
}
