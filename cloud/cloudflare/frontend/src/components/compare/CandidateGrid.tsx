import { useState } from 'react'
import { useNavigate, useParams } from 'react-router'
import type { CheckpointCandidate, CompareResult } from './types'
import { MAX_COMPARE_CANDIDATES } from './types'
import { CandidateCard } from './CandidateCard'

type FilterTab = 'all' | 'passed' | 'rejected'

export function CandidateGrid({
  candidates,
  selectedIds,
  onToggle,
  results,
  generating,
  applyingCandidateId,
  onApply,
  selectionError,
  onSelectPreset,
}: {
  candidates: CheckpointCandidate[]
  selectedIds: Set<string>
  onToggle: (id: string) => void
  results: Record<string, CompareResult>
  generating: boolean
  applyingCandidateId: string
  onApply: (candidate: CheckpointCandidate) => void
  selectionError: string
  onSelectPreset: (preset: 'trusted-recommended' | 'recommended-rejected' | 'clear') => void
}) {
  const { voiceId = '' } = useParams()
  const navigate = useNavigate()
  const [filter, setFilter] = useState<FilterTab>('all')

  const filtered = candidates.filter((c) => {
    if (filter === 'passed') return c.validationPassed
    if (filter === 'rejected') return !c.validationPassed && !c.isCurrentProduction
    return true
  })

  const tabs: { key: FilterTab; label: string; count: number }[] = [
    { key: 'all', label: 'All', count: candidates.length },
    { key: 'passed', label: 'Passed', count: candidates.filter((c) => c.validationPassed).length },
    { key: 'rejected', label: 'Rejected', count: candidates.filter((c) => !c.validationPassed && !c.isCurrentProduction).length },
  ]

  if (candidates.length === 0) {
    return (
      <div className="rounded-xl border border-edge bg-raised p-5">
        <h2 className="text-heading font-semibold text-sm mb-4">Checkpoint Candidates</h2>
        <div className="rounded-xl border border-dashed border-edge bg-surface px-4 py-8 text-center">
          <div className="text-primary text-sm font-semibold">No checkpoints in the current training cycle yet</div>
          <p className="mt-2 text-subtle text-sm">
            Start a fresh training run first. As soon as checkpoints are saved, this page will surface trusted, recommended, and rejected candidates here.
          </p>
          <button
            onClick={() => navigate(`/voices/${voiceId}/training`)}
            className="mt-4 inline-flex items-center rounded-lg bg-accent px-3 py-2 text-[11px] font-semibold text-void transition-colors hover:bg-accent-light"
            type="button"
          >
            Open Training
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="rounded-xl border border-edge bg-raised p-5">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
        <div className="flex items-center gap-3">
          <h2 className="text-heading font-semibold text-sm">Checkpoint Candidates</h2>
          <span className="text-muted text-[10px] font-mono">{candidates.length} total</span>
        </div>
        <span className="text-muted text-[10px] font-mono">max {MAX_COMPARE_CANDIDATES} selected</span>
      </div>

      {/* Filter tabs */}
      <div className="flex items-center gap-1 mb-3">
        {tabs.map((tab) => (
          <button
            key={tab.key}
            onClick={() => setFilter(tab.key)}
            className={`rounded-lg px-3 py-1.5 text-[11px] font-semibold transition-colors ${
              filter === tab.key
                ? 'bg-accent-dim text-accent'
                : 'text-muted hover:text-primary'
            }`}
            type="button"
          >
            {tab.label}
            <span className="ml-1 opacity-60">{tab.count}</span>
          </button>
        ))}
      </div>

      {/* Quick selection buttons */}
      <div className="flex flex-wrap gap-2 mb-4">
        <button
          onClick={() => onSelectPreset('trusted-recommended')}
          className="inline-flex items-center rounded-lg border border-edge px-2.5 py-1.5 text-[10px] font-semibold text-primary transition-colors hover:border-accent hover:text-accent"
          type="button"
        >
          Trusted + Recommended
        </button>
        <button
          onClick={() => onSelectPreset('recommended-rejected')}
          className="inline-flex items-center rounded-lg border border-edge px-2.5 py-1.5 text-[10px] font-semibold text-primary transition-colors hover:border-accent hover:text-accent"
          type="button"
        >
          Recommended + Rejected
        </button>
        <button
          onClick={() => onSelectPreset('clear')}
          className="inline-flex items-center rounded-lg border border-edge px-2.5 py-1.5 text-[10px] font-semibold text-muted transition-colors hover:text-primary"
          type="button"
        >
          Clear
        </button>
      </div>

      {selectionError && (
        <div className="rounded-lg border border-warning/20 bg-warning-dim px-3 py-2 text-warning text-xs mb-4">
          {selectionError}
        </div>
      )}

      {/* Grid */}
      <div className="grid xl:grid-cols-2 gap-3">
        {filtered.map((candidate) => {
          const isSelected = selectedIds.has(candidate.id)
          const result = results[candidate.id]
          const hasResult = isSelected && result !== undefined

          return (
            <CandidateCard
              key={candidate.id}
              candidate={candidate}
              selected={isSelected}
              onToggle={() => onToggle(candidate.id)}
              mode={hasResult ? 'listening' : 'browse'}
              result={isSelected ? result : undefined}
              generating={generating}
              applyingId={applyingCandidateId}
              onApply={() => onApply(candidate)}
            />
          )
        })}
      </div>
    </div>
  )
}
