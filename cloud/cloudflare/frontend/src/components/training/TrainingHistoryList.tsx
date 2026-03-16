import { useMemo, useState } from 'react'
import { type TrainingJob } from '../../lib/api'
import { shouldWatchTrainingJob } from '../../lib/trainingCheckout'
import { TrainingJobRow } from './TrainingJobRow'

type Props = {
  jobs: TrainingJob[]
  voiceName?: string
  onCancel?: (jobId: string) => Promise<void> | void
  onRefresh?: (jobId: string) => Promise<void>
  onReconcile?: (jobId: string) => Promise<void>
  onRevalidate?: (jobId: string) => Promise<void>
  defaultCollapsed?: boolean
}

export function TrainingHistoryList({
  jobs,
  voiceName,
  onCancel,
  onRefresh,
  onReconcile,
  onRevalidate,
  defaultCollapsed = true,
}: Props) {
  const [collapsed, setCollapsed] = useState(defaultCollapsed)
  const historyJobs = useMemo(() => (
    [...jobs]
      .filter((job) => !shouldWatchTrainingJob(job) && (job.status === 'completed' || job.status === 'failed' || job.status === 'cancelled'))
      .sort((a, b) => (b.updated_at ?? b.created_at) - (a.updated_at ?? a.created_at))
  ), [jobs])

  return (
    <section className="rounded-xl border border-edge bg-raised p-4">
      <button
        type="button"
        onClick={() => setCollapsed((value) => !value)}
        className="flex w-full items-center justify-between text-left"
      >
        <h3 className="text-heading text-sm font-semibold">Training History</h3>
        <span className="text-muted text-xs font-mono">
          {collapsed ? `show (${historyJobs.length})` : 'hide'}
        </span>
      </button>

      {!collapsed && (
        <div className="mt-4 space-y-3">
          {historyJobs.length === 0 ? (
            <p className="rounded-lg border border-dashed border-edge bg-surface px-3 py-6 text-center text-sm text-subtle">
              No completed attempts yet.
            </p>
          ) : (
            historyJobs.map((job) => (
              <TrainingJobRow
                key={job.job_id}
                job={job}
                voiceName={voiceName}
                compact
                onCancel={onCancel}
                onRefresh={onRefresh}
                onReconcile={onReconcile}
                onRevalidate={onRevalidate}
              />
            ))
          )}
        </div>
      )}
    </section>
  )
}
