import { type CampaignDirection, type TrainingCampaign } from '../../lib/api'

type Props = {
  datasetReady: boolean
  datasetName: string | null
  direction: CampaignDirection
  attempts: number
  parallelism: number
  starting: boolean
  campaign: TrainingCampaign | null
  onDirectionChange: (direction: CampaignDirection) => void
  onAttemptsChange: (attempts: number) => void
  onParallelismChange: (parallelism: number) => void
  onStart: () => void
  onStop: () => void
}

const DIRECTIONS: Array<{ value: CampaignDirection; label: string; description: string }> = [
  {
    value: 'conservative',
    label: 'Conservative',
    description: 'Favor proven presets and stable quality.'
  },
  {
    value: 'balanced',
    label: 'Balanced',
    description: 'Mix safe retries with moderate exploration.'
  },
  {
    value: 'exploratory',
    label: 'Exploratory',
    description: 'Search aggressively for better checkpoints.'
  },
]

function isCampaignRunning(campaign: TrainingCampaign | null) {
  return campaign?.status === 'planning' || campaign?.status === 'running'
}

export function AutopilotCard({
  datasetReady,
  datasetName,
  direction,
  attempts,
  parallelism,
  starting,
  campaign,
  onDirectionChange,
  onAttemptsChange,
  onParallelismChange,
  onStart,
  onStop,
}: Props) {
  const running = isCampaignRunning(campaign)
  const attemptsCreated = Number(campaign?.summary.attempts_created ?? 0)
  const progressPct = campaign
    ? Math.min(100, (attemptsCreated / Math.max(1, campaign.attempt_count)) * 100)
    : 0
  const bestScore = Number(campaign?.planner_state.best_score)

  return (
    <section className="rounded-xl border-2 border-accent/30 bg-surface">
      <div className="border-b border-accent/20 bg-accent-dim/30 px-4 py-3">
        <div className="flex items-center justify-between gap-3">
          <div>
            <h2 className="text-heading text-sm font-semibold">Autopilot Training</h2>
            <p className="text-muted text-xs mt-0.5">Primary path for checkpoint search and promotion.</p>
          </div>
          {campaign && (
            <span className="rounded-full bg-raised px-2 py-0.5 text-[10px] font-mono uppercase tracking-wider text-primary">
              {campaign.status}
            </span>
          )}
        </div>
      </div>

      <div className="space-y-3 px-4 py-4">
        <div className={`rounded-lg border px-3 py-2 text-xs ${datasetReady ? 'border-accent/20 bg-accent-dim text-primary' : 'border-warning/20 bg-warning-dim text-warning'}`}>
          <span className="font-semibold">Dataset:</span>{' '}
          {datasetReady ? `${datasetName ?? 'Linked dataset'} is ready.` : 'Finalize/select a dataset before starting autopilot.'}
        </div>

        {!running && (
          <>
            <div className="grid gap-2 sm:grid-cols-3">
              {DIRECTIONS.map((item) => (
                <button
                  key={item.value}
                  type="button"
                  onClick={() => onDirectionChange(item.value)}
                  className={`rounded-lg border px-3 py-2 text-left transition-colors ${
                    direction === item.value
                      ? 'border-accent bg-accent-dim'
                      : 'border-edge bg-raised hover:border-accent/50'
                  }`}
                >
                  <p className="text-xs font-semibold text-primary">{item.label}</p>
                  <p className="mt-1 text-[10px] text-muted">{item.description}</p>
                </button>
              ))}
            </div>

            <div className="grid gap-3 sm:grid-cols-2">
              <label className="text-xs text-subtle">
                Attempts
                <input
                  type="number"
                  min={1}
                  max={12}
                  value={attempts}
                  onChange={(event) => onAttemptsChange(parseInt(event.target.value, 10) || 1)}
                  className="mt-1 w-full rounded-lg border border-edge bg-raised px-3 py-2 text-sm text-primary"
                />
              </label>
              <label className="text-xs text-subtle">
                Parallelism
                <input
                  type="number"
                  min={1}
                  max={6}
                  value={parallelism}
                  onChange={(event) => onParallelismChange(parseInt(event.target.value, 10) || 1)}
                  className="mt-1 w-full rounded-lg border border-edge bg-raised px-3 py-2 text-sm text-primary"
                />
              </label>
            </div>
          </>
        )}

        {campaign && (
          <div className="rounded-lg border border-edge bg-raised px-3 py-3">
            <div className="flex items-center justify-between text-xs text-primary">
              <span>{attemptsCreated}/{campaign.attempt_count} attempts</span>
              <span className="font-mono">{progressPct.toFixed(0)}%</span>
            </div>
            <div className="mt-2 h-2 overflow-hidden rounded-full bg-edge">
              <div className="h-full bg-accent transition-all" style={{ width: `${progressPct}%` }} />
            </div>
            <div className="mt-2 flex flex-wrap gap-3 text-[10px] font-mono text-muted">
              <span>phase={String(campaign.planner_state.phase ?? 'n/a')}</span>
              <span>best={Number.isFinite(bestScore) ? bestScore.toFixed(3) : 'n/a'}</span>
            </div>
          </div>
        )}

        {running ? (
          <button
            type="button"
            onClick={onStop}
            className="w-full rounded-lg border border-warning/40 bg-warning-dim px-4 py-3 text-xs font-semibold text-warning hover:border-warning"
          >
            Stop Campaign
          </button>
        ) : (
          <button
            type="button"
            onClick={onStart}
            disabled={!datasetReady || starting}
            className="w-full rounded-lg bg-accent px-4 py-3 text-sm font-semibold text-void disabled:opacity-50"
          >
            {starting ? 'Starting...' : 'Start Autopilot Training'}
          </button>
        )}
      </div>
    </section>
  )
}
