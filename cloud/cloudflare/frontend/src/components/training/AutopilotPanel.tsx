import { useEffect, useState } from 'react'
import { Link } from 'react-router'
import type {
  CampaignDirection,
  TrainingAdvice,
  TrainingCampaign,
  TrainingConfig,
  Voice,
} from '../../lib/api'

type GoalKey = 'first_checkpoint' | 'improve_score' | 'fix_failed'

const GOALS: Array<{
  value: GoalKey
  label: string
  direction: CampaignDirection
  hint: string
}> = [
  {
    value: 'first_checkpoint',
    label: 'Get first checkpoint',
    direction: 'balanced',
    hint: 'Balanced mix of safe retries and exploration.',
  },
  {
    value: 'improve_score',
    label: 'Improve best score',
    direction: 'conservative',
    hint: 'Refine around proven configs for higher quality.',
  },
  {
    value: 'fix_failed',
    label: 'Recover from failures',
    direction: 'exploratory',
    hint: 'Search aggressively across different config regions.',
  },
]

function detectBestGoal(
  voice: Voice | null,
  campaign: TrainingCampaign | null,
): GoalKey {
  const hasCheckpoint = Boolean(voice?.run_name) || Boolean(voice?.checkpoint_r2_prefix)
  const campaignBadOutcome =
    campaign?.status === 'failed' ||
    campaign?.status === 'blocked_dataset' ||
    campaign?.status === 'blocked_budget'

  if (campaignBadOutcome && !hasCheckpoint) return 'fix_failed'
  if (!hasCheckpoint) return 'first_checkpoint'
  if (campaignBadOutcome) return 'fix_failed'
  return 'improve_score'
}

function goalFromDirection(direction: CampaignDirection): GoalKey {
  const match = GOALS.find((g) => g.direction === direction)
  return match?.value ?? 'first_checkpoint'
}

function isCampaignRunning(campaign: TrainingCampaign | null) {
  return campaign?.status === 'planning' || campaign?.status === 'running'
}

function formatCompactConfig(config: TrainingConfig): string {
  const parts: string[] = []
  if (config.batch_size != null) parts.push(`batch=${config.batch_size}`)
  if (config.num_epochs != null) parts.push(`epochs=${config.num_epochs}`)
  if (config.learning_rate != null) parts.push(`lr=${config.learning_rate}`)
  if (config.seed != null) parts.push(`seed=${config.seed}`)
  return parts.join(' ')
}

type Props = {
  voiceId: string
  datasetReady: boolean
  datasetName: string | null
  direction: CampaignDirection
  attempts: number
  parallelism: number
  starting: boolean
  campaign: TrainingCampaign | null
  trainingAdvice: TrainingAdvice | null
  voice: Voice | null
  onDirectionChange: (direction: CampaignDirection) => void
  onAttemptsChange: (attempts: number) => void
  onParallelismChange: (parallelism: number) => void
  onStart: () => void
  onStop: () => void
  onApplyConfig?: (config: TrainingConfig) => void
}

export function AutopilotPanel({
  voiceId,
  datasetReady,
  datasetName,
  direction,
  attempts,
  parallelism,
  starting,
  campaign,
  trainingAdvice,
  voice,
  onDirectionChange,
  onAttemptsChange,
  onParallelismChange,
  onStart,
  onStop,
  onApplyConfig,
}: Props) {
  const running = isCampaignRunning(campaign)
  const recommendedGoal = detectBestGoal(voice, campaign)
  const recommendedMeta = GOALS.find((g) => g.value === recommendedGoal)!
  const selectedGoal = goalFromDirection(direction)
  const isUsingRecommended = selectedGoal === recommendedGoal
  const [showOverride, setShowOverride] = useState(false)
  useEffect(() => { setShowOverride(false) }, [voiceId])
  const goalMeta = GOALS.find((g) => g.value === selectedGoal)

  const attemptsCreated = Number(campaign?.summary.attempts_created ?? 0)
  const progressPct = campaign
    ? Math.min(100, (attemptsCreated / Math.max(1, campaign.attempt_count)) * 100)
    : 0
  const campaignBestScore = Number(campaign?.planner_state.best_score)

  const voiceBestScore = voice?.checkpoint_score
  const voiceBestEpoch = voice?.epoch
  const hasCheckpoint =
    Boolean(voice?.run_name) || Boolean(voice?.checkpoint_r2_prefix)

  function handleGoalChange(goalValue: string) {
    const goal = GOALS.find((g) => g.value === goalValue)
    if (goal) {
      onDirectionChange(goal.direction)
    }
  }

  return (
    <section className="rounded-xl border-2 border-accent/30 bg-surface overflow-hidden">
      <div className="border-b border-accent/20 bg-accent-dim/30 px-5 py-3">
        <div className="flex items-center justify-between gap-3">
          <div>
            <h2 className="text-heading text-sm font-semibold">
              Autopilot Training
            </h2>
            <p className="text-muted text-[11px] mt-0.5">
              Goal-driven checkpoint search and promotion.
            </p>
          </div>
          {campaign && (
            <span className="rounded-full bg-raised px-2.5 py-0.5 text-[10px] font-mono uppercase tracking-wider text-primary">
              {campaign.status}
            </span>
          )}
        </div>
      </div>

      <div className="px-5 py-5 space-y-4">
        <div
          className={`rounded-lg border px-3 py-2 text-xs ${
            datasetReady
              ? 'border-accent/20 bg-accent-dim text-primary'
              : 'border-warning/20 bg-warning-dim text-warning'
          }`}
        >
          {datasetReady ? (
            <>
              <span className="font-semibold">Dataset:</span>{' '}
              {datasetName || (hasCheckpoint ? 'Previous training dataset' : 'Linked dataset')} is ready.
            </>
          ) : (
            <>
              <span className="font-semibold">Dataset needed.</span>{' '}
              <Link
                to={`/voices/${voiceId}/dataset`}
                className="underline hover:no-underline"
              >
                Finalize a dataset
              </Link>{' '}
              before starting autopilot.
            </>
          )}
        </div>

        {!running && (
          <div className="space-y-3">
            <div className="rounded-lg border border-edge bg-raised px-4 py-3">
              <div className="flex items-center justify-between gap-3">
                <div className="min-w-0">
                  <div className="text-xs font-semibold text-heading">
                    {isUsingRecommended ? recommendedMeta.label : goalMeta?.label}
                  </div>
                  <p className="text-[11px] text-muted mt-0.5">
                    {isUsingRecommended ? recommendedMeta.hint : goalMeta?.hint}
                  </p>
                </div>
                {isUsingRecommended && (
                  <span className="shrink-0 rounded-full bg-accent-dim px-2 py-0.5 text-[9px] font-mono uppercase tracking-wider text-accent">
                    recommended
                  </span>
                )}
              </div>
              {!showOverride ? (
                <button
                  type="button"
                  onClick={() => setShowOverride(true)}
                  className="mt-2 text-[10px] text-muted hover:text-accent transition-colors"
                >
                  Change strategy...
                </button>
              ) : (
                <div className="mt-3 space-y-1.5">
                  {GOALS.map((goal) => (
                    <button
                      key={goal.value}
                      type="button"
                      onClick={() => {
                        handleGoalChange(goal.value)
                        setShowOverride(false)
                      }}
                      className={`w-full text-left rounded-lg border px-3 py-2 text-xs transition-colors ${
                        selectedGoal === goal.value
                          ? 'border-accent bg-accent-dim text-accent'
                          : 'border-edge bg-surface text-subtle hover:border-accent/40 hover:text-primary'
                      }`}
                    >
                      <span className="font-medium">{goal.label}</span>
                      {goal.value === recommendedGoal && (
                        <span className="ml-1.5 text-[9px] font-mono text-accent uppercase">best</span>
                      )}
                      <span className="block text-[10px] text-muted mt-0.5">{goal.hint}</span>
                    </button>
                  ))}
                </div>
              )}
            </div>

            <div className="grid gap-3 sm:grid-cols-2">
              <label className="text-xs text-subtle">
                Attempts
                <input
                  type="number"
                  min={1}
                  max={12}
                  value={attempts}
                  onChange={(e) =>
                    onAttemptsChange(parseInt(e.target.value, 10) || 1)
                  }
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
                  onChange={(e) =>
                    onParallelismChange(parseInt(e.target.value, 10) || 1)
                  }
                  className="mt-1 w-full rounded-lg border border-edge bg-raised px-3 py-2 text-sm text-primary"
                />
              </label>
            </div>
          </div>
        )}

        {running ? (
          <button
            type="button"
            onClick={onStop}
            className="w-full rounded-lg border border-warning/40 bg-warning-dim px-4 py-3.5 text-sm font-semibold text-warning hover:border-warning transition-colors"
          >
            Stop Campaign
          </button>
        ) : (
          <button
            type="button"
            onClick={onStart}
            disabled={!datasetReady || starting}
            className="w-full rounded-lg bg-accent px-4 py-3.5 text-sm font-bold text-void disabled:opacity-50 transition-colors hover:bg-accent-light"
          >
            {starting ? 'Starting...' : 'Start Autopilot'}
          </button>
        )}

        {campaign && (
          <div className="rounded-lg border border-edge bg-raised px-4 py-3 space-y-2">
            <div className="flex items-center justify-between text-xs text-primary">
              <span className="font-medium">
                {attemptsCreated}/{campaign.attempt_count} attempts
              </span>
              <span className="font-mono text-accent">
                {progressPct.toFixed(0)}%
              </span>
            </div>
            <div className="h-2 overflow-hidden rounded-full bg-edge">
              <div
                className="h-full bg-accent transition-all duration-300"
                style={{ width: `${progressPct}%` }}
              />
            </div>
            <div className="flex flex-wrap gap-x-4 gap-y-1 text-[10px] font-mono text-muted">
              <span>
                status={campaign.status}
              </span>
              <span>
                phase={String(campaign.planner_state.phase ?? 'n/a')}
              </span>
              <span>
                best=
                {Number.isFinite(campaignBestScore)
                  ? campaignBestScore.toFixed(3)
                  : 'n/a'}
              </span>
            </div>
          </div>
        )}

        {trainingAdvice && (
          <div className="rounded-lg border border-edge bg-raised px-4 py-3 space-y-2">
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-[11px] font-semibold text-heading">
                Advisor
              </span>
              <span className="rounded-full border border-edge bg-surface px-2 py-0.5 text-[10px] font-mono uppercase tracking-wider text-muted">
                {trainingAdvice.mode}
              </span>
              {trainingAdvice.analysisProvider === 'llm' && (
                <span className="rounded-full border border-accent/30 bg-accent-dim px-2 py-0.5 text-[10px] font-mono uppercase tracking-wider text-accent">
                  AI
                </span>
              )}
              <span
                className={`ml-auto rounded-full px-2 py-0.5 text-[10px] font-mono uppercase tracking-wider ${
                  trainingAdvice.confidence === 'high'
                    ? 'bg-accent-dim text-accent'
                    : 'bg-surface text-muted'
                }`}
              >
                {trainingAdvice.confidence}
              </span>
            </div>

            <p className="text-xs leading-relaxed text-subtle">
              {trainingAdvice.summary}
            </p>

            {trainingAdvice.suggestedConfig && (
              <div className="flex items-center gap-2">
                <div className="min-w-0 flex-1 rounded border border-edge bg-surface px-2 py-1 text-[10px] font-mono text-muted truncate">
                  {formatCompactConfig(trainingAdvice.suggestedConfig)}
                </div>
                {onApplyConfig && !trainingAdvice.reviewDatasetFirst && (
                  <button
                    type="button"
                    onClick={() =>
                      onApplyConfig(trainingAdvice.suggestedConfig!)
                    }
                    className="shrink-0 rounded border border-accent/40 bg-accent-dim px-2.5 py-1 text-[10px] font-semibold text-accent hover:border-accent transition-colors"
                  >
                    {trainingAdvice.primaryActionLabel ?? 'Apply'}
                  </button>
                )}
              </div>
            )}

            {trainingAdvice.reasons.length > 0 && (
              <div className="space-y-0.5">
                {trainingAdvice.reasons.map((reason) => (
                  <p
                    key={reason}
                    className="text-[11px] text-muted leading-relaxed"
                  >
                    &bull; {reason}
                  </p>
                ))}
              </div>
            )}

            {(trainingAdvice.compareFirst ||
              trainingAdvice.reviewDatasetFirst) && (
              <div className="flex flex-wrap gap-3 pt-1">
                {trainingAdvice.compareFirst && (
                  <Link
                    to={`/voices/${voiceId}/compare`}
                    className="text-[11px] font-semibold text-accent hover:text-accent-light transition-colors"
                  >
                    Open Compare &rarr;
                  </Link>
                )}
                {trainingAdvice.reviewDatasetFirst && (
                  <Link
                    to={`/voices/${voiceId}/dataset`}
                    className="text-[11px] font-semibold text-accent hover:text-accent-light transition-colors"
                  >
                    Review Dataset &rarr;
                  </Link>
                )}
              </div>
            )}
          </div>
        )}

        {hasCheckpoint && (
          <div className="rounded-lg border border-edge bg-raised px-4 py-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-6">
                <div>
                  <p className="text-[10px] font-mono text-muted uppercase tracking-wider">
                    Best Score
                  </p>
                  <p className="text-base font-bold text-accent tabular-nums">
                    {typeof voiceBestScore === 'number'
                      ? voiceBestScore.toFixed(3)
                      : 'n/a'}
                  </p>
                </div>
                {typeof voiceBestEpoch === 'number' && (
                  <div>
                    <p className="text-[10px] font-mono text-muted uppercase tracking-wider">
                      Epoch
                    </p>
                    <p className="text-base font-bold text-primary tabular-nums">
                      {voiceBestEpoch}
                    </p>
                  </div>
                )}
              </div>
              <Link
                to={`/voices/${voiceId}/compare`}
                className="rounded-lg border border-edge px-3 py-1.5 text-[11px] font-semibold text-subtle hover:border-accent hover:text-accent transition-colors"
              >
                Compare
              </Link>
            </div>
          </div>
        )}
      </div>
    </section>
  )
}
