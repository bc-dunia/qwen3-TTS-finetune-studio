import { Link } from 'react-router'
import type { TrainingConfig } from '../lib/api'
import type { TrainingAdvice } from '../lib/trainingAdvisor'

function formatSuggestedConfig(config: TrainingConfig): string {
  return `batch=${config.batch_size} epochs=${config.num_epochs} lr=${config.learning_rate} grad_acc=${config.gradient_accumulation_steps ?? 4} subtalker=${config.subtalker_loss_weight ?? 0} seed=${config.seed ?? 0} gpu=${config.gpu_type_id ?? 'auto'}`
}

const CONFIDENCE_STYLES = {
  high: 'bg-accent-dim text-accent',
  medium: 'bg-surface text-muted',
}

export function TrainingAdviceCard({
  voiceId,
  advice,
  onApplyConfig,
  compact = false,
  showCompareLink = true,
  showTrainingLink = true,
}: {
  voiceId: string
  advice: TrainingAdvice | null
  onApplyConfig?: (config: TrainingConfig) => void
  compact?: boolean
  showCompareLink?: boolean
  showTrainingLink?: boolean
}) {
  if (!advice) return null

  return (
    <div className="rounded-xl border border-edge bg-raised p-5 space-y-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <div className="text-heading text-sm font-semibold">Training Advisor</div>
          <div className="mt-1 text-primary text-sm">{advice.title}</div>
        </div>
        <span className={`rounded-full px-2 py-0.5 text-[10px] font-mono uppercase tracking-wider ${CONFIDENCE_STYLES[advice.confidence]}`}>
          {advice.confidence}
        </span>
      </div>

      <p className="text-sm leading-relaxed text-subtle">{advice.summary}</p>

      {advice.suggestedConfig && (
        <div className="rounded-lg border border-edge bg-surface px-3 py-2 text-[11px] font-mono text-muted">
          {formatSuggestedConfig(advice.suggestedConfig)}
        </div>
      )}

      <div className="space-y-2">
        {advice.reasons.map((reason) => (
          <div key={reason} className="rounded-lg border border-edge bg-surface px-3 py-2 text-[12px] text-subtle">
            {reason}
          </div>
        ))}
      </div>

      <div className="flex flex-wrap gap-2">
        {onApplyConfig && advice.suggestedConfig && !advice.reviewDatasetFirst && (
          <button
            onClick={() => onApplyConfig(advice.suggestedConfig!)}
            className="inline-flex items-center rounded-lg bg-accent px-3 py-2 text-[11px] font-semibold text-void transition-colors hover:bg-accent-light"
            type="button"
          >
            Apply Suggested Setup
          </button>
        )}
        {advice.compareFirst && showCompareLink && (
          <Link
            to={`/voices/${voiceId}/compare`}
            className="inline-flex items-center rounded-lg border border-edge px-3 py-2 text-[11px] font-semibold text-primary transition-colors hover:border-accent hover:text-accent"
          >
            Open Compare
          </Link>
        )}
        {advice.reviewDatasetFirst && (
          <Link
            to={`/voices/${voiceId}/dataset`}
            className="inline-flex items-center rounded-lg border border-edge px-3 py-2 text-[11px] font-semibold text-primary transition-colors hover:border-accent hover:text-accent"
          >
            Review Dataset
          </Link>
        )}
        {!compact && showTrainingLink && (
          <Link
            to="/training"
            className="inline-flex items-center rounded-lg border border-edge px-3 py-2 text-[11px] font-semibold text-muted transition-colors hover:text-primary"
          >
            Open Training
          </Link>
        )}
      </div>
    </div>
  )
}
