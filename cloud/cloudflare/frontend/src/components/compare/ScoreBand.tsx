import { formatScore, scoreBandLabel, scoreBandStyle, scoreColor } from '../../lib/voiceScoreUi'

export function ScoreBand({
  score,
  size = 'sm',
}: {
  score: number | null
  size?: 'sm' | 'md'
}) {
  const label = scoreBandLabel(score)
  const style = scoreBandStyle(score)
  const sizeClasses = size === 'md' ? 'px-2.5 py-1 text-xs' : 'px-2 py-0.5 text-[10px]'

  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full font-semibold ${style.text} ${style.bg} ${sizeClasses}`}>
      <span>{label}</span>
      {score != null && (
        <span className="font-mono opacity-70">{formatScore(score)}</span>
      )}
    </span>
  )
}

export function TraitBars({
  style: styleScore,
  tone,
  speed,
}: {
  style?: number | null
  tone?: number | null
  speed?: number | null
}) {
  const traits = [
    { label: 'Style', value: styleScore },
    { label: 'Tone', value: tone },
    { label: 'Speed', value: speed },
  ].filter((t) => t.value != null)

  if (traits.length === 0) return null

  return (
    <div className="space-y-1.5">
      {traits.map((trait) => (
        <TraitBar key={trait.label} label={trait.label} value={trait.value!} />
      ))}
    </div>
  )
}

function TraitBar({ label, value }: { label: string; value: number }) {
  const pct = Math.max(0, Math.min(100, value * 100))
  const color = scoreColor(value)

  return (
    <div className="flex items-center gap-2">
      <span className="text-[10px] font-mono text-muted w-10 shrink-0">{label}</span>
      <div className="flex-1 h-1.5 bg-edge rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-[width] duration-300 ${color === 'text-accent' ? 'bg-accent' : color === 'text-warning' ? 'bg-warning' : color === 'text-error' ? 'bg-error' : 'bg-muted'}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className={`text-[10px] font-mono shrink-0 ${color}`}>{formatScore(value)}</span>
    </div>
  )
}
