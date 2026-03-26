import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router'
import {
  applyArenaCalibration,
  getArenaCalibration,
  type CalibrationResult,
} from '../lib/api'

const VALIDATION_RANKING_WEIGHTS: Record<string, number> = {
  asr: 0.25,
  speaker: 0.25,
  style: 0.20,
  tone: 0.05,
  speed: 0.05,
  overall: 0.05,
  duration: 0.05,
}

const METRIC_LABELS: Record<string, string> = {
  asr: 'ASR',
  speaker: 'Speaker',
  tone: 'Tone',
  speed: 'Speed',
  overall: 'Overall',
  duration: 'Duration',
  style: 'Style',
}

function getShiftColor(shift: number): string {
  if (Math.abs(shift) < 0.01) return 'text-muted'
  return shift > 0 ? 'text-accent' : 'text-error'
}

function getShiftArrow(shift: number): string {
  if (Math.abs(shift) < 0.01) return '~'
  return shift > 0 ? '\u2191' : '\u2193'
}

function getConfidenceBadge(confidence: string): { label: string; className: string } {
  switch (confidence) {
    case 'high':
      return { label: 'High Confidence', className: 'bg-accent-dim text-accent' }
    case 'calibrated':
      return { label: 'Calibrated', className: 'bg-warning-dim text-warning' }
    case 'preliminary':
      return { label: 'Preliminary', className: 'bg-raised text-muted' }
    default:
      return { label: 'Insufficient Data', className: 'bg-error-dim text-error' }
  }
}

function WeightBar({ label, current, learned, maxVal }: { label: string; current: number; learned: number; maxVal: number }) {
  const currentPct = maxVal > 0 ? (current / maxVal) * 100 : 0
  const learnedPct = maxVal > 0 ? (learned / maxVal) * 100 : 0

  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between text-xs">
        <span className="text-primary font-medium">{label}</span>
        <span className="text-muted font-mono text-[10px]">
          {current.toFixed(3)} / {learned.toFixed(3)}
        </span>
      </div>
      <div className="space-y-1">
        <div className="flex items-center gap-2">
          <span className="text-[9px] text-muted font-mono w-12 shrink-0">current</span>
          <div className="flex-1 h-3 bg-surface rounded overflow-hidden">
            <div
              className="h-full bg-edge rounded transition-[width] duration-300"
              style={{ width: `${Math.min(100, currentPct)}%` }}
            />
          </div>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-[9px] text-accent font-mono w-12 shrink-0">learned</span>
          <div className="flex-1 h-3 bg-surface rounded overflow-hidden">
            <div
              className="h-full bg-accent/60 rounded transition-[width] duration-300"
              style={{ width: `${Math.min(100, learnedPct)}%` }}
            />
          </div>
        </div>
      </div>
    </div>
  )
}

export function ArenaCalibration() {
  const { voiceId } = useParams()

  const [calibration, setCalibration] = useState<CalibrationResult | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [applying, setApplying] = useState(false)
  const [actionMessage, setActionMessage] = useState('')

  useEffect(() => {
    let cancelled = false

    async function load() {
      setLoading(true)
      setError('')
      try {
        const data = await getArenaCalibration(voiceId)
        if (!cancelled) setCalibration(data)
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : 'Failed to load calibration data')
      } finally {
        if (!cancelled) setLoading(false)
      }
    }

    void load()
    return () => { cancelled = true }
  }, [voiceId])

  async function handleApply() {
    if (!calibration) return
    setApplying(true)
    setError('')
    setActionMessage('')
    try {
      await applyArenaCalibration(calibration.learned_weights, voiceId)
      setActionMessage('Calibrated weights applied successfully!')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to apply calibration')
    } finally {
      setApplying(false)
    }
  }

  const canApply = calibration !== null &&
    (calibration.confidence === 'calibrated' || calibration.confidence === 'high')

  if (loading) {
    return (
      <div className="space-y-6 animate-pulse">
        <div className="h-7 bg-raised rounded w-64" />
        <div className="h-60 bg-raised rounded-xl" />
        <div className="h-40 bg-raised rounded-xl" />
      </div>
    )
  }

  const allMetrics = calibration
    ? [...new Set([...Object.keys(VALIDATION_RANKING_WEIGHTS), ...Object.keys(calibration.learned_weights)])]
    : Object.keys(VALIDATION_RANKING_WEIGHTS)

  const maxWeight = calibration
    ? Math.max(
        ...Object.values(VALIDATION_RANKING_WEIGHTS),
        ...Object.values(calibration.learned_weights),
        0.01,
      )
    : 0.3

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-heading text-lg font-bold">Calibration Insights</h2>
          <p className="text-subtle text-sm mt-1">
            Human arena preferences vs automated scoring weights.
          </p>
        </div>
        <Link
          to={voiceId ? `/voices/${voiceId}/arena` : '/voices'}
          className="inline-flex items-center rounded-lg border border-edge px-3 py-1.5 text-[11px] font-semibold text-primary transition-colors hover:border-accent hover:text-accent"
        >
          Back to Arena
        </Link>
      </div>

      {error && (
        <div className="rounded-lg border border-error/20 bg-error-dim px-3 py-2 text-sm text-error">{error}</div>
      )}
      {actionMessage && (
        <div className="rounded-lg border border-accent/20 bg-accent-dim px-4 py-3 text-accent text-sm">{actionMessage}</div>
      )}

      {!calibration ? (
        <div className="rounded-xl border border-dashed border-edge bg-raised px-4 py-12 text-center">
          <div className="text-primary text-sm font-semibold">No calibration data available</div>
          <p className="mt-2 text-subtle text-sm">
            Run arena sessions and vote on matches to generate calibration insights.
          </p>
        </div>
      ) : (
        <>
          <div className="grid md:grid-cols-4 gap-4">
            <div className="rounded-xl border border-edge bg-raised p-4">
              <div className="text-muted text-[10px] font-mono uppercase tracking-wider">Confidence</div>
              <div className="mt-2">
                <span className={`rounded-full px-2.5 py-1 text-xs font-mono ${getConfidenceBadge(calibration.confidence).className}`}>
                  {getConfidenceBadge(calibration.confidence).label}
                </span>
              </div>
            </div>
            <div className="rounded-xl border border-edge bg-raised p-4">
              <div className="text-muted text-[10px] font-mono uppercase tracking-wider">Matchups</div>
              <div className="mt-2 text-heading text-xl font-bold font-mono">{calibration.matchup_count}</div>
            </div>
            <div className="rounded-xl border border-edge bg-raised p-4">
              <div className="text-muted text-[10px] font-mono uppercase tracking-wider">Accuracy</div>
              <div className="mt-2 text-heading text-xl font-bold font-mono">
                {(calibration.accuracy * 100).toFixed(1)}%
              </div>
            </div>
            <div className="rounded-xl border border-edge bg-raised p-4">
              <div className="text-muted text-[10px] font-mono uppercase tracking-wider">Both Bad Rate</div>
              <div className="mt-2 text-heading text-xl font-bold font-mono">
                {(calibration.gate_diagnostics.both_bad_rate * 100).toFixed(1)}%
              </div>
            </div>
          </div>

          <div className="bg-raised border border-edge rounded-xl p-5">
            <h3 className="text-heading text-sm font-semibold mb-4">Weight Comparison</h3>
            <p className="text-subtle text-xs mb-4">
              Current (hardcoded) weights vs learned weights from arena data.
            </p>
            <div className="space-y-4">
              {allMetrics.map((metric) => (
                <WeightBar
                  key={metric}
                  label={METRIC_LABELS[metric] ?? metric}
                  current={VALIDATION_RANKING_WEIGHTS[metric] ?? 0}
                  learned={calibration.learned_weights[metric] ?? 0}
                  maxVal={maxWeight}
                />
              ))}
            </div>
          </div>

          <div className="bg-raised border border-edge rounded-xl p-5">
            <h3 className="text-heading text-sm font-semibold mb-4">Per-Metric Shifts</h3>
            <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-3">
              {Object.entries(calibration.weight_shifts).map(([metric, shift]) => (
                <div key={metric} className="rounded-lg border border-edge bg-surface px-3 py-2.5 flex items-center justify-between">
                  <span className="text-primary text-sm">{METRIC_LABELS[metric] ?? metric}</span>
                  <span className={`font-mono text-sm font-bold ${getShiftColor(shift)}`}>
                    {getShiftArrow(shift)} {shift > 0 ? '+' : ''}{shift.toFixed(3)}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {calibration.gate_diagnostics.suggested_gate_changes.length > 0 && (
            <div className="bg-raised border border-edge rounded-xl p-5">
              <h3 className="text-heading text-sm font-semibold mb-4">Gate Suggestions</h3>
              <div className="space-y-2">
                {calibration.gate_diagnostics.suggested_gate_changes.map((change) => (
                  <div key={change.metric} className="rounded-lg border border-edge bg-surface px-3 py-2.5 flex items-center justify-between">
                    <div>
                      <span className="text-primary text-sm font-medium">{METRIC_LABELS[change.metric] ?? change.metric}</span>
                      <span className="text-muted text-xs ml-2">
                        {change.direction === 'tighten' ? 'Tighten' : 'Loosen'} gate
                      </span>
                    </div>
                    <div className="text-muted font-mono text-xs">
                      {change.current.toFixed(2)} {'\u2192'} {change.suggested.toFixed(2)}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="flex items-center gap-3">
            <button
              onClick={handleApply}
              disabled={applying || !canApply}
              className="bg-accent hover:bg-accent-light text-void font-semibold text-sm px-6 py-2.5 rounded-lg disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
              type="button"
            >
              {applying ? 'Applying...' : 'Apply Calibrated Weights'}
            </button>
            {!canApply && calibration.confidence !== 'insufficient' && (
              <span className="text-muted text-xs">
                Requires at least &quot;calibrated&quot; confidence (80+ matchups).
              </span>
            )}
          </div>
        </>
      )}
    </div>
  )
}
