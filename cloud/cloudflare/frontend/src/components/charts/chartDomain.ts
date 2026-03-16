const NICE_STEPS = [0.01, 0.02, 0.025, 0.05, 0.1, 0.2, 0.25, 0.5, 1, 2, 5, 10, 20, 50, 100]

function niceStep(span: number, maxTicks: number): number {
  for (const step of NICE_STEPS) {
    if (span / step <= maxTicks) return step
  }
  return NICE_STEPS[NICE_STEPS.length - 1]
}

function floorTo(value: number, step: number): number {
  return Math.floor(value / step) * step
}

function ceilTo(value: number, step: number): number {
  return Math.ceil(value / step) * step
}

export interface NiceDomainOptions {
  paddingRatio?: number
  minPadding?: number
  clampMin?: number
  clampMax?: number
  tickCount?: number
}

export interface NiceDomainResult {
  min: number
  max: number
  step: number
  ticks: number[]
}

/**
 * Padded, nice-number-snapped Y-axis domain from raw data values.
 * Handles: empty arrays, identical values, tight clusters, bounded metrics.
 * Clamp options prevent domain from exceeding valid ranges (e.g. scores 0-1).
 */
export function computeNiceDomain(
  values: number[],
  options: NiceDomainOptions = {},
): NiceDomainResult {
  const {
    paddingRatio = 0.1,
    minPadding = 0.05,
    clampMin,
    clampMax,
    tickCount = 4,
  } = options

  if (values.length === 0) {
    const lo = clampMin ?? 0
    const hi = clampMax ?? 1
    const step = niceStep(hi - lo, tickCount)
    return { min: lo, max: hi, step, ticks: buildTicks(lo, hi, step) }
  }

  const rawMin = Math.min(...values)
  const rawMax = Math.max(...values)
  const rawSpan = rawMax - rawMin
  const padding = Math.max(rawSpan * paddingRatio, minPadding)

  let lo = rawMin - padding
  let hi = rawMax + padding

  if (clampMin !== undefined) lo = Math.max(lo, clampMin)
  if (clampMax !== undefined) hi = Math.min(hi, clampMax)

  if (hi <= lo) {
    hi = lo + (minPadding * 2 || 0.1)
    if (clampMax !== undefined) hi = Math.min(hi, clampMax)
  }

  const span = hi - lo
  const step = niceStep(span, tickCount)

  lo = floorTo(lo, step)
  hi = ceilTo(hi, step)

  if (clampMin !== undefined) lo = Math.max(lo, clampMin)
  if (clampMax !== undefined) hi = Math.min(hi, clampMax)

  return { min: lo, max: hi, step, ticks: buildTicks(lo, hi, step) }
}

function buildTicks(min: number, max: number, step: number): number[] {
  const ticks: number[] = []
  let current = min
  for (let i = 0; i <= 20; i++) {
    ticks.push(roundFloat(current))
    current += step
    if (current > max + step * 0.001) break
  }
  const lastTick = ticks.length > 0 ? ticks[ticks.length - 1] : min
  const gap = roundFloat(max) - lastTick
  if (gap > step * 0.3) {
    ticks.push(roundFloat(max))
  }
  return ticks
}

function roundFloat(value: number): number {
  return Math.round(value * 1_000_000) / 1_000_000
}

export function precisionForStep(step: number): number {
  if (!Number.isFinite(step) || step <= 0 || step >= 1) return 0
  const s = step.toString()
  const dot = s.indexOf('.')
  return dot === -1 ? 0 : s.length - dot - 1
}
