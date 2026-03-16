export function formatScore(score: number | null | undefined): string {
  if (score == null || !Number.isFinite(score)) return '—'
  return score.toFixed(3)
}

export function scoreColor(score: number | null | undefined): string {
  if (score == null || !Number.isFinite(score)) return 'text-muted'
  if (score >= 0.85) return 'text-accent'
  if (score >= 0.70) return 'text-warning'
  return 'text-error'
}

export function scoreBgColor(score: number | null | undefined): string {
  if (score == null || !Number.isFinite(score)) return 'bg-surface'
  if (score >= 0.85) return 'bg-accent-dim'
  if (score >= 0.70) return 'bg-warning-dim'
  return 'bg-error-dim'
}
