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

export function scoreBandLabel(score: number | null | undefined): string {
  if (score == null || !Number.isFinite(score)) return '\u2014'
  if (score >= 0.90) return 'Excellent'
  if (score >= 0.85) return 'Strong'
  if (score >= 0.70) return 'Borderline'
  return 'Weak'
}

export function scoreBandStyle(score: number | null | undefined): { text: string; bg: string } {
  if (score == null || !Number.isFinite(score)) return { text: 'text-muted', bg: 'bg-surface' }
  if (score >= 0.90) return { text: 'text-accent', bg: 'bg-accent-dim' }
  if (score >= 0.85) return { text: 'text-accent/80', bg: 'bg-accent-dim/60' }
  if (score >= 0.70) return { text: 'text-warning', bg: 'bg-warning-dim' }
  return { text: 'text-error', bg: 'bg-error-dim' }
}
