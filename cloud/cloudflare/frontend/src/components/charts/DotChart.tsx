import { useMemo } from 'react'
import { computeNiceDomain, precisionForStep, type NiceDomainOptions } from './chartDomain'

interface DotChartProps {
  data: { label: string; value: number; color?: string }[]
  height?: number
  yLabel?: string
  domainOptions?: NiceDomainOptions
}

function formatDotValue(value: number, precision?: number): string {
  if (!Number.isFinite(value)) return '0'
  if (precision !== undefined) return value.toFixed(precision)
  if (Math.abs(value) >= 10 || Number.isInteger(value)) return value.toFixed(0)
  return value.toFixed(2)
}

export function DotChart({ data, height = 240, yLabel, domainOptions }: DotChartProps) {
  const width = 640
  const margin = { top: 18, right: 12, bottom: 42, left: 40 }
  const chartWidth = width - margin.left - margin.right
  const chartHeight = height - margin.top - margin.bottom

  const domain = useMemo(() => {
    const values = data.map((d) => d.value)
    return computeNiceDomain(values, domainOptions)
  }, [data, domainOptions])

  const ySpan = domain.max - domain.min || 1

  const getY = (value: number): number => {
    const normalized = (value - domain.min) / ySpan
    return margin.top + chartHeight - normalized * chartHeight
  }

  return (
    <div className="w-full rounded-lg border border-edge bg-surface p-3">
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-auto" role="img" aria-label="Dot chart">
        {(() => {
          const precision = domain.step > 0 ? precisionForStep(domain.step) : undefined
          return domain.ticks.map((value) => {
            const y = getY(value)
            return (
              <g key={`grid-${value.toFixed(4)}`}>
                <line x1={margin.left} y1={y} x2={width - margin.right} y2={y} stroke="var(--color-edge)" strokeWidth="1" />
                <text x={margin.left - 6} y={y + 4} textAnchor="end" fontSize="10" fill="var(--color-muted)">
                  {formatDotValue(value, precision)}
                </text>
              </g>
            )
          })
        })()}

        <line x1={margin.left} y1={margin.top + chartHeight} x2={width - margin.right} y2={margin.top + chartHeight} stroke="var(--color-edge)" />
        <line x1={margin.left} y1={margin.top} x2={margin.left} y2={margin.top + chartHeight} stroke="var(--color-edge)" />

        {data.map((item, index) => {
          const slotWidth = chartWidth / Math.max(1, data.length)
          const cx = margin.left + index * slotWidth + slotWidth / 2
          const cy = getY(item.value)
          const dotColor = item.color ?? 'var(--color-accent)'
          return (
            <g key={`${item.label}-${item.value.toString()}`}>
              <line
                x1={cx}
                y1={margin.top + chartHeight}
                x2={cx}
                y2={cy}
                stroke={dotColor}
                strokeWidth="1.5"
                strokeDasharray="3 2"
                opacity="0.35"
              />
              <circle cx={cx} cy={cy} r="5" fill={dotColor} fillOpacity="0.9" />
              <text x={cx} y={cy - 10} textAnchor="middle" fontSize="10" fontWeight="600" fill="var(--color-primary)">
                {formatDotValue(item.value)}
              </text>
              <text
                x={cx}
                y={height - 16}
                textAnchor="middle"
                fontSize="10"
                fill="var(--color-muted)"
              >
                {item.label}
              </text>
            </g>
          )
        })}

        {yLabel && (
          <text x={12} y={12} fontSize="10" fill="var(--color-muted)">
            {yLabel}
          </text>
        )}
      </svg>
    </div>
  )
}

export type { DotChartProps }
