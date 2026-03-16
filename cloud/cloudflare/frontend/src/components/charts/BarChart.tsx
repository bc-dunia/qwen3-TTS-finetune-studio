interface BarChartProps {
  data: { label: string; value: number; color?: string }[]
  height?: number
  yLabel?: string
}

function formatBarValue(value: number): string {
  if (!Number.isFinite(value)) return '0'
  if (Math.abs(value) >= 10 || Number.isInteger(value)) return value.toFixed(0)
  return value.toFixed(2)
}

export function BarChart({ data, height = 240, yLabel }: BarChartProps) {
  const width = 640
  const margin = { top: 18, right: 12, bottom: 42, left: 40 }
  const chartWidth = width - margin.left - margin.right
  const chartHeight = height - margin.top - margin.bottom
  const maxValue = Math.max(1, ...data.map((item) => item.value))
  const tickCount = 4

  const getY = (value: number): number => margin.top + chartHeight - (value / maxValue) * chartHeight

  return (
    <div className="w-full rounded-lg border border-edge bg-surface p-3">
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-auto" role="img" aria-label="Bar chart">
        {Array.from({ length: tickCount + 1 }).map((_, tickIndex) => {
          const ratio = tickIndex / tickCount
          const y = margin.top + ratio * chartHeight
          const value = maxValue * (1 - ratio)
          return (
            <g key={`grid-${value.toFixed(4)}`}>
              <line x1={margin.left} y1={y} x2={width - margin.right} y2={y} stroke="var(--color-edge)" strokeWidth="1" />
              <text x={margin.left - 6} y={y + 4} textAnchor="end" fontSize="10" fill="var(--color-muted)">
                {formatBarValue(value)}
              </text>
            </g>
          )
        })}

        {data.map((item, index) => {
          const slotWidth = chartWidth / Math.max(1, data.length)
          const barWidth = Math.min(34, slotWidth * 0.7)
          const x = margin.left + index * slotWidth + (slotWidth - barWidth) / 2
          const y = getY(item.value)
          const barHeight = margin.top + chartHeight - y
          return (
            <g key={`${item.label}-${item.value.toString()}`}>
              <rect
                x={x}
                y={y}
                width={barWidth}
                height={Math.max(0, barHeight)}
                rx="3"
                fill={item.color ?? 'var(--color-accent)'}
                fillOpacity="0.85"
              />
              <text x={x + barWidth / 2} y={y - 6} textAnchor="middle" fontSize="10" fill="var(--color-primary)">
                {formatBarValue(item.value)}
              </text>
              <text
                x={x + barWidth / 2}
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

export type { BarChartProps }
