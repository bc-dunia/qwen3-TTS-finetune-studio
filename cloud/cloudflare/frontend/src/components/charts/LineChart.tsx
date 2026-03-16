import { useMemo, useState } from 'react'

interface LineChartProps {
  data: { label: string; series: { name: string; color: string; values: (number | null)[] }[] }
  height?: number
  yMin?: number
  yMax?: number
  yLabel?: string
  xLabel?: string
}

type Point = { x: number; y: number; value: number; index: number }

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value))
}

function formatTick(value: number): string {
  if (!Number.isFinite(value)) return '0'
  if (Math.abs(value) >= 10 || Number.isInteger(value)) return value.toFixed(0)
  return value.toFixed(2)
}

function seriesPoints(values: (number | null)[], getX: (index: number) => number, getY: (value: number) => number): Point[] {
  const points: Point[] = []
  values.forEach((value, index) => {
    if (typeof value === 'number' && Number.isFinite(value)) {
      points.push({ x: getX(index), y: getY(value), value, index })
    }
  })
  return points
}

function pointsToPath(points: Point[]): string {
  if (points.length === 0) return ''
  return points.map((point, index) => `${index === 0 ? 'M' : 'L'} ${point.x.toFixed(2)} ${point.y.toFixed(2)}`).join(' ')
}

export function LineChart({ data, height = 260, yMin, yMax, yLabel, xLabel }: LineChartProps) {
  const width = 760
  const margin = { top: 14, right: 16, bottom: 34, left: 42 }
  const chartWidth = width - margin.left - margin.right
  const chartHeight = height - margin.top - margin.bottom

  const pointCount = useMemo(() => {
    return Math.max(0, ...data.series.map((series) => series.values.length))
  }, [data.series])

  const [hoverIndex, setHoverIndex] = useState<number | null>(null)

  const numericValues = useMemo(() => {
    return data.series.flatMap((series) => series.values).filter((value): value is number => typeof value === 'number' && Number.isFinite(value))
  }, [data.series])

  const resolvedMin = yMin ?? (numericValues.length > 0 ? Math.min(...numericValues) : 0)
  const resolvedMax = yMax ?? (numericValues.length > 0 ? Math.max(...numericValues) : 1)
  const ySpan = resolvedMax - resolvedMin || 1

  const getX = (index: number): number => {
    if (pointCount <= 1) return margin.left + chartWidth / 2
    return margin.left + (index / (pointCount - 1)) * chartWidth
  }
  const getY = (value: number): number => {
    const normalized = (value - resolvedMin) / ySpan
    return margin.top + chartHeight - normalized * chartHeight
  }

  const hoveredSeriesValues = data.series.map((series) => {
    if (hoverIndex === null) return null
    const value = series.values[hoverIndex]
    if (typeof value !== 'number' || !Number.isFinite(value)) return null
    return { name: series.name, value, color: series.color }
  })

  const tooltipLines = hoveredSeriesValues.filter((line): line is { name: string; value: number; color: string } => line !== null)
  const tooltipWidth = 170
  const tooltipHeight = 20 + tooltipLines.length * 14
  const tooltipX = hoverIndex === null ? 0 : clamp(getX(hoverIndex) + 8, margin.left, width - tooltipWidth - 6)
  const tooltipY = margin.top + 6

  return (
    <div className="w-full rounded-lg border border-edge bg-surface p-3">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="w-full h-auto"
        role="img"
        aria-label="Line chart"
        onMouseMove={(event) => {
          if (pointCount === 0) return
          const rect = event.currentTarget.getBoundingClientRect()
          const relativeX = event.clientX - rect.left
          const x = (relativeX / rect.width) * width
          const index = Math.round(((x - margin.left) / chartWidth) * Math.max(1, pointCount - 1))
          setHoverIndex(clamp(index, 0, Math.max(0, pointCount - 1)))
        }}
        onMouseLeave={() => setHoverIndex(null)}
      >
        {Array.from({ length: 5 }).map((_, tickIndex) => {
          const ratio = tickIndex / 4
          const y = margin.top + ratio * chartHeight
          const value = resolvedMax - ratio * ySpan
          return (
            <g key={`y-${value.toFixed(4)}`}>
              <line x1={margin.left} y1={y} x2={width - margin.right} y2={y} stroke="var(--color-edge)" strokeWidth="1" />
              <text x={margin.left - 6} y={y + 4} textAnchor="end" fontSize="10" fill="var(--color-muted)">
                {formatTick(value)}
              </text>
            </g>
          )
        })}

        <line x1={margin.left} y1={margin.top + chartHeight} x2={width - margin.right} y2={margin.top + chartHeight} stroke="var(--color-edge)" />
        <line x1={margin.left} y1={margin.top} x2={margin.left} y2={margin.top + chartHeight} stroke="var(--color-edge)" />

        {data.series.map((series) => {
          const points = seriesPoints(series.values, getX, getY)
          if (points.length === 0) {
            return null
          }
          const linePath = pointsToPath(points)
          const areaPath = `${linePath} L ${points[points.length - 1].x.toFixed(2)} ${(margin.top + chartHeight).toFixed(2)} L ${points[0].x.toFixed(2)} ${(margin.top + chartHeight).toFixed(2)} Z`
          return (
            <g key={series.name}>
              <path d={areaPath} fill={series.color} fillOpacity="0.1" />
              <path d={linePath} fill="none" stroke={series.color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
              {hoverIndex !== null && typeof series.values[hoverIndex] === 'number' && (
                <circle cx={getX(hoverIndex)} cy={getY(series.values[hoverIndex] as number)} r="3" fill={series.color} />
              )}
            </g>
          )
        })}

        {hoverIndex !== null && (
          <line
            x1={getX(hoverIndex)}
            y1={margin.top}
            x2={getX(hoverIndex)}
            y2={margin.top + chartHeight}
            stroke="var(--color-muted)"
            strokeDasharray="3 3"
          />
        )}

        <text x={margin.left} y={height - 8} fontSize="10" fill="var(--color-muted)">{pointCount > 0 ? '1' : '0'}</text>
        <text x={margin.left + chartWidth / 2} y={height - 8} textAnchor="middle" fontSize="10" fill="var(--color-muted)">
          {data.label}
        </text>
        <text x={width - margin.right} y={height - 8} textAnchor="end" fontSize="10" fill="var(--color-muted)">
          {pointCount > 0 ? String(pointCount) : '0'}
        </text>

        {yLabel && <text x={6} y={12} fontSize="10" fill="var(--color-muted)">{yLabel}</text>}
        {xLabel && <text x={width - margin.right} y={12} textAnchor="end" fontSize="10" fill="var(--color-muted)">{xLabel}</text>}

        {hoverIndex !== null && tooltipLines.length > 0 && (
          <g>
            <rect x={tooltipX} y={tooltipY} width={tooltipWidth} height={tooltipHeight} rx="6" fill="var(--color-raised)" stroke="var(--color-edge)" />
            <text x={tooltipX + 8} y={tooltipY + 12} fontSize="10" fill="var(--color-muted)">
              Attempt #{hoverIndex + 1}
            </text>
            {tooltipLines.map((line, index) => (
              <text key={`${line.name}-tip`} x={tooltipX + 8} y={tooltipY + 26 + index * 14} fontSize="10" fill={line.color}>
                {line.name}: {line.value.toFixed(3)}
              </text>
            ))}
          </g>
        )}
      </svg>
    </div>
  )
}

export type { LineChartProps }
