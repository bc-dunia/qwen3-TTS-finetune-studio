import { useState, useEffect, useRef, useCallback } from 'react'
import { formatDuration } from '../lib/api'

interface AudioPlayerProps {
  src?: string
  blob?: Blob
  generating?: boolean
}

export function AudioPlayer({ src, blob, generating = false }: AudioPlayerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const audioRef = useRef<HTMLAudioElement>(null)
  const animRef = useRef<number>(0)
  const startTimeRef = useRef<number>(0)

  const [audioUrl, setAudioUrl] = useState<string>('')
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [waveformBars, setWaveformBars] = useState<number[]>([])

  // Create/revoke object URL from blob
  useEffect(() => {
    if (blob) {
      const url = URL.createObjectURL(blob)
      setAudioUrl(url)
      return () => URL.revokeObjectURL(url)
    }
    if (src) {
      setAudioUrl(src)
    }
    return undefined
  }, [blob, src])

  // Decode audio buffer for waveform data
  useEffect(() => {
    if (!blob && !src) {
      setWaveformBars([])
      return
    }

    let cancelled = false

    async function decode() {
      try {
        let buffer: ArrayBuffer
        if (blob) {
          buffer = await blob.arrayBuffer()
        } else if (src) {
          const resp = await fetch(src)
          buffer = await resp.arrayBuffer()
        } else {
          return
        }

        const ctx = new AudioContext()
        const audioBuffer = await ctx.decodeAudioData(buffer)
        await ctx.close()

        if (cancelled) return

        const channelData = audioBuffer.getChannelData(0)
        const barCount = 80
        const samplesPerBar = Math.floor(channelData.length / barCount)
        const bars: number[] = []

        for (let i = 0; i < barCount; i++) {
          let peak = 0
          const offset = i * samplesPerBar
          for (let j = 0; j < samplesPerBar; j++) {
            const abs = Math.abs(channelData[offset + j])
            if (abs > peak) peak = abs
          }
          bars.push(peak)
        }

        // Normalize
        const max = Math.max(...bars, 0.01)
        setWaveformBars(bars.map((v) => v / max))
      } catch {
        // Decoding failed — show flat bars
        setWaveformBars(Array.from({ length: 80 }, () => 0.1))
      }
    }

    decode()
    return () => { cancelled = true }
  }, [blob, src])

  // Draw waveform
  const drawWaveform = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const dpr = window.devicePixelRatio || 1
    const rect = canvas.getBoundingClientRect()
    canvas.width = rect.width * dpr
    canvas.height = rect.height * dpr
    ctx.scale(dpr, dpr)

    const w = rect.width
    const h = rect.height

    ctx.clearRect(0, 0, w, h)

    if (generating) {
      drawGeneratingAnimation(ctx, w, h)
      return
    }

    if (waveformBars.length === 0) {
      drawEmptyState(ctx, w, h)
      return
    }

    const barCount = waveformBars.length
    const gap = 2
    const barWidth = (w - gap * barCount) / barCount
    const progress = duration > 0 ? currentTime / duration : 0

    for (let i = 0; i < barCount; i++) {
      const amplitude = waveformBars[i]
      const barHeight = Math.max(2, amplitude * h * 0.75)
      const x = i * (barWidth + gap)
      const y = (h - barHeight) / 2
      const isPlayed = i / barCount <= progress

      if (isPlayed) {
        ctx.fillStyle = '#10b981'
        ctx.shadowColor = '#10b981'
        ctx.shadowBlur = 4
      } else {
        ctx.fillStyle = '#2a2a38'
        ctx.shadowColor = 'transparent'
        ctx.shadowBlur = 0
      }

      roundedRect(ctx, x, y, barWidth, barHeight, barWidth / 2)
    }

    ctx.shadowBlur = 0
  }, [waveformBars, currentTime, duration, generating])

  // Generating animation loop
  const drawGeneratingAnimation = useCallback(
    (ctx: CanvasRenderingContext2D, w: number, h: number) => {
      const barCount = 60
      const gap = 2
      const barWidth = (w - gap * barCount) / barCount
      const elapsed = (performance.now() - startTimeRef.current) * 0.001

      for (let i = 0; i < barCount; i++) {
        const x = i * (barWidth + gap)
        const norm = i / barCount

        const wave1 = Math.sin(norm * Math.PI * 4 + elapsed * 3) * 0.3
        const wave2 = Math.sin(norm * Math.PI * 7 + elapsed * 2.3) * 0.2
        const wave3 = Math.cos(norm * Math.PI * 3 + elapsed * 4.1) * 0.15
        const amplitude = 0.12 + Math.abs(wave1 + wave2 + wave3)

        const barHeight = Math.max(2, amplitude * h * 0.8)
        const y = (h - barHeight) / 2

        const alpha = 0.4 + amplitude * 0.6
        ctx.fillStyle = `rgba(16, 185, 129, ${alpha})`
        ctx.shadowColor = '#10b981'
        ctx.shadowBlur = 6 * amplitude

        roundedRect(ctx, x, y, barWidth, barHeight, barWidth / 2)
      }

      ctx.shadowBlur = 0
    },
    [],
  )

  // Animation frame loop
  useEffect(() => {
    if (!generating && !isPlaying) {
      drawWaveform()
      return
    }

    startTimeRef.current = performance.now()

    function tick() {
      drawWaveform()
      animRef.current = requestAnimationFrame(tick)
    }

    animRef.current = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(animRef.current)
  }, [generating, isPlaying, drawWaveform])

  // Redraw on data change
  useEffect(() => {
    if (!generating && !isPlaying) {
      drawWaveform()
    }
  }, [waveformBars, currentTime, generating, isPlaying, drawWaveform])

  // Audio event handlers
  function handleTimeUpdate() {
    if (audioRef.current) {
      setCurrentTime(audioRef.current.currentTime)
    }
  }

  function handleLoadedMetadata() {
    if (audioRef.current) {
      setDuration(audioRef.current.duration)
    }
  }

  function handleEnded() {
    setIsPlaying(false)
    setCurrentTime(0)
  }

  function togglePlay() {
    const audio = audioRef.current
    if (!audio || !audioUrl) return

    if (isPlaying) {
      audio.pause()
      setIsPlaying(false)
    } else {
      audio.play().catch(() => {
        // Autoplay blocked
      })
      setIsPlaying(true)
    }
  }

  function handleCanvasClick(e: React.MouseEvent<HTMLCanvasElement>) {
    const audio = audioRef.current
    if (!audio || !duration) return

    const rect = e.currentTarget.getBoundingClientRect()
    const x = e.clientX - rect.left
    const progress = x / rect.width
    audio.currentTime = progress * duration
    setCurrentTime(audio.currentTime)
  }

  const hasAudio = Boolean(audioUrl) && !generating

  return (
    <div className="bg-raised border border-edge rounded-xl p-4">
      {/* Waveform canvas */}
      <canvas
        ref={canvasRef}
        className={`w-full h-20 rounded-lg ${hasAudio ? 'cursor-pointer' : 'cursor-default'}`}
        onClick={hasAudio ? handleCanvasClick : undefined}
      />

      {/* Controls */}
      <div className="flex items-center gap-3 mt-3">
        <button
          onClick={togglePlay}
          disabled={!hasAudio}
          className="w-9 h-9 rounded-full bg-accent/20 text-accent flex items-center justify-center hover:bg-accent/30 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
          type="button"
          aria-label={isPlaying ? 'Pause' : 'Play'}
        >
          {isPlaying ? (
            <svg className="w-4 h-4" viewBox="0 0 24 24" fill="currentColor">
              <rect x="6" y="4" width="4" height="16" rx="1" />
              <rect x="14" y="4" width="4" height="16" rx="1" />
            </svg>
          ) : (
            <svg className="w-4 h-4 ml-0.5" viewBox="0 0 24 24" fill="currentColor">
              <path d="M8 5.14v13.72a1 1 0 0 0 1.5.86l11.24-7.36a1 1 0 0 0 0-1.72L9.5 4.28A1 1 0 0 0 8 5.14z" />
            </svg>
          )}
        </button>

        {/* Download */}
        <a
          href={audioUrl || undefined}
          download="generated_speech.wav"
          className={`w-8 h-8 rounded-full flex items-center justify-center transition-colors ${
            hasAudio
              ? 'bg-elevated text-muted hover:bg-accent/20 hover:text-accent cursor-pointer'
              : 'bg-elevated text-muted opacity-30 cursor-not-allowed pointer-events-none'
          }`}
          aria-label="Download audio"
          tabIndex={hasAudio ? 0 : -1}
          onClick={(e) => { if (!hasAudio) e.preventDefault() }}
        >
          <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
            <polyline points="7 10 12 15 17 10" />
            <line x1="12" y1="15" x2="12" y2="3" />
          </svg>
        </a>

        {/* Progress bar */}
        <div className="flex-1 h-1 bg-edge rounded-full overflow-hidden">
          <div
            className="h-full bg-accent rounded-full transition-[width] duration-100"
            style={{ width: duration > 0 ? `${(currentTime / duration) * 100}%` : '0%' }}
          />
        </div>

        {/* Time */}
        <span className="text-muted text-xs font-mono tabular-nums shrink-0">
          {generating ? (
            <span className="text-accent animate-pulse-glow">Generating...</span>
          ) : (
            `${formatDuration(currentTime)} / ${formatDuration(duration)}`
          )}
        </span>
      </div>

      {/* Hidden audio element */}
      {audioUrl && (
        <audio
          ref={audioRef}
          src={audioUrl}
          onTimeUpdate={handleTimeUpdate}
          onLoadedMetadata={handleLoadedMetadata}
          onEnded={handleEnded}
          preload="metadata"
        />
      )}
    </div>
  )
}

// ── Canvas Helpers ─────────────────────────────────────────────────────────────

function roundedRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  r: number,
) {
  const radius = Math.min(r, w / 2, h / 2)
  ctx.beginPath()
  ctx.moveTo(x + radius, y)
  ctx.lineTo(x + w - radius, y)
  ctx.quadraticCurveTo(x + w, y, x + w, y + radius)
  ctx.lineTo(x + w, y + h - radius)
  ctx.quadraticCurveTo(x + w, y + h, x + w - radius, y + h)
  ctx.lineTo(x + radius, y + h)
  ctx.quadraticCurveTo(x, y + h, x, y + h - radius)
  ctx.lineTo(x, y + radius)
  ctx.quadraticCurveTo(x, y, x + radius, y)
  ctx.closePath()
  ctx.fill()
}

function drawEmptyState(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
) {
  const barCount = 60
  const gap = 2
  const barWidth = (w - gap * barCount) / barCount

  for (let i = 0; i < barCount; i++) {
    const x = i * (barWidth + gap)
    const barHeight = 2
    const y = (h - barHeight) / 2

    ctx.fillStyle = '#2a2a38'
    roundedRect(ctx, x, y, barWidth, barHeight, 1)
  }
}
