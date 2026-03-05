import { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router'
import {
  fetchVoice,
  deleteVoice,
  getSpeechGenerationStatus,
  startSpeechGenerationAsync,
  type Voice,
  type VoiceSettings,
  DEFAULT_VOICE_SETTINGS,
  formatDate,
} from '../lib/api'
import { AudioPlayer } from '../components/AudioPlayer'

export function VoiceDetail() {
  const { voiceId = '' } = useParams()
  const navigate = useNavigate()

  const [voice, setVoice] = useState<Voice | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  // Settings
  const [settings, setSettings] = useState<VoiceSettings>(DEFAULT_VOICE_SETTINGS)

  // Quick generate
  const [text, setText] = useState('')
  const [generating, setGenerating] = useState(false)
  const [generateStatus, setGenerateStatus] = useState('')
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null)
  const [genError, setGenError] = useState('')

  // Delete
  const [confirmDelete, setConfirmDelete] = useState(false)
  const [deleting, setDeleting] = useState(false)

  useEffect(() => {
    if (!voiceId) return
    let cancelled = false

    async function load() {
      try {
        const data = await fetchVoice(voiceId)
        if (!cancelled) {
          setVoice(data)
          setSettings(data.settings ?? DEFAULT_VOICE_SETTINGS)
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : 'Failed to load voice')
        }
      } finally {
        if (!cancelled) setLoading(false)
      }
    }

    load()
    return () => { cancelled = true }
  }, [voiceId])

  async function handleGenerate() {
    if (!voiceId || !text.trim()) return

    setGenerating(true)
    setGenerateStatus('Queued...')
    setGenError('')
    setAudioBlob(null)

    try {
      const asyncJob = await startSpeechGenerationAsync(voiceId, text.trim(), settings)
      let completed = false
      let attempts = 0
      while (attempts < 180) {
        attempts += 1
        const status = await getSpeechGenerationStatus(asyncJob.job_id)
        if (status.status === 'COMPLETED' && status.audio) {
          const bytes = Uint8Array.from(atob(status.audio), (c) => c.charCodeAt(0))
          setAudioBlob(new Blob([bytes], { type: 'audio/wav' }))
          setGenerateStatus('Completed')
          completed = true
          break
        }
        if (status.status === 'FAILED') {
          throw new Error(status.error || 'Generation failed')
        }
        setGenerateStatus(status.status)
        await new Promise((resolve) => setTimeout(resolve, 1000))
      }
      if (!completed) {
        throw new Error('Generation timed out. Please try again.')
      }
    } catch (err) {
      setGenError(err instanceof Error ? err.message : 'Generation failed')
      setGenerateStatus('')
    } finally {
      setGenerating(false)
    }
  }

  async function handleDelete() {
    if (!voiceId) return
    setDeleting(true)
    try {
      await deleteVoice(voiceId)
      navigate('/voices')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete voice')
      setDeleting(false)
      setConfirmDelete(false)
    }
  }

  function updateSetting(key: keyof VoiceSettings, value: number) {
    setSettings((prev) => ({ ...prev, [key]: value }))
  }

  // Loading state
  if (loading) {
    return (
      <div className="space-y-6 animate-pulse">
        <div className="h-8 bg-raised rounded w-1/3" />
        <div className="h-4 bg-raised rounded w-1/2" />
        <div className="grid lg:grid-cols-2 gap-6">
          <div className="h-64 bg-raised rounded-xl" />
          <div className="h-64 bg-raised rounded-xl" />
        </div>
      </div>
    )
  }

  // Error state
  if (error && !voice) {
    return (
      <div className="text-center py-16">
        <div className="text-error text-sm mb-4">{error}</div>
        <button
          onClick={() => navigate('/voices')}
          className="text-accent text-sm hover:text-accent-light"
          type="button"
        >
          ← Back to Voices
        </button>
      </div>
    )
  }

  if (!voice) return null

  const statusStyles: Record<string, string> = {
    ready: 'bg-accent-dim text-accent',
    training: 'bg-warning-dim text-warning',
    created: 'bg-raised text-muted',
  }

  return (
    <div className="space-y-8">
      {/* Back link */}
      <button
        onClick={() => navigate('/voices')}
        className="text-subtle text-sm hover:text-accent transition-colors inline-flex items-center gap-1"
        type="button"
      >
        <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
          <polyline points="15 18 9 12 15 6" />
        </svg>
        Back to Voices
      </button>

      {/* Voice Header */}
      <div className="flex items-start justify-between gap-4">
        <div>
          <div className="flex items-center gap-3 mb-2">
            <h1 className="text-heading text-2xl font-bold">{voice.name}</h1>
            <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-[10px] font-mono uppercase tracking-wider ${statusStyles[voice.status] ?? statusStyles.created}`}>
              {voice.status}
            </span>
          </div>
          {voice.description && (
            <p className="text-subtle text-sm max-w-xl">{voice.description}</p>
          )}
          <div className="flex items-center gap-4 mt-3 text-muted text-xs font-mono">
            <span>Model: {voice.model_size || 'base'}</span>
            <span>Created: {formatDate(voice.created_at)}</span>
            <span>ID: {voice.voice_id.slice(0, 12)}...</span>
          </div>
        </div>
      </div>

      <div className="grid lg:grid-cols-2 gap-6">
        {/* Voice Settings */}
        <div className="bg-raised border border-edge rounded-xl p-5">
          <h2 className="text-heading font-semibold text-sm mb-5">Voice Settings</h2>
          <div className="space-y-5">
            <SettingSlider
              label="Stability"
              value={settings.stability}
              onChange={(v) => updateSetting('stability', v)}
              min={0}
              max={1}
              step={0.01}
              leftLabel="Variable"
              rightLabel="Stable"
            />
            <SettingSlider
              label="Similarity Boost"
              value={settings.similarity_boost}
              onChange={(v) => updateSetting('similarity_boost', v)}
              min={0}
              max={1}
              step={0.01}
              leftLabel="Low"
              rightLabel="High"
            />
            <SettingSlider
              label="Style"
              value={settings.style}
              onChange={(v) => updateSetting('style', v)}
              min={0}
              max={1}
              step={0.01}
              leftLabel="None"
              rightLabel="Exaggerated"
            />
            <SettingSlider
              label="Speed"
              value={settings.speed}
              onChange={(v) => updateSetting('speed', v)}
              min={0.5}
              max={2.0}
              step={0.05}
              leftLabel="0.5x"
              rightLabel="2.0x"
            />
          </div>
        </div>

        {/* Quick Generate */}
        <div className="bg-raised border border-edge rounded-xl p-5">
          <h2 className="text-heading font-semibold text-sm mb-4">Generate Speech</h2>

          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Enter text to generate speech..."
            rows={4}
            className="w-full bg-surface border border-edge rounded-lg px-3 py-2.5 text-sm text-primary placeholder:text-muted focus:border-accent transition-colors resize-none mb-4"
          />

          <button
            onClick={handleGenerate}
            disabled={!text.trim() || generating || voice.status !== 'ready'}
            className="w-full bg-accent hover:bg-accent-light text-void font-semibold text-sm py-2.5 rounded-lg disabled:opacity-40 disabled:cursor-not-allowed transition-colors mb-4"
            type="button"
          >
            {generating ? (generateStatus ? `Generating (${generateStatus})...` : 'Generating...') : 'Generate'}
          </button>

          {genError && (
            <div className="bg-error-dim border border-error/20 rounded-lg px-3 py-2 text-error text-xs mb-4">
              {genError}
            </div>
          )}

          <AudioPlayer blob={audioBlob ?? undefined} generating={generating} />
        </div>
      </div>

      {/* Danger Zone */}
      <div className="bg-raised border border-error/20 rounded-xl p-5">
        <h2 className="text-error font-semibold text-sm mb-2">Danger Zone</h2>
        <p className="text-subtle text-xs mb-4">
          Permanently delete this voice and all associated data. This action cannot be undone.
        </p>

        {confirmDelete ? (
          <div className="flex items-center gap-3">
            <span className="text-error text-xs font-medium">Are you sure?</span>
            <button
              onClick={handleDelete}
              disabled={deleting}
              className="bg-error hover:bg-red-500 text-white text-xs font-semibold px-4 py-2 rounded-lg disabled:opacity-50 transition-colors"
              type="button"
            >
              {deleting ? 'Deleting...' : 'Yes, Delete'}
            </button>
            <button
              onClick={() => setConfirmDelete(false)}
              className="text-subtle text-xs hover:text-primary"
              type="button"
            >
              Cancel
            </button>
          </div>
        ) : (
          <button
            onClick={() => setConfirmDelete(true)}
            className="border border-error/30 text-error text-xs font-medium px-4 py-2 rounded-lg hover:bg-error-dim transition-colors"
            type="button"
          >
            Delete Voice
          </button>
        )}
      </div>
    </div>
  )
}

// ── Setting Slider ─────────────────────────────────────────────────────────────

function SettingSlider({
  label,
  value,
  onChange,
  min,
  max,
  step,
  leftLabel,
  rightLabel,
}: {
  label: string
  value: number
  onChange: (v: number) => void
  min: number
  max: number
  step: number
  leftLabel: string
  rightLabel: string
}) {
  return (
    <div>
      <div className="flex items-center justify-between mb-2">
        <label className="text-subtle text-xs font-medium">{label}</label>
        <span className="text-accent text-xs font-mono tabular-nums">{value.toFixed(2)}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full"
      />
      <div className="flex justify-between mt-1">
        <span className="text-muted text-[10px] font-mono">{leftLabel}</span>
        <span className="text-muted text-[10px] font-mono">{rightLabel}</span>
      </div>
    </div>
  )
}
