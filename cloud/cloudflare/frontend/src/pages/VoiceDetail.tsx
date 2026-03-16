import { useState, useEffect, useRef } from 'react'
import { useParams, useNavigate, useOutletContext } from 'react-router'
import {
  deleteVoice,
  pollSpeechGeneration,
  startSpeechGenerationAsync,
  type SpeechGenerationOptions,
  type Voice,
  type VoiceSettings,
  DEFAULT_VOICE_SETTINGS,
  formatDate,
} from '../lib/api'
import { AudioPlayer } from '../components/AudioPlayer'

function normalizeVoiceSettings(value: Partial<VoiceSettings> | null | undefined): VoiceSettings {
  const src = value ?? {}
  return {
    stability: typeof src.stability === 'number' ? src.stability : DEFAULT_VOICE_SETTINGS.stability,
    similarity_boost: typeof src.similarity_boost === 'number' ? src.similarity_boost : DEFAULT_VOICE_SETTINGS.similarity_boost,
    style: typeof src.style === 'number' ? src.style : DEFAULT_VOICE_SETTINGS.style,
    speed: typeof src.speed === 'number' ? src.speed : DEFAULT_VOICE_SETTINGS.speed,
  }
}

export function VoiceDetail() {
  const { voiceId = '' } = useParams()
  const navigate = useNavigate()
  const { voice, loading } = useOutletContext<{ voice: Voice | null; loading: boolean }>()

  const [settings, setSettings] = useState<VoiceSettings>(DEFAULT_VOICE_SETTINGS)
  const lastInitVoiceId = useRef('')

  const [text, setText] = useState('')
  const [stylePrompt, setStylePrompt] = useState('')
  const [instruct, setInstruct] = useState('')
  const [generating, setGenerating] = useState(false)
  const [generateStatus, setGenerateStatus] = useState('')
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null)
  const [genError, setGenError] = useState('')

  const [confirmDelete, setConfirmDelete] = useState(false)
  const [deleting, setDeleting] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    if (!voice || voice.voice_id === lastInitVoiceId.current) return
    setSettings(normalizeVoiceSettings(voice.settings))
    lastInitVoiceId.current = voice.voice_id
  }, [voice])

  async function handleGenerate() {
    if (!voiceId || !text.trim()) return

    setGenerating(true)
    setGenerateStatus('Queued...')
    setGenError('')
    setAudioBlob(null)

    try {
      const promptOptions: SpeechGenerationOptions = {
        stylePrompt: stylePrompt.trim() || undefined,
        instruct: instruct.trim() || undefined,
      }
      const asyncJob = await startSpeechGenerationAsync(voiceId, text.trim(), settings, promptOptions)
      const result = await pollSpeechGeneration(asyncJob.job_id, setGenerateStatus)
      if (result.audio) {
        const bytes = Uint8Array.from(atob(result.audio), (c) => c.charCodeAt(0))
        setAudioBlob(new Blob([bytes], { type: 'audio/wav' }))
        setGenerateStatus('Completed')
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

  if (loading) {
    return (
      <div className="space-y-6 animate-pulse">
        <div className="grid lg:grid-cols-2 gap-6">
          <div className="h-64 bg-raised rounded-xl" />
          <div className="h-64 bg-raised rounded-xl" />
        </div>
      </div>
    )
  }

  if (!voice) {
    return (
      <div className="rounded-lg border border-dashed border-edge bg-surface px-4 py-8 text-center text-sm text-muted">
        Voice not found or failed to load.
      </div>
    )
  }
  const supportsPromptControls = Boolean(voice.model_size?.includes('1.7'))

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-muted text-xs font-mono">
        <span>{voice.model_size || 'base'}</span>
        {typeof voice.epoch === 'number' && <span>epoch {voice.epoch}</span>}
        {voice.run_name && <span>run {voice.run_name}</span>}
        <span>updated {formatDate(voice.updated_at ?? voice.created_at)}</span>
        <button type="button" className="cursor-pointer hover:text-accent transition-colors" title={`Click to copy: ${voice.voice_id}`} onClick={() => { void navigator.clipboard.writeText(voice.voice_id) }}>{voice.voice_id.slice(0, 12)}</button>
      </div>

      {voice.description && (
        <p className="text-subtle text-sm max-w-xl -mt-3">{voice.description}</p>
      )}

      {error && (
        <div className="bg-error-dim border border-error/20 rounded-lg px-3 py-2 text-error text-xs">{error}</div>
      )}

      <div className="grid lg:grid-cols-2 gap-6">
        {/* Voice Settings */}
        <div className="bg-raised border border-edge rounded-xl p-5">
          <h2 className="text-heading font-semibold text-sm mb-5">Voice Settings</h2>
          <div className="mb-4 rounded-lg border border-edge bg-surface px-3 py-2 text-[11px] font-mono text-subtle">
            Current preset: stab {settings.stability.toFixed(2)} · sim {settings.similarity_boost.toFixed(2)} · style {settings.style.toFixed(2)} · speed {settings.speed.toFixed(2)}
          </div>
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

          <div className="mt-5 pt-5 border-t border-edge space-y-3">
            <div>
              <h3 className="text-heading text-xs font-semibold">Prompt Controls</h3>
              <p className="text-muted text-[10px] mt-1 leading-relaxed">
                Applied on `1.7B` voices. `0.6B` custom voices ignore this path.
              </p>
            </div>

            <div>
              <label className="text-subtle text-xs font-medium mb-1.5 block">Style Prompt</label>
              <textarea
                value={stylePrompt}
                onChange={(e) => setStylePrompt(e.target.value)}
                disabled={!supportsPromptControls}
                placeholder="Calm, high-authority market briefing tone with crisp phrasing."
                rows={3}
                className="w-full bg-surface border border-edge rounded-lg px-3 py-2 text-sm text-primary placeholder:text-muted disabled:opacity-50 disabled:cursor-not-allowed focus:border-accent transition-colors resize-none"
              />
            </div>

            <div>
              <label className="text-subtle text-xs font-medium mb-1.5 block">Instruct</label>
              <textarea
                value={instruct}
                onChange={(e) => setInstruct(e.target.value)}
                disabled={!supportsPromptControls}
                placeholder="Stay highly similar to the source speaker and end sentences with controlled emphasis."
                rows={3}
                className="w-full bg-surface border border-edge rounded-lg px-3 py-2 text-sm text-primary placeholder:text-muted disabled:opacity-50 disabled:cursor-not-allowed focus:border-accent transition-colors resize-none"
              />
            </div>
          </div>
        </div>

        {/* Quick Generate */}
        <div className="bg-raised border border-edge rounded-xl p-5">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-heading font-semibold text-sm">Generate Speech</h2>
            {text.length > 0 && <span className="text-muted text-[10px] font-mono">{text.length} chars</span>}
          </div>

          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => {
              if ((e.metaKey || e.ctrlKey) && e.key === 'Enter' && text.trim() && !generating && voice.status === 'ready') {
                e.preventDefault()
                handleGenerate()
              }
            }}
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
            {generating ? (generateStatus ? `Generating (${generateStatus})...` : 'Generating...') : (<>Generate Speech <kbd className="hidden sm:inline-block ml-1 text-[10px] opacity-50 font-mono">{'\u2318'}↵</kbd></>)}
          </button>

          {genError && (
            <div className="bg-error-dim border border-error/20 rounded-lg px-3 py-2 text-error text-xs mb-4">
              {genError}
            </div>
          )}

          <AudioPlayer blob={audioBlob ?? undefined} generating={generating} />
        </div>
      </div>

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
