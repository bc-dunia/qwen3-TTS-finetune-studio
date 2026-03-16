import { useState, useEffect, useCallback } from 'react'
import { useSearchParams } from 'react-router'
import {
  fetchVoices,
  pollSpeechGeneration,
  startSpeechGenerationAsync,
  type SpeechGenerationOptions,
  type Voice,
  type VoiceSettings,
  DEFAULT_VOICE_SETTINGS,
} from '../lib/api'
import { AudioPlayer } from '../components/AudioPlayer'

interface HistoryEntry {
  id: string
  text: string
  voiceName: string
  voiceId: string
  createdAt: string
}

const HISTORY_KEY = 'tts-generation-history'
const MAX_HISTORY = 20

export function Playground() {
  const [searchParams, setSearchParams] = useSearchParams()
  const [voices, setVoices] = useState<Voice[]>([])
  const [loadingVoices, setLoadingVoices] = useState(true)

  const [selectedVoiceId, setSelectedVoiceId] = useState('')
  const [text, setText] = useState('')
  const [settings, setSettings] = useState<VoiceSettings>(DEFAULT_VOICE_SETTINGS)
  const [stylePrompt, setStylePrompt] = useState('')
  const [instruct, setInstruct] = useState('')

  const [generating, setGenerating] = useState(false)
  const [generateStatus, setGenerateStatus] = useState('')
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null)
  const [error, setError] = useState('')

  const [history, setHistory] = useState<HistoryEntry[]>(() => {
    try {
      const stored = localStorage.getItem(HISTORY_KEY)
      return stored ? (JSON.parse(stored) as HistoryEntry[]) : []
    } catch {
      return []
    }
  })
  const requestedVoiceId = searchParams.get('voice') ?? ''

  // Load voices
  useEffect(() => {
    let cancelled = false

    async function load() {
      try {
        const data = await fetchVoices()
        if (cancelled) return
        const readyVoices = data.voices.filter((v) => v.status === 'ready')
        setVoices(readyVoices)
        if (readyVoices.length > 0) {
          const requestedVoice = requestedVoiceId
            ? readyVoices.find((voice) => voice.voice_id === requestedVoiceId)
            : null
          if (requestedVoice) {
            setSelectedVoiceId(requestedVoice.voice_id)
          } else if (!selectedVoiceId || !readyVoices.some((voice) => voice.voice_id === selectedVoiceId)) {
            setSelectedVoiceId(readyVoices[0].voice_id)
          }
        }
      } catch {
        // silently fail — user can still type
      } finally {
        if (!cancelled) setLoadingVoices(false)
      }
    }

    load()
    return () => { cancelled = true }
  }, [requestedVoiceId, selectedVoiceId])

  // Save history to localStorage
  const saveHistory = useCallback((entries: HistoryEntry[]) => {
    const trimmed = entries.slice(0, MAX_HISTORY)
    setHistory(trimmed)
    localStorage.setItem(HISTORY_KEY, JSON.stringify(trimmed))
  }, [])

  async function handleGenerate() {
    if (!selectedVoiceId || !text.trim()) return

    setGenerating(true)
    setGenerateStatus('Queued...')
    setError('')
    setAudioBlob(null)

    try {
      const promptOptions: SpeechGenerationOptions = {
        stylePrompt: stylePrompt.trim() || undefined,
        instruct: instruct.trim() || undefined,
      }
      const asyncJob = await startSpeechGenerationAsync(selectedVoiceId, text.trim(), settings, promptOptions)
      const result = await pollSpeechGeneration(asyncJob.job_id, setGenerateStatus)
      if (result.audio) {
        const bytes = Uint8Array.from(atob(result.audio), (c) => c.charCodeAt(0))
        setAudioBlob(new Blob([bytes], { type: 'audio/wav' }))
        setGenerateStatus('Completed')
      }

      // Add to history
      const voice = voices.find((v) => v.voice_id === selectedVoiceId)
      const entry: HistoryEntry = {
        id: crypto.randomUUID(),
        text: text.trim(),
        voiceName: voice?.name ?? 'Unknown',
        voiceId: selectedVoiceId,
        createdAt: new Date().toISOString(),
      }
      saveHistory([entry, ...history])
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Generation failed')
      setGenerateStatus('')
    } finally {
      setGenerating(false)
    }
  }

  function updateSetting(key: keyof VoiceSettings, value: number) {
    setSettings((prev) => ({ ...prev, [key]: value }))
  }

  function clearHistory() {
    setHistory([])
    localStorage.removeItem(HISTORY_KEY)
  }

  const selectedVoice = voices.find((v) => v.voice_id === selectedVoiceId)
  const supportsPromptControls = Boolean(selectedVoice?.model_size?.includes('1.7'))

  useEffect(() => {
    if (!selectedVoice) return
    if (requestedVoiceId !== selectedVoice.voice_id) {
      const next = new URLSearchParams()
      next.set('voice', selectedVoice.voice_id)
      setSearchParams(next, { replace: true })
    }
    setSettings({
      stability: selectedVoice.settings.stability ?? DEFAULT_VOICE_SETTINGS.stability,
      similarity_boost: selectedVoice.settings.similarity_boost ?? DEFAULT_VOICE_SETTINGS.similarity_boost,
      style: selectedVoice.settings.style ?? DEFAULT_VOICE_SETTINGS.style,
      speed: selectedVoice.settings.speed ?? DEFAULT_VOICE_SETTINGS.speed,
    })
  }, [requestedVoiceId, selectedVoice, setSearchParams])

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-heading text-2xl font-bold">Playground</h1>
        <p className="text-subtle text-sm mt-1">Generate speech with your cloned voices</p>
      </div>

      <div className="grid lg:grid-cols-[1fr_320px] gap-6">
        {/* Main Panel */}
        <div className="space-y-5">
          {/* Voice Selector */}
          <div className="bg-raised border border-edge rounded-xl p-5">
            <label className="text-subtle text-xs font-medium mb-2 block">Voice</label>
            {loadingVoices ? (
              <div className="h-10 bg-surface rounded-lg animate-pulse" />
            ) : voices.length === 0 ? (
              <div className="text-muted text-sm py-2">
                No ready voices available. Create and train a voice first.
              </div>
            ) : (
              <select
                value={selectedVoiceId}
                onChange={(e) => setSelectedVoiceId(e.target.value)}
                className="w-full bg-surface border border-edge rounded-lg px-3 py-2.5 text-sm text-primary focus:border-accent transition-colors appearance-none cursor-pointer"
              >
                {voices.map((v) => (
                  <option key={v.voice_id} value={v.voice_id}>
                    {v.name} — {v.model_size || 'base'}{typeof v.epoch === 'number' ? ` · epoch ${v.epoch}` : ''}
                  </option>
                ))}
              </select>
            )}
          </div>

          {/* Text Input */}
          <div className="bg-raised border border-edge rounded-xl p-5">
            <div className="flex items-center justify-between mb-2">
              <label className="text-subtle text-xs font-medium">Text</label>
              <span className="text-muted text-[10px] font-mono">{text.length} chars</span>
            </div>
            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Enter the text you want to convert to speech..."
              rows={6}
              className="w-full bg-surface border border-edge rounded-lg px-3 py-2.5 text-sm text-primary placeholder:text-muted focus:border-accent transition-colors resize-none leading-relaxed"
            />
          </div>

          {/* Generate Button */}
          <button
            onClick={handleGenerate}
            disabled={!selectedVoiceId || !text.trim() || generating}
            className="w-full bg-accent hover:bg-accent-light text-void font-bold text-sm py-3.5 rounded-xl disabled:opacity-40 disabled:cursor-not-allowed transition-colors flex items-center justify-center gap-2"
            type="button"
          >
            {generating ? (
              <>
                <svg className="w-4 h-4 animate-spin" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                  <circle cx="12" cy="12" r="10" strokeDasharray="60" strokeDashoffset="15" strokeLinecap="round" />
                </svg>
                {generateStatus ? `Generating (${generateStatus})...` : 'Generating...'}
              </>
            ) : (
              <>
                <svg className="w-4 h-4" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M8 5.14v13.72a1 1 0 0 0 1.5.86l11.24-7.36a1 1 0 0 0 0-1.72L9.5 4.28A1 1 0 0 0 8 5.14z" />
                </svg>
                Generate Speech
              </>
            )}
          </button>

          {/* Error */}
          {error && (
            <div className="bg-error-dim border border-error/20 rounded-lg px-4 py-3 text-error text-sm">
              {error}
            </div>
          )}

          {/* Audio Player */}
          <AudioPlayer
            blob={audioBlob ?? undefined}
            generating={generating}
          />

          {/* Generation History */}
          <div className="bg-raised border border-edge rounded-xl p-5">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-heading font-semibold text-sm">Generation History</h2>
              {history.length > 0 && (
                <button
                  onClick={clearHistory}
                  className="text-muted text-xs hover:text-error transition-colors"
                  type="button"
                >
                  Clear
                </button>
              )}
            </div>

            {history.length === 0 ? (
              <p className="text-muted text-sm text-center py-6">No generations yet</p>
            ) : (
              <div className="space-y-2 max-h-64 overflow-y-auto">
                {history.map((entry) => (
                  <div
                    key={entry.id}
                    className="flex items-start gap-3 px-3 py-2.5 rounded-lg bg-surface"
                  >
                    <div className="w-6 h-6 rounded-full bg-accent-dim flex items-center justify-center shrink-0 mt-0.5">
                      <svg className="w-3 h-3 text-accent" viewBox="0 0 24 24" fill="currentColor">
                        <path d="M8 5.14v13.72a1 1 0 0 0 1.5.86l11.24-7.36a1 1 0 0 0 0-1.72L9.5 4.28A1 1 0 0 0 8 5.14z" />
                      </svg>
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-primary text-xs leading-relaxed line-clamp-2">{entry.text}</p>
                      <p className="text-muted text-[10px] font-mono mt-1">
                        {entry.voiceName} · {new Date(entry.createdAt).toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' })}
                      </p>
                    </div>
                    <button
                      onClick={() => {
                        setText(entry.text)
                        setSelectedVoiceId(entry.voiceId)
                      }}
                      className="text-muted hover:text-accent text-[10px] font-mono shrink-0"
                      type="button"
                    >
                      Reuse
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Settings Panel */}
        <div className="space-y-5">
          <div className="bg-raised border border-edge rounded-xl p-5 lg:sticky lg:top-8">
            <h2 className="text-heading font-semibold text-sm mb-5">Voice Settings</h2>

            {selectedVoice && (
              <div className="mb-5 pb-4 border-b border-edge">
                <div className="text-primary text-xs font-medium">{selectedVoice.name}</div>
                <div className="text-muted text-[10px] font-mono mt-0.5">
                  {selectedVoice.model_size || 'base'} · {selectedVoice.voice_id.slice(0, 8)}
                </div>
              </div>
            )}

            <div className="space-y-5">
              <SliderControl
                label="Stability"
                value={settings.stability}
                onChange={(v) => updateSetting('stability', v)}
                min={0}
                max={1}
                step={0.01}
              />
              <SliderControl
                label="Similarity"
                value={settings.similarity_boost}
                onChange={(v) => updateSetting('similarity_boost', v)}
                min={0}
                max={1}
                step={0.01}
              />
              <SliderControl
                label="Style"
                value={settings.style}
                onChange={(v) => updateSetting('style', v)}
                min={0}
                max={1}
                step={0.01}
              />
              <SliderControl
                label="Speed"
                value={settings.speed}
                onChange={(v) => updateSetting('speed', v)}
                min={0.5}
                max={2.0}
                step={0.05}
              />
            </div>

            <div className="mt-5 pt-5 border-t border-edge space-y-3">
              <div>
                <h3 className="text-heading text-xs font-semibold">Prompt Controls</h3>
                <p className="text-muted text-[10px] mt-1 leading-relaxed">
                  Style Prompt and Instruct are applied on `1.7B` voices. `0.6B` custom voices ignore this path.
                </p>
              </div>

              <div>
                <label className="text-subtle text-xs font-medium mb-1.5 block">Style Prompt</label>
                <textarea
                  value={stylePrompt}
                  onChange={(e) => setStylePrompt(e.target.value)}
                  disabled={!supportsPromptControls}
                  placeholder="Warm, polished investor briefing tone with measured emphasis."
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
                  placeholder="Speak with higher similarity, slightly slower pacing, and clean sentence endings."
                  rows={3}
                  className="w-full bg-surface border border-edge rounded-lg px-3 py-2 text-sm text-primary placeholder:text-muted disabled:opacity-50 disabled:cursor-not-allowed focus:border-accent transition-colors resize-none"
                />
              </div>
            </div>

            <button
              onClick={() => setSettings(
                selectedVoice
                  ? {
                      stability: selectedVoice.settings.stability ?? DEFAULT_VOICE_SETTINGS.stability,
                      similarity_boost: selectedVoice.settings.similarity_boost ?? DEFAULT_VOICE_SETTINGS.similarity_boost,
                      style: selectedVoice.settings.style ?? DEFAULT_VOICE_SETTINGS.style,
                      speed: selectedVoice.settings.speed ?? DEFAULT_VOICE_SETTINGS.speed,
                    }
                  : DEFAULT_VOICE_SETTINGS,
              )}
              className="w-full mt-5 text-muted text-xs hover:text-primary border border-edge rounded-lg py-2 transition-colors"
              type="button"
            >
              Reset to Voice Defaults
            </button>
            <button
              onClick={() => {
                setStylePrompt('')
                setInstruct('')
              }}
              className="w-full mt-2 text-muted text-xs hover:text-primary border border-edge rounded-lg py-2 transition-colors"
              type="button"
            >
              Clear Prompt Controls
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

// ── Slider Control ─────────────────────────────────────────────────────────────

function SliderControl({
  label,
  value,
  onChange,
  min,
  max,
  step,
}: {
  label: string
  value: number
  onChange: (v: number) => void
  min: number
  max: number
  step: number
}) {
  return (
    <div>
      <div className="flex items-center justify-between mb-1.5">
        <span className="text-subtle text-xs">{label}</span>
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
    </div>
  )
}
