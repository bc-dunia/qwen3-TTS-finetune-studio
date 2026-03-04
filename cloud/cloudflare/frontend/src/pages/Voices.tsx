import { useState, useEffect, useRef } from 'react'
import { fetchVoices, createVoice, type Voice } from '../lib/api'
import { VoiceCard } from '../components/VoiceCard'

export function Voices() {
  const [voices, setVoices] = useState<Voice[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [showCreate, setShowCreate] = useState(false)

  async function loadVoices() {
    setLoading(true)
    setError('')
    try {
      const data = await fetchVoices()
      setVoices(data.voices)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load voices')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadVoices()
  }, [])

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-heading text-2xl font-bold">Voices</h1>
          <p className="text-subtle text-sm mt-1">
            {voices.length} voice{voices.length !== 1 ? 's' : ''} total
          </p>
        </div>
        <button
          onClick={() => setShowCreate(true)}
          className="inline-flex items-center gap-2 bg-accent hover:bg-accent-light text-void font-semibold text-sm px-5 py-2.5 rounded-lg transition-colors"
          type="button"
        >
          <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
            <line x1="12" y1="5" x2="12" y2="19" />
            <line x1="5" y1="12" x2="19" y2="12" />
          </svg>
          Create Voice
        </button>
      </div>

      {/* Error */}
      {error && (
        <div className="bg-error-dim border border-error/20 rounded-lg px-4 py-3 text-error text-sm">
          {error}
        </div>
      )}

      {/* Loading */}
      {loading && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {Array.from({ length: 6 }, (_, i) => (
            <div key={i} className="bg-raised border border-edge rounded-xl p-5 animate-pulse">
              <div className="h-4 bg-elevated rounded w-2/3 mb-3" />
              <div className="h-3 bg-elevated rounded w-full mb-2" />
              <div className="h-3 bg-elevated rounded w-1/2" />
            </div>
          ))}
        </div>
      )}

      {/* Voice Grid */}
      {!loading && voices.length > 0 && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {voices.map((voice) => (
            <VoiceCard key={voice.voice_id} voice={voice} />
          ))}
        </div>
      )}

      {/* Empty State */}
      {!loading && voices.length === 0 && !error && (
        <div className="text-center py-16">
          <div className="w-16 h-16 mx-auto mb-4 rounded-2xl bg-raised border border-edge flex items-center justify-center">
            <svg className="w-8 h-8 text-muted" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
              <rect x="9" y="2" width="6" height="11" rx="3" fill="currentColor" stroke="none" opacity="0.3" />
              <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
              <line x1="12" y1="19" x2="12" y2="22" />
            </svg>
          </div>
          <h3 className="text-heading font-semibold mb-1">No voices yet</h3>
          <p className="text-subtle text-sm mb-4">Create your first voice to get started</p>
          <button
            onClick={() => setShowCreate(true)}
            className="text-accent text-sm font-medium hover:text-accent-light"
            type="button"
          >
            Create Voice →
          </button>
        </div>
      )}

      {/* Create Voice Modal */}
      {showCreate && (
        <CreateVoiceModal
          onClose={() => setShowCreate(false)}
          onCreated={() => {
            setShowCreate(false)
            loadVoices()
          }}
        />
      )}
    </div>
  )
}

// ── Create Voice Modal ─────────────────────────────────────────────────────────

function CreateVoiceModal({
  onClose,
  onCreated,
}: {
  onClose: () => void
  onCreated: () => void
}) {
  const [name, setName] = useState('')
  const [description, setDescription] = useState('')
  const [audioFiles, setAudioFiles] = useState<File[]>([])
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState('')
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [dragOver, setDragOver] = useState(false)

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    if (!name.trim() || audioFiles.length === 0) return

    setSubmitting(true)
    setError('')

    try {
      await createVoice(name.trim(), description.trim(), audioFiles[0])
      onCreated()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create voice')
      setSubmitting(false)
    }
  }

  function handleDrop(e: React.DragEvent) {
    e.preventDefault()
    setDragOver(false)
    const files = Array.from(e.dataTransfer.files).filter(f => f.type.startsWith('audio/'))
    if (files.length > 0) {
      setAudioFiles(prev => [...prev, ...files])
    }
  }

  function handleDragOver(e: React.DragEvent) {
    e.preventDefault()
    setDragOver(true)
  }

  return (
    <div className="fixed inset-0 bg-black/60 z-50 flex items-center justify-center p-4">
      <div
        className="bg-surface border border-edge rounded-2xl w-full max-w-md animate-slide-up"
        role="dialog"
        aria-label="Create voice"
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-edge">
          <h2 className="text-heading font-semibold">Create Voice</h2>
          <button
            onClick={onClose}
            className="text-muted hover:text-primary p-1"
            type="button"
            aria-label="Close"
          >
            <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
              <line x1="18" y1="6" x2="6" y2="18" />
              <line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </button>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="p-6 space-y-4">
          {/* Name */}
          <div>
            <label className="text-subtle text-xs font-medium mb-1.5 block">
              Voice Name <span className="text-error">*</span>
            </label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="My Custom Voice"
              className="w-full bg-raised border border-edge rounded-lg px-3 py-2 text-sm text-primary placeholder:text-muted focus:border-accent transition-colors"
              required
            />
          </div>

          {/* Description */}
          <div>
            <label className="text-subtle text-xs font-medium mb-1.5 block">
              Description
            </label>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Describe the voice characteristics..."
              rows={3}
              className="w-full bg-raised border border-edge rounded-lg px-3 py-2 text-sm text-primary placeholder:text-muted focus:border-accent transition-colors resize-none"
            />
          </div>

          {/* Audio Upload */}
          <div>
            <label className="text-subtle text-xs font-medium mb-1.5 block">
              Reference Audio <span className="text-error">*</span>
            </label>
            <div
              className={`
                border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-colors
                ${dragOver
                  ? 'border-accent bg-accent-dim'
                  : audioFiles.length > 0
                    ? 'border-accent/30 bg-accent-dim'
                    : 'border-edge hover:border-subtle'
                }
              `}
              onClick={() => fileInputRef.current?.click()}
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              onDragLeave={() => setDragOver(false)}
              onKeyDown={(e) => { if (e.key === 'Enter') fileInputRef.current?.click() }}
              role="button"
              tabIndex={0}
            >
              {audioFiles.length > 0 ? (
                <div>
                  <div className="text-accent text-sm font-medium">
                    {audioFiles.length} file{audioFiles.length !== 1 ? 's' : ''} selected
                  </div>
                  <div className="text-muted text-xs mt-1">
                    {(audioFiles.reduce((sum, f) => sum + f.size, 0) / 1024 / 1024).toFixed(1)} MB total
                  </div>
                  <div className="flex flex-wrap gap-1 mt-2 justify-center">
                    {audioFiles.map((f, i) => (
                      <span key={i} className="inline-flex items-center gap-1 bg-elevated text-subtle text-xs px-2 py-0.5 rounded-full">
                        {f.name.length > 20 ? f.name.slice(0, 17) + '...' : f.name}
                        <button
                          type="button"
                          onClick={(e) => {
                            e.stopPropagation()
                            setAudioFiles(prev => prev.filter((_, idx) => idx !== i))
                          }}
                          className="text-muted hover:text-error ml-0.5"
                          aria-label={"Remove " + f.name}
                        >
                          {"\u00d7"}
                        </button>
                      </span>
                    ))}
                  </div>
                </div>
              ) : (
                <div>
                  <svg className="w-8 h-8 mx-auto text-muted mb-2" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                    <polyline points="17 8 12 3 7 8" />
                    <line x1="12" y1="3" x2="12" y2="15" />
                  </svg>
                  <div className="text-subtle text-sm">Drop audio files or click to browse</div>
                  <div className="text-muted text-xs mt-1">WAV format recommended — multiple files supported</div>
                </div>
              )}
            </div>
            <input
              ref={fileInputRef}
              type="file"
              accept="audio/*"
              multiple
              onChange={(e) => {
                const files = e.target.files ? Array.from(e.target.files) : []
                if (files.length > 0) setAudioFiles(prev => [...prev, ...files])
              }}
              className="hidden"
            />
          </div>

          {/* Error */}
          {error && (
            <div className="bg-error-dim border border-error/20 rounded-lg px-3 py-2 text-error text-xs">
              {error}
            </div>
          )}

          {/* Actions */}
          <div className="flex gap-3 pt-2">
            <button
              type="button"
              onClick={onClose}
              className="flex-1 bg-raised border border-edge text-subtle font-medium text-sm py-2.5 rounded-lg hover:text-primary transition-colors"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={!name.trim() || audioFiles.length === 0 || submitting}
              className="flex-1 bg-accent hover:bg-accent-light text-void font-semibold text-sm py-2.5 rounded-lg disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            >
              {submitting ? 'Creating...' : 'Create Voice'}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}
