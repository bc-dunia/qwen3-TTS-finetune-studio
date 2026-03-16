import { useState, useEffect, useRef } from 'react'
import { useNavigate } from 'react-router'
import {
  ApiError,
  fetchVoices,
  createVoiceDraft,
  deleteVoice,
  startTraining,
  uploadVoiceDatasetFile,
  type UploadProgress,
  type Voice,
  type VoiceModelSize,
} from '../lib/api'
import { VoiceCard } from '../components/VoiceCard'

const LONGFORM_RAW_UPLOAD_THRESHOLD_BYTES = 32 * 1024 * 1024
const LONGFORM_RAW_UPLOAD_EXTENSIONS = ['.mp3', '.mp4', '.m4a', '.flac']

function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return '0 B'
  const units = ['B', 'KB', 'MB', 'GB']
  let size = bytes
  let unitIndex = 0
  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024
    unitIndex += 1
  }
  const digits = size >= 10 || unitIndex === 0 ? 0 : 1
  return `${size.toFixed(digits)} ${units[unitIndex]}`
}

function isSupportedUploadFile(file: File): boolean {
  const lowerName = file.name.toLowerCase()
  return (
    file.type.startsWith('audio/') ||
    file.type === 'video/mp4' ||
    lowerName.endsWith('.wav') ||
    lowerName.endsWith('.wave') ||
    lowerName.endsWith('.mp3') ||
    lowerName.endsWith('.mp4') ||
    lowerName.endsWith('.m4a') ||
    lowerName.endsWith('.flac')
  )
}

function hasMatchingExtension(name: string, extensions: string[]): boolean {
  const lowerName = name.toLowerCase()
  return extensions.some((extension) => lowerName.endsWith(extension))
}

function shouldAutoStartTrainingFromRawUploads(files: File[]): boolean {
  if (files.length === 0) return false
  const longFormFiles = files.filter((file) =>
    file.size >= LONGFORM_RAW_UPLOAD_THRESHOLD_BYTES ||
    hasMatchingExtension(file.name, LONGFORM_RAW_UPLOAD_EXTENSIONS),
  )
  return longFormFiles.length > 0 && (files.length <= 3 || longFormFiles.length === files.length)
}

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

  useEffect(() => {
    const onApiKeyChanged = () => {
      loadVoices()
    }
    window.addEventListener('xi-api-key-changed', onApiKeyChanged)
    return () => {
      window.removeEventListener('xi-api-key-changed', onApiKeyChanged)
    }
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
  const navigate = useNavigate()
  const [name, setName] = useState('')
  const [description, setDescription] = useState('')
  const [audioFiles, setAudioFiles] = useState<File[]>([])
  const [modelSize, setModelSize] = useState<VoiceModelSize>('0.6B')
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState('')
  const [statusText, setStatusText] = useState('')
  const [autoPrepare, setAutoPrepare] = useState(true)
  const [openTrainingWhenReady, setOpenTrainingWhenReady] = useState(true)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [dragOver, setDragOver] = useState(false)
  const [uploadProgress, setUploadProgress] = useState<{
    fileName: string
    fileIndex: number
    totalFiles: number
    loadedBytes: number
    totalBytes: number
    completedBytes: number
    totalUploadBytes: number
  } | null>(null)

  function buildUploadStatusText(
    fileIndex: number,
    totalFiles: number,
    fileName: string,
    progress?: UploadProgress,
  ): string {
    if (!progress) {
      return `Uploading ${fileIndex + 1}/${totalFiles}: ${fileName}`
    }
    if (progress.totalBytes > 0 && progress.loadedBytes >= progress.totalBytes) {
      return `Processing ${fileIndex + 1}/${totalFiles}: ${fileName}`
    }
    return `Uploading ${fileIndex + 1}/${totalFiles}: ${fileName}`
  }

  function buildDatasetTarget(voiceId: string, partialUpload = false): string {
    const params = new URLSearchParams()
    if (autoPrepare) params.set('autoPrepare', '1')
    if (openTrainingWhenReady) params.set('openTraining', '1')
    if (partialUpload) params.set('uploadWarning', '1')
    const query = params.toString()
    return `/voices/${voiceId}/dataset${query ? `?${query}` : ''}`
  }

  function buildTrainingTarget(voiceId: string): string {
    const params = new URLSearchParams({
      recommended: '1',
    })
    return `/voices/${voiceId}/training?${params.toString()}`
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    if (!name.trim() || audioFiles.length === 0) return

    setSubmitting(true)
    setError('')
    setStatusText('Creating voice...')
    setUploadProgress(null)

    let createdVoiceId = ''
    let uploadedCount = 0
    const totalUploadBytes = audioFiles.reduce((sum, file) => sum + file.size, 0)
    let completedBytes = 0
    try {
      const { voice_id } = await createVoiceDraft(name.trim(), description.trim(), modelSize)
      createdVoiceId = voice_id
      for (let index = 0; index < audioFiles.length; index += 1) {
        const file = audioFiles[index]
        setStatusText(buildUploadStatusText(index, audioFiles.length, file.name))
        setUploadProgress({
          fileName: file.name,
          fileIndex: index,
          totalFiles: audioFiles.length,
          loadedBytes: 0,
          totalBytes: file.size,
          completedBytes,
          totalUploadBytes,
        })
        await uploadVoiceDatasetFile(voice_id, file, (progress: UploadProgress) => {
          setStatusText(buildUploadStatusText(index, audioFiles.length, file.name, progress))
          setUploadProgress({
            fileName: file.name,
            fileIndex: index,
            totalFiles: audioFiles.length,
            loadedBytes: progress.loadedBytes,
            totalBytes: progress.totalBytes,
            completedBytes,
            totalUploadBytes,
          })
        })
        completedBytes += file.size
        uploadedCount += 1
      }

      const shouldStartTrainingFromRaw = autoPrepare && shouldAutoStartTrainingFromRawUploads(audioFiles)
      if (shouldStartTrainingFromRaw) {
        setStatusText('Upload complete. Starting raw-media preprocessing and training...')
        try {
          await startTraining(voice_id, {})
        } catch (err) {
          if (!(err instanceof ApiError) || err.status !== 409) {
            throw err
          }
        }

        setUploadProgress(null)
        navigate(openTrainingWhenReady ? buildTrainingTarget(voice_id) : `/voices/${voice_id}`)
        onCreated()
        return
      }

      setStatusText('Upload complete. Opening Dataset Studio...')
      setUploadProgress(null)
      navigate(buildDatasetTarget(voice_id))
      onCreated()
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to create voice'
      setUploadProgress(null)
      if (createdVoiceId && uploadedCount === 0) {
        try {
          await deleteVoice(createdVoiceId)
        } catch {
          // Ignore cleanup failure; surface the original upload error.
        }
        setError(`${message} Empty draft voice was removed.`)
      } else if (createdVoiceId && uploadedCount > 0) {
        navigate(buildDatasetTarget(createdVoiceId, true))
        onCreated()
        return
      } else {
        setError(message)
      }
      setStatusText('')
      setSubmitting(false)
    }
  }

  function handleDrop(e: React.DragEvent) {
    e.preventDefault()
    setDragOver(false)
    addAudioFiles(Array.from(e.dataTransfer.files).filter((file) => isSupportedUploadFile(file)))
  }

  function handleDragOver(e: React.DragEvent) {
    e.preventDefault()
    setDragOver(true)
  }

  function addAudioFiles(files: File[]) {
    if (files.length === 0) return
    setError('')
    setAudioFiles((prev) => [...prev, ...files])
  }

  const totalUploadedBytes = uploadProgress
    ? Math.min(uploadProgress.totalUploadBytes, uploadProgress.completedBytes + uploadProgress.loadedBytes)
    : 0
  const totalProgressPercent = uploadProgress && uploadProgress.totalUploadBytes > 0
    ? Math.round((totalUploadedBytes / uploadProgress.totalUploadBytes) * 100)
    : 0
  const currentFilePercent = uploadProgress && uploadProgress.totalBytes > 0
    ? Math.round((uploadProgress.loadedBytes / uploadProgress.totalBytes) * 100)
    : 0

  return (
    <div className="fixed inset-0 z-50 overflow-y-auto bg-black/60 p-3 sm:p-4">
      <div className="flex min-h-full items-start justify-center sm:items-center">
        <div
          className="bg-surface border border-edge rounded-2xl w-full max-w-lg max-h-[calc(100dvh-1.5rem)] sm:max-h-[calc(100dvh-2rem)] animate-slide-up overflow-hidden flex flex-col"
          role="dialog"
          aria-label="Create voice"
        >
          {/* Header */}
          <div className="flex items-center justify-between px-5 py-4 border-b border-edge shrink-0">
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
          <form onSubmit={handleSubmit} className="flex min-h-0 flex-1 flex-col">
            <div className="min-h-0 flex-1 overflow-y-auto px-5 py-5 space-y-4">
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

              <div>
                <label className="text-subtle text-xs font-medium mb-1.5 block">
                  Base Model
                </label>
                <select
                  value={modelSize}
                  onChange={(e) => setModelSize(e.target.value as VoiceModelSize)}
                  className="w-full bg-raised border border-edge rounded-lg px-3 py-2 text-sm text-primary focus:border-accent transition-colors"
                >
                  <option value="0.6B">Qwen3-TTS 0.6B (recommended for cloud training/serving)</option>
                  <option value="1.7B">Qwen3-TTS 1.7B (higher capacity, slower/more expensive)</option>
                </select>
              </div>

              {/* Audio Upload */}
              <div>
                <label className="text-subtle text-xs font-medium mb-1.5 block">
                  Training Audio Files <span className="text-error">*</span>
                </label>
                <div className="mb-2 rounded-lg border border-edge bg-raised px-3 py-2 text-[11px] leading-relaxed text-subtle">
                  Files upload through the API into storage. Short clips continue to Dataset Studio for transcription and cleanup; large MP3, M4A, FLAC, or MP4 source files automatically start the raw-media training pipeline after upload.
                  Best results: 24kHz mono WAV, 3-15s per clip, clean speech only, at least 10 minutes total.
                </div>
                <div
                  className={`
                    border-2 border-dashed rounded-lg p-5 text-center cursor-pointer transition-colors
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
                      <div className="text-muted text-xs mt-1">WAV, MP3, M4A, FLAC, or MP4 audio. Upload the full training set, not just one reference clip.</div>
                    </div>
                  )}
                </div>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".wav,.wave,.mp3,.mp4,.m4a,.flac,audio/*,video/mp4"
                  multiple
                  onChange={(e) => {
                    const files = e.target.files ? Array.from(e.target.files) : []
                    addAudioFiles(files.filter((file) => isSupportedUploadFile(file)))
                    e.target.value = ''
                  }}
                  className="hidden"
                />
              </div>

              <div className="rounded-lg border border-edge bg-raised px-3 py-3 space-y-3">
                <label className="flex items-start gap-3 text-sm text-primary">
                  <input
                    type="checkbox"
                    checked={autoPrepare}
                    onChange={(e) => setAutoPrepare(e.target.checked)}
                    className="mt-0.5 accent-accent"
                  />
                  <span>
                    <span className="font-medium">Auto prepare dataset after upload</span>
                    <span className="block text-[11px] text-subtle mt-1">
                      Runs auto transcription, transcript review, finalized dataset creation, and active dataset selection.
                    </span>
                  </span>
                </label>

                <label className="flex items-start gap-3 text-sm text-primary">
                  <input
                    type="checkbox"
                    checked={openTrainingWhenReady}
                    onChange={(e) => setOpenTrainingWhenReady(e.target.checked)}
                    disabled={!autoPrepare}
                    className="mt-0.5 accent-accent"
                  />
                  <span>
                    <span className="font-medium">Open training when dataset is ready</span>
                    <span className="block text-[11px] text-subtle mt-1">
                      Leaves one final check on the Training page before you start the run.
                    </span>
                  </span>
                </label>
              </div>
            </div>

            <div className="shrink-0 border-t border-edge bg-surface/95 px-5 py-4 space-y-3">
              {/* Error */}
              {error && (
                <div className="bg-error-dim border border-error/20 rounded-lg px-3 py-2 text-error text-xs">
                  {error}
                </div>
              )}

              {statusText && !error && (
                <div className="bg-accent-dim border border-accent/20 rounded-lg px-3 py-2 text-accent text-xs">
                  {statusText}
                </div>
              )}

              {uploadProgress && !error && (
                <div className="rounded-lg border border-accent/20 bg-accent-dim px-3 py-3 text-xs text-primary space-y-2">
                  <div className="flex items-center justify-between gap-3">
                    <div className="min-w-0">
                      <div className="truncate font-medium text-accent">
                        {uploadProgress.fileName}
                      </div>
                      <div className="text-[11px] text-subtle mt-0.5">
                        file {uploadProgress.fileIndex + 1}/{uploadProgress.totalFiles}
                        {' · '}
                        {uploadProgress.loadedBytes >= uploadProgress.totalBytes && uploadProgress.totalBytes > 0
                          ? 'uploaded, finalizing on server'
                          : `current file ${currentFilePercent}%`}
                      </div>
                    </div>
                    <div className="text-right font-mono text-accent">
                      {uploadProgress.loadedBytes >= uploadProgress.totalBytes && uploadProgress.totalBytes > 0
                        ? 'finalizing'
                        : `${totalProgressPercent}%`}
                    </div>
                  </div>

                  <div className="h-2 overflow-hidden rounded-full bg-surface">
                    <div
                      className={`h-full rounded-full bg-accent transition-[width] duration-150 ${
                        uploadProgress.loadedBytes >= uploadProgress.totalBytes && uploadProgress.totalBytes > 0
                          ? 'animate-pulse'
                          : ''
                      }`}
                      style={{
                        width:
                          uploadProgress.loadedBytes >= uploadProgress.totalBytes && uploadProgress.totalBytes > 0
                            ? '100%'
                            : `${totalProgressPercent}%`,
                      }}
                    />
                  </div>

                  <div className="flex items-center justify-between gap-3 text-[11px] font-mono text-subtle">
                    <span>{formatBytes(totalUploadedBytes)} / {formatBytes(uploadProgress.totalUploadBytes)}</span>
                    <span>{formatBytes(uploadProgress.loadedBytes)} / {formatBytes(uploadProgress.totalBytes)}</span>
                  </div>
                </div>
              )}

              {/* Actions */}
              <div className="flex gap-3">
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
                  {submitting ? 'Uploading...' : 'Create Voice'}
                </button>
              </div>
            </div>
          </form>
        </div>
      </div>
    </div>
  )
}
