import { useEffect, useRef, useState } from 'react'
import { useNavigate, useParams, useSearchParams } from 'react-router'
import {
  ApiError,
  createFinalizedDataset,
  fetchVoice,
  fetchVoiceDatasets,
  fetchVoiceRawDatasetFiles,
  formatDateTime,
  reviewDatasetTexts,
  retranscribeDatasetEntries,
  selectVoiceDataset,
  startTraining,
  type RawDatasetFile,
  type Voice,
} from '../lib/api'

type DraftRow = RawDatasetFile & {
  selected: boolean
  text: string
  asrScore: number | null
  reviewScore: number | null
  reviewIssues: string[]
  provider: string | null
}

type AutoPrepareProgress = {
  phase: 'transcribe' | 'review' | 'create' | 'select' | 'start' | 'done'
  completed: number
  total: number
  detail: string
}

const LONGFORM_RAW_UPLOAD_THRESHOLD_BYTES = 32 * 1024 * 1024
const LONGFORM_RAW_UPLOAD_EXTENSIONS = ['.mp3', '.mp4', '.m4a', '.flac']

function getDefaultDatasetName(): string {
  const date = new Date()
  const y = date.getFullYear()
  const m = String(date.getMonth() + 1).padStart(2, '0')
  const d = String(date.getDate()).padStart(2, '0')
  return `curated_clean_${y}${m}${d}`
}

function isReferenceAudioKey(refAudioKey: string | null | undefined): boolean {
  return Boolean(refAudioKey && /\/ref_audio\.[^/]+$/i.test(refAudioKey))
}

function inferCurrentDatasetName(voice: Voice | null): string | null {
  const refAudioKey = voice?.ref_audio_r2_key
  if (!isReferenceAudioKey(refAudioKey)) {
    return null
  }
  const parts = String(refAudioKey).split('/').filter(Boolean)
  return parts.length >= 4 ? parts[2] ?? null : null
}

function hasMatchingExtension(name: string, extensions: string[]): boolean {
  const lowerName = name.toLowerCase()
  return extensions.some((extension) => lowerName.endsWith(extension))
}

function shouldUseRawTrainingPipeline(rows: RawDatasetFile[]): boolean {
  if (rows.length === 0) return false
  const longFormRows = rows.filter((row) =>
    row.size >= LONGFORM_RAW_UPLOAD_THRESHOLD_BYTES ||
    hasMatchingExtension(row.filename, LONGFORM_RAW_UPLOAD_EXTENSIONS),
  )
  return longFormRows.length > 0 && (rows.length <= 3 || longFormRows.length === rows.length)
}

function mergeDraftRows(rawFiles: RawDatasetFile[], previous: DraftRow[]): DraftRow[] {
  const previousByKey = new Map(previous.map((row) => [row.key, row]))
  return rawFiles.map((file, index) => {
    const existing = previousByKey.get(file.key)
    return {
      ...file,
      selected: existing?.selected ?? index < 12,
      text: existing?.text ?? '',
      asrScore: existing?.asrScore ?? null,
      reviewScore: existing?.reviewScore ?? null,
      reviewIssues: existing?.reviewIssues ?? [],
      provider: existing?.provider ?? null,
    }
  })
}

function chunkArray<T>(items: T[], size: number): T[][] {
  const chunks: T[][] = []
  for (let index = 0; index < items.length; index += size) {
    chunks.push(items.slice(index, index + size))
  }
  return chunks
}

function encodeTrainingQuery(voiceId: string, datasetName?: string): string {
  const params = new URLSearchParams({
    voiceId,
    recommended: '1',
  })
  if (datasetName) {
    params.set('datasetName', datasetName)
  }
  return `/training?${params.toString()}`
}

export function VoiceDataset() {
  const { voiceId = '' } = useParams()
  const navigate = useNavigate()
  const [searchParams] = useSearchParams()

  const [voice, setVoice] = useState<Voice | null>(null)
  const [datasets, setDatasets] = useState<Array<{ name: string; r2_prefix: string; file_count: number }>>([])
  const [rows, setRows] = useState<DraftRow[]>([])
  const [datasetName, setDatasetName] = useState(getDefaultDatasetName())
  const [referenceAudioKey, setReferenceAudioKey] = useState('')
  const [referenceText, setReferenceText] = useState('')
  const [languageCode, setLanguageCode] = useState('ko')

  const [loading, setLoading] = useState(true)
  const [busyAction, setBusyAction] = useState<'refresh' | 'transcribe' | 'review' | 'create' | 'select' | 'auto' | ''>('')
  const [error, setError] = useState('')
  const [message, setMessage] = useState('')
  const [autoProgress, setAutoProgress] = useState<AutoPrepareProgress | null>(null)
  const autoStartedRef = useRef(false)

  const selectedRows = rows.filter((row) => row.selected)
  const currentDatasetName = inferCurrentDatasetName(voice)
  const autoPrepareRequested = searchParams.get('autoPrepare') === '1'
  const openTrainingWhenReady = searchParams.get('openTraining') === '1'
  const partialUploadWarning = searchParams.get('uploadWarning') === '1'
  const useRawTrainingPipeline = shouldUseRawTrainingPipeline(rows)

  async function loadData() {
    if (!voiceId) return
    setLoading(true)
    setError('')
    try {
      const [voiceData, datasetsData, rawFilesData] = await Promise.all([
        fetchVoice(voiceId),
        fetchVoiceDatasets(voiceId),
        fetchVoiceRawDatasetFiles(voiceId, 1000),
      ])
      setVoice(voiceData)
      setDatasets(datasetsData.datasets)
      setRows((previous) => mergeDraftRows(rawFilesData.files, previous))
      setLanguageCode(voiceData.labels.language ?? 'ko')
      setReferenceAudioKey((previous) => {
        if (previous && rawFilesData.files.some((file) => file.key === previous)) {
          return previous
        }
        if (rawFilesData.files.some((file) => file.key === voiceData.ref_audio_r2_key)) {
          return voiceData.ref_audio_r2_key ?? ''
        }
        return rawFilesData.files[0]?.key ?? ''
      })
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load dataset studio')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadData()
  }, [voiceId])

  useEffect(() => {
    if (!partialUploadWarning) return
    setMessage('Some files uploaded before the request failed. Review the raw uploads below and add the remaining files from this draft voice if needed.')
  }, [partialUploadWarning])

  useEffect(() => {
    if (!referenceAudioKey && selectedRows.length > 0) {
      setReferenceAudioKey(selectedRows[0].key)
    }
  }, [referenceAudioKey, selectedRows])

  function updateRow(key: string, updates: Partial<DraftRow>) {
    setRows((previous) =>
      previous.map((row) => (row.key === key ? { ...row, ...updates } : row)),
    )
  }

  async function handleRetranscribeSelected() {
    if (!voiceId || selectedRows.length === 0) return
    setBusyAction('transcribe')
    setError('')
    setMessage('')
    try {
      const response = await retranscribeDatasetEntries(
        voiceId,
        selectedRows.map((row) => ({
          audio_r2_key: row.key,
          text: row.text,
        })),
        languageCode,
      )
      setRows((previous) =>
        previous.map((row) => {
          const result = response.results.find((value) => value.audio_r2_key === row.key)
          if (!result || result.error) return row
          return {
            ...row,
            text: result.asr_text?.trim() || row.text,
            asrScore: typeof result.asr_score === 'number' ? result.asr_score : null,
            provider: result.provider ?? row.provider,
          }
        }),
      )
      setMessage(`Retranscribed ${response.results.filter((item) => !item.error).length} file(s).`)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to retranscribe files')
    } finally {
      setBusyAction('')
    }
  }

  async function handleReviewSelected() {
    if (!voiceId || selectedRows.length === 0) return
    setBusyAction('review')
    setError('')
    setMessage('')
    try {
      const response = await reviewDatasetTexts(
        voiceId,
        selectedRows.map((row) => ({
          segment: row.filename,
          text: row.text,
        })),
      )
      setRows((previous) =>
        previous.map((row) => {
          const index = selectedRows.findIndex((value) => value.key === row.key)
          const result = index >= 0 ? response.results[index] : null
          if (!result) return row
          return {
            ...row,
            text: result.corrected?.trim() || row.text,
            reviewScore: result.score,
            reviewIssues: result.issues ?? [],
            provider: response.provider,
          }
        }),
      )
      setMessage(`Reviewed ${response.results.length} transcript(s) with ${response.provider}.`)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to review transcripts')
    } finally {
      setBusyAction('')
    }
  }

  async function handleCreateDataset() {
    if (!voiceId || selectedRows.length === 0 || !referenceAudioKey || !datasetName.trim()) return
    setBusyAction('create')
    setError('')
    setMessage('')
    try {
      const refRow = rows.find((row) => row.key === referenceAudioKey)
      const response = await createFinalizedDataset(voiceId, {
        dataset_name: datasetName.trim(),
        items: selectedRows.map((row) => ({
          audio_r2_key: row.key,
          text: row.text.trim(),
        })).filter((row) => row.text),
        ref_audio_r2_key: referenceAudioKey,
        ref_text: referenceText.trim() || refRow?.text.trim() || undefined,
      })
      setMessage(`Created dataset ${response.dataset_name} with ${response.items_count} item(s).`)
      await loadData()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create finalized dataset')
    } finally {
      setBusyAction('')
    }
  }

  async function handleSelectDataset(name: string) {
    if (!voiceId) return
    setBusyAction('select')
    setError('')
    setMessage('')
    try {
      await selectVoiceDataset(voiceId, name)
      setMessage(`Set ${name} as the active dataset for future training.`)
      await loadData()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to activate dataset')
    } finally {
      setBusyAction('')
    }
  }

  async function runAutoPrepare(options?: { navigateToTraining?: boolean }) {
    if (!voiceId || rows.length === 0 || !datasetName.trim()) return

    setBusyAction('auto')
    setError('')
    setMessage('')
    setAutoProgress(null)

    try {
      const working = new Map(
        rows.map((row) => [row.key, { ...row, selected: autoPrepareRequested ? true : row.selected }]),
      )
      const selected = [...working.values()].filter((row) => row.selected)
      if (selected.length === 0) {
        throw new Error('Select at least one audio clip before auto-prepare.')
      }

      if (autoPrepareRequested) {
        setRows((previous) => previous.map((row) => ({ ...row, selected: true })))
      }

      const transcribeBatches = chunkArray(selected, 25)
      let transcribedCount = 0
      for (const batch of transcribeBatches) {
        setAutoProgress({
          phase: 'transcribe',
          completed: transcribedCount,
          total: selected.length,
          detail: `Auto-transcribing ${transcribedCount + 1}-${Math.min(selected.length, transcribedCount + batch.length)} of ${selected.length}`,
        })
        const response = await retranscribeDatasetEntries(
          voiceId,
          batch.map((row) => ({
            audio_r2_key: row.key,
            text: row.text,
          })),
          languageCode,
        )
        const resultMap = new Map(response.results.map((result) => [result.audio_r2_key, result]))
        for (const row of batch) {
          const existing = working.get(row.key)
          const result = resultMap.get(row.key)
          if (!existing || !result || result.error) continue
          working.set(row.key, {
            ...existing,
            text: result.asr_text?.trim() || existing.text,
            asrScore: typeof result.asr_score === 'number' ? result.asr_score : existing.asrScore,
            provider: result.provider ?? existing.provider,
          })
        }
        setRows((previous) =>
          previous.map((row) => {
            const updated = working.get(row.key)
            return updated ?? row
          }),
        )
        transcribedCount += batch.length
      }

      const reviewRows = [...working.values()].filter((row) => row.selected && row.text.trim())
      const reviewBatches = chunkArray(reviewRows, 50)
      let reviewedCount = 0
      for (const batch of reviewBatches) {
        setAutoProgress({
          phase: 'review',
          completed: reviewedCount,
          total: reviewRows.length,
          detail: `Reviewing transcripts ${reviewedCount + 1}-${Math.min(reviewRows.length, reviewedCount + batch.length)} of ${reviewRows.length}`,
        })
        const response = await reviewDatasetTexts(
          voiceId,
          batch.map((row) => ({
            segment: row.filename,
            text: row.text,
          })),
        )
        batch.forEach((row, index) => {
          const existing = working.get(row.key)
          const result = response.results[index]
          if (!existing || !result) return
          working.set(row.key, {
            ...existing,
            text: result.corrected?.trim() || existing.text,
            reviewScore: result.score,
            reviewIssues: result.issues ?? [],
            provider: response.provider,
          })
        })
        setRows((previous) =>
          previous.map((row) => {
            const updated = working.get(row.key)
            return updated ?? row
          }),
        )
        reviewedCount += batch.length
      }

      const finalizedRows = [...working.values()]
        .filter((row) => row.selected)
        .map((row) => ({
          audio_r2_key: row.key,
          text: row.text.trim(),
        }))
        .filter((row) => row.text)

      if (finalizedRows.length === 0) {
        throw new Error('No usable transcript was produced. Check the uploaded clips and try again.')
      }

      const resolvedReferenceAudioKey =
        referenceAudioKey && working.has(referenceAudioKey)
          ? referenceAudioKey
          : finalizedRows[0]?.audio_r2_key ?? ''
      if (!resolvedReferenceAudioKey) {
        throw new Error('A reference clip is required before the dataset can be finalized.')
      }

      const refRow = working.get(resolvedReferenceAudioKey)

      setAutoProgress({
        phase: 'create',
        completed: finalizedRows.length,
        total: finalizedRows.length,
        detail: `Creating finalized dataset ${datasetName.trim()}`,
      })
      const createResponse = await createFinalizedDataset(voiceId, {
        dataset_name: datasetName.trim(),
        items: finalizedRows,
        ref_audio_r2_key: resolvedReferenceAudioKey,
        ref_text: referenceText.trim() || refRow?.text.trim() || undefined,
      })

      setAutoProgress({
        phase: 'select',
        completed: 1,
        total: 1,
        detail: `Setting ${createResponse.dataset_name} as active dataset`,
      })
      await selectVoiceDataset(voiceId, createResponse.dataset_name)
      await loadData()

      setAutoProgress({
        phase: 'done',
        completed: finalizedRows.length,
        total: finalizedRows.length,
        detail: `Dataset ${createResponse.dataset_name} is ready`,
      })
      setMessage(
        `Auto-prepared ${createResponse.dataset_name} with ${createResponse.items_count} item(s) and set it as the active dataset.`,
      )

      if (options?.navigateToTraining) {
        navigate(encodeTrainingQuery(voiceId, createResponse.dataset_name))
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Auto prepare failed')
    } finally {
      setBusyAction('')
    }
  }

  async function runRawTrainingAutoStart(options?: { navigateToTraining?: boolean }) {
    if (!voiceId || !voice) return

    setBusyAction('auto')
    setError('')
    setMessage('')
    setAutoProgress({
      phase: 'start',
      completed: 0,
      total: 1,
      detail: 'Large raw upload detected. Starting preprocessing and training directly from the raw media.',
    })

    try {
      await startTraining(voiceId, {})
      setAutoProgress({
        phase: 'done',
        completed: 1,
        total: 1,
        detail: 'Training job started. Segmentation and transcription will run on the training worker.',
      })
      setMessage('Started training directly from the raw upload. Segmentation, transcription, filtering, and reference selection now run automatically during preprocessing.')
      if (options?.navigateToTraining) {
        navigate(encodeTrainingQuery(voiceId))
      }
    } catch (err) {
      if (err instanceof ApiError && err.status === 409) {
        setAutoProgress({
          phase: 'done',
          completed: 1,
          total: 1,
          detail: 'A training job is already active for this voice.',
        })
        setMessage('A training job is already active for this voice. Following the existing raw-media preprocessing run.')
        if (options?.navigateToTraining) {
          navigate(encodeTrainingQuery(voiceId))
        }
        return
      }
      setError(err instanceof Error ? err.message : 'Failed to start training from raw uploads')
    } finally {
      setBusyAction('')
    }
  }

  useEffect(() => {
    if (loading || rows.length === 0 || !autoPrepareRequested || autoStartedRef.current) {
      return
    }
    autoStartedRef.current = true
    if (useRawTrainingPipeline) {
      void runRawTrainingAutoStart({ navigateToTraining: openTrainingWhenReady })
      return
    }
    void runAutoPrepare({ navigateToTraining: openTrainingWhenReady })
  }, [loading, rows.length, autoPrepareRequested, openTrainingWhenReady, useRawTrainingPipeline])

  if (loading) {
    return (
      <div className="space-y-6 animate-pulse">
        <div className="h-7 bg-raised rounded w-48" />
        <div className="grid lg:grid-cols-[340px_1fr] gap-6">
          <div className="h-80 bg-raised rounded-xl" />
          <div className="h-[32rem] bg-raised rounded-xl" />
        </div>
      </div>
    )
  }

  if (!voice) {
    return (
      <div className="text-center py-16 space-y-4">
        <div className="text-error text-sm">{error || 'Voice not found'}</div>
        <button
          onClick={() => navigate('/voices')}
          className="text-accent text-sm hover:text-accent-light"
          type="button"
        >
          Back
        </button>
      </div>
    )
  }

  return (
    <div className="space-y-8">
      <button
        onClick={() => navigate(`/voices/${voice.voice_id}`)}
        className="text-subtle text-sm hover:text-accent transition-colors inline-flex items-center gap-1"
        type="button"
      >
        <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
          <polyline points="15 18 9 12 15 6" />
        </svg>
        Back to Voice
      </button>

      <div className="flex flex-col gap-2">
        <h1 className="text-heading text-2xl font-bold">Dataset Studio</h1>
        <p className="text-subtle text-sm">
          {voice.name} · review raw uploads, clean transcripts, choose a reference clip, and create the finalized dataset used for training.
        </p>
        <div className="flex flex-wrap gap-3 text-[11px] font-mono text-muted">
          <span>raw_files={rows.length}</span>
          <span>selected={selectedRows.length}</span>
          <span>finalized={datasets.length}</span>
          <span>active_dataset={currentDatasetName ?? 'none'}</span>
        </div>
      </div>

      {error && (
        <div className="rounded-lg border border-error/20 bg-error-dim px-4 py-3 text-error text-sm">
          {error}
        </div>
      )}

      {message && (
        <div className="rounded-lg border border-accent/20 bg-accent-dim px-4 py-3 text-accent text-sm">
          {message}
        </div>
      )}

      {autoProgress && (
        <div className="rounded-lg border border-accent/20 bg-surface px-4 py-3">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <div className="text-primary text-sm font-semibold">Auto Prepare Progress</div>
              <div className="mt-1 text-subtle text-xs">{autoProgress.detail}</div>
            </div>
            <div className="text-[11px] font-mono text-muted">
              {autoProgress.phase} {autoProgress.completed}/{autoProgress.total}
            </div>
          </div>
        </div>
      )}

      {useRawTrainingPipeline && (
        <div className="rounded-lg border border-accent/20 bg-accent-dim px-4 py-3 text-sm text-accent">
          Large raw media was detected. This voice should skip clip-by-clip browser transcription and start training directly from the uploaded source so segmentation and ASR run on the training worker.
        </div>
      )}

      <div className="grid lg:grid-cols-[340px_1fr] gap-6">
        <div className="space-y-6">
          <div className="bg-raised border border-edge rounded-xl p-5 space-y-4">
            <div className="flex items-center justify-between gap-3">
              <div>
                <h2 className="text-heading font-semibold text-sm">Finalized Datasets</h2>
                <p className="text-subtle text-xs mt-1">Training should use one of these finalized datasets, not the raw upload bucket.</p>
              </div>
              <button
                onClick={loadData}
                disabled={busyAction === 'refresh'}
                className="inline-flex items-center rounded-lg border border-edge px-3 py-2 text-[11px] font-semibold text-primary transition-colors hover:border-accent hover:text-accent disabled:opacity-50"
                type="button"
              >
                Refresh
              </button>
            </div>

            {datasets.length === 0 ? (
              <div className="rounded-lg border border-dashed border-edge bg-surface px-4 py-6 text-center text-sm text-subtle">
                No finalized dataset yet.
              </div>
            ) : (
              <div className="space-y-3">
                {datasets.map((dataset) => {
                  const isActive = dataset.name === currentDatasetName
                  return (
                    <div key={dataset.name} className="rounded-lg border border-edge bg-surface px-3 py-3">
                      <div className="flex items-center justify-between gap-3">
                        <div>
                          <div className="text-primary text-sm font-semibold">{dataset.name}</div>
                          <div className="mt-1 text-[10px] font-mono text-muted">
                            files={dataset.file_count}
                          </div>
                        </div>
                        <button
                          onClick={() => handleSelectDataset(dataset.name)}
                          disabled={busyAction === 'select' || isActive}
                          className="inline-flex items-center rounded-lg border border-edge px-3 py-2 text-[11px] font-semibold text-primary transition-colors hover:border-accent hover:text-accent disabled:opacity-50"
                          type="button"
                        >
                          {isActive ? 'Active' : 'Set Active'}
                        </button>
                      </div>
                    </div>
                  )
                })}
              </div>
            )}
          </div>

          <div className="bg-raised border border-edge rounded-xl p-5 space-y-4">
            <div>
              <h2 className="text-heading font-semibold text-sm">Finalize New Dataset</h2>
              <p className="text-subtle text-xs mt-1">
                {useRawTrainingPipeline
                  ? 'Large raw uploads should go straight to training. The training worker will segment, transcribe, filter, and choose the reference clip automatically.'
                  : 'Choose the raw clips you want, correct their transcripts, and create a clean dataset for the next training run.'}
              </p>
            </div>

            <div>
              <label className="text-subtle text-xs font-medium mb-1.5 block">Dataset Name</label>
              <input
                value={datasetName}
                onChange={(e) => setDatasetName(e.target.value)}
                className="w-full bg-surface border border-edge rounded-lg px-3 py-2 text-sm text-primary font-mono focus:border-accent transition-colors"
              />
            </div>

            <div>
              <label className="text-subtle text-xs font-medium mb-1.5 block">Language</label>
              <select
                value={languageCode}
                onChange={(e) => setLanguageCode(e.target.value)}
                className="w-full bg-surface border border-edge rounded-lg px-3 py-2 text-sm text-primary focus:border-accent transition-colors"
              >
                <option value="ko">Korean</option>
                <option value="en">English</option>
                <option value="ja">Japanese</option>
                <option value="zh">Chinese</option>
              </select>
            </div>

            <div>
              <label className="text-subtle text-xs font-medium mb-1.5 block">Reference Text</label>
              <textarea
                value={referenceText}
                onChange={(e) => setReferenceText(e.target.value)}
                rows={3}
                placeholder="Leave empty to use the selected reference clip transcript."
                className="w-full bg-surface border border-edge rounded-lg px-3 py-2 text-sm text-primary placeholder:text-muted focus:border-accent transition-colors resize-none"
              />
            </div>

            <div className="flex flex-wrap gap-2">
              <button
                onClick={handleRetranscribeSelected}
                disabled={useRawTrainingPipeline || busyAction !== '' || selectedRows.length === 0}
                className="inline-flex items-center rounded-lg border border-edge px-3 py-2 text-[11px] font-semibold text-primary transition-colors hover:border-accent hover:text-accent disabled:opacity-50"
                type="button"
              >
                Auto Transcribe Selected
              </button>
              <button
                onClick={handleReviewSelected}
                disabled={useRawTrainingPipeline || busyAction !== '' || selectedRows.length === 0}
                className="inline-flex items-center rounded-lg border border-edge px-3 py-2 text-[11px] font-semibold text-primary transition-colors hover:border-accent hover:text-accent disabled:opacity-50"
                type="button"
              >
                Review Selected Texts
              </button>
              <button
                onClick={() => {
                  if (useRawTrainingPipeline) {
                    void runRawTrainingAutoStart({ navigateToTraining: true })
                    return
                  }
                  void runAutoPrepare()
                }}
                disabled={busyAction !== '' || selectedRows.length === 0 || (!useRawTrainingPipeline && !datasetName.trim())}
                className="inline-flex items-center rounded-lg border border-accent/40 bg-accent-dim px-3 py-2 text-[11px] font-semibold text-accent transition-colors hover:border-accent hover:text-accent-light disabled:opacity-50"
                type="button"
              >
                {useRawTrainingPipeline ? 'Start Training From Raw Uploads' : 'Auto Prepare Selected'}
              </button>
              <button
                onClick={() => setRows((previous) => previous.map((row) => ({ ...row, selected: true })))}
                disabled={rows.length === 0}
                className="inline-flex items-center rounded-lg border border-edge px-3 py-2 text-[11px] font-semibold text-muted transition-colors hover:text-primary"
                type="button"
              >
                Select All
              </button>
              <button
                onClick={() => setRows((previous) => previous.map((row) => ({ ...row, selected: false })))}
                disabled={rows.length === 0}
                className="inline-flex items-center rounded-lg border border-edge px-3 py-2 text-[11px] font-semibold text-muted transition-colors hover:text-primary"
                type="button"
              >
                Clear
              </button>
            </div>

            <button
              onClick={handleCreateDataset}
              disabled={useRawTrainingPipeline || busyAction !== '' || selectedRows.length === 0 || !referenceAudioKey || !datasetName.trim()}
              className="w-full bg-accent hover:bg-accent-light text-void font-semibold text-sm py-2.5 rounded-lg disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
              type="button"
            >
              {busyAction === 'create' ? 'Creating Dataset…' : 'Create Finalized Dataset'}
            </button>

            <button
              onClick={() => {
                const targetDataset = currentDatasetName ?? datasetName.trim()
                if (!targetDataset) return
                navigate(encodeTrainingQuery(voice.voice_id, targetDataset))
              }}
              disabled={!currentDatasetName && !datasetName.trim()}
              className="w-full rounded-lg border border-edge px-3 py-2 text-sm font-semibold text-primary transition-colors hover:border-accent hover:text-accent disabled:opacity-40"
              type="button"
            >
              Open Training With Recommended Setup
            </button>
          </div>
        </div>

        <div className="bg-raised border border-edge rounded-xl p-5">
          <div className="flex items-center justify-between gap-4 mb-4">
            <div>
              <h2 className="text-heading font-semibold text-sm">Raw Uploads</h2>
              <p className="text-subtle text-xs mt-1">Pick the clips that belong in the finalized dataset and edit the transcript directly here.</p>
            </div>
            <div className="text-[10px] font-mono text-muted">
              ref={referenceAudioKey ? rows.find((row) => row.key === referenceAudioKey)?.filename ?? 'selected' : 'none'}
            </div>
          </div>

          {rows.length === 0 ? (
            <div className="rounded-xl border border-dashed border-edge bg-surface px-4 py-12 text-center">
              <div className="text-primary text-sm font-semibold">No raw uploads found</div>
              <p className="mt-2 text-subtle text-sm">
                Upload WAV files on the Voices page first, then return here to finalize the dataset.
              </p>
            </div>
          ) : (
            <div className="space-y-3 max-h-[70vh] overflow-y-auto pr-1">
              {rows.map((row) => (
                <div key={row.key} className="rounded-xl border border-edge bg-surface p-4">
                  <div className="flex items-start justify-between gap-3">
                    <div className="min-w-0">
                      <div className="flex flex-wrap items-center gap-2">
                        <label className="inline-flex items-center gap-2 text-[11px] font-mono text-muted">
                          <input
                            type="checkbox"
                            checked={row.selected}
                            onChange={(e) => updateRow(row.key, { selected: e.target.checked })}
                            className="accent-accent"
                          />
                          include
                        </label>
                        <label className="inline-flex items-center gap-2 text-[11px] font-mono text-muted">
                          <input
                            type="radio"
                            name="dataset-ref-audio"
                            checked={referenceAudioKey === row.key}
                            onChange={() => setReferenceAudioKey(row.key)}
                            className="accent-accent"
                          />
                          reference
                        </label>
                      </div>
                      <div className="mt-2 text-primary text-sm font-semibold break-all">{row.filename}</div>
                      <div className="mt-1 flex flex-wrap gap-2 text-[10px] font-mono text-muted">
                        <span>{formatDateTime(row.uploaded)}</span>
                        <span>{(row.size / 1024 / 1024).toFixed(1)} MB</span>
                        {row.asrScore !== null && <span>asr={row.asrScore.toFixed(3)}</span>}
                        {row.reviewScore !== null && <span>review={row.reviewScore}/5</span>}
                        {row.provider && <span>{row.provider}</span>}
                      </div>
                    </div>
                  </div>

                  <textarea
                    value={row.text}
                    onChange={(e) => updateRow(row.key, { text: e.target.value })}
                    rows={3}
                    placeholder="Transcript for this clip"
                    className="mt-3 w-full bg-raised border border-edge rounded-lg px-3 py-2 text-sm text-primary placeholder:text-muted focus:border-accent transition-colors resize-none"
                  />

                  {row.reviewIssues.length > 0 && (
                    <div className="mt-3 flex flex-wrap gap-2 text-[10px] font-mono">
                      {row.reviewIssues.map((issue) => (
                        <span key={issue} className="rounded-full bg-warning-dim px-2 py-0.5 text-warning">
                          {issue}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
