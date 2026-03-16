import { useCallback, useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router'
import {
  fetchTrainingCheckoutLedger,
  fetchTrainingLogChunkText,
  fetchTrainingLogs,
  fetchTrainingPreprocessCache,
  formatDateTime,
  formatDurationMs,
  formatTime,
  type DatasetPreprocessCacheEntry,
  type TrainingCheckoutLedgerEntry,
  type TrainingJob,
  type TrainingLogChunk,
  type TrainingPreprocessCacheResponse,
  updateTrainingPreprocessCache,
  updateTrainingPreprocessEntry,
} from '../../lib/api'
import {
  getTrainingCheckoutSearch,
  getTrainingJobDisplayStatus,
  isActiveTrainingJobStatus,
  needsTrainingValidationFollowup,
  shouldWatchTrainingJob,
} from '../../lib/trainingCheckout'

type Props = {
  job: TrainingJob
  voiceName?: string
  compact?: boolean
  onCancel?: (jobId: string) => Promise<void> | void
  onRefresh?: (jobId: string) => Promise<void>
  onReconcile?: (jobId: string) => Promise<void>
  onRevalidate?: (jobId: string) => Promise<void>
}

function readTimestamp(value: unknown): number | null {
  if (typeof value === 'number' && Number.isFinite(value)) return value
  if (typeof value === 'string' && value.trim()) {
    const numeric = Number(value)
    if (Number.isFinite(numeric)) return numeric
    const parsed = Date.parse(value)
    return Number.isFinite(parsed) ? parsed : null
  }
  return null
}

function progressPercent(job: TrainingJob): number {
  const step = Number(job.progress.step)
  const totalSteps = Number(job.progress.total_steps)
  if (Number.isFinite(step) && Number.isFinite(totalSteps) && totalSteps > 0) {
    return Math.max(0, Math.min(100, (step / totalSteps) * 100))
  }
  const epoch = Number(job.progress.epoch)
  const totalEpochs = Number(job.progress.total_epochs)
  if (Number.isFinite(epoch) && Number.isFinite(totalEpochs) && totalEpochs > 0) {
    return Math.max(0, Math.min(100, (epoch / totalEpochs) * 100))
  }
  return 0
}

function getStatusClasses(statusLabel: string) {
  const map: Record<string, string> = {
    pending: 'bg-warning-dim text-warning',
    queued: 'bg-warning-dim text-warning',
    provisioning: 'bg-warning-dim text-warning',
    running: 'bg-accent-dim text-accent',
    downloading: 'bg-accent-dim text-accent',
    preprocessing: 'bg-accent-dim text-accent',
    preparing: 'bg-accent-dim text-accent',
    training: 'bg-accent-dim text-accent',
    uploading: 'bg-accent-dim text-accent',
    completed: 'bg-accent-dim text-accent',
    validating: 'bg-accent-dim text-accent',
    promoted: 'bg-accent-dim text-accent',
    manual_promoted: 'bg-accent-dim text-accent',
    candidate_ready: 'bg-warning-dim text-warning',
    kept_current: 'bg-raised text-muted',
    rejected: 'bg-warning-dim text-warning',
    failed: 'bg-error-dim text-error',
    cancelled: 'bg-raised text-muted',
  }
  return map[statusLabel] ?? map.pending
}

function formatBytes(value: number): string {
  if (!Number.isFinite(value) || value <= 0) return '0 B'
  const units = ['B', 'KB', 'MB', 'GB']
  let size = value
  let unitIndex = 0
  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024
    unitIndex += 1
  }
  return `${size >= 10 || unitIndex === 0 ? size.toFixed(0) : size.toFixed(1)} ${units[unitIndex]}`
}

function getPathLabel(path: string): string {
  const parts = path.split('/').filter(Boolean)
  return parts[parts.length - 1] ?? path
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-muted text-[10px] font-mono uppercase tracking-wider">{label}</div>
      <div className="text-primary text-sm font-mono tabular-nums">{value}</div>
    </div>
  )
}

export function TrainingJobRow({
  job,
  voiceName,
  compact = false,
  onCancel,
  onRefresh,
  onReconcile,
  onRevalidate,
}: Props) {
  const [busyAction, setBusyAction] = useState<'refresh' | 'reconcile' | 'revalidate' | ''>('')
  const [actionError, setActionError] = useState('')
  const [showLogs, setShowLogs] = useState(false)

  const checkout = getTrainingCheckoutSearch(job)
  const statusLabel = getTrainingJobDisplayStatus(job)
  const statusClass = getStatusClasses(statusLabel)
  const isActive = isActiveTrainingJobStatus(job.status)
  const isValidationPending = needsTrainingValidationFollowup(job)
  const isPolling = isActive || isValidationPending
  const progressPct = progressPercent(job)

  const startedAt = readTimestamp(job.started_at) ?? readTimestamp(job.created_at)
  const finishedAt = readTimestamp(job.completed_at) ?? readTimestamp(job.summary?.completed_at)
  const now = Date.now()
  const durationMs = useMemo(() => {
    const summaryDuration = Number(job.summary?.duration_ms)
    if (Number.isFinite(summaryDuration) && summaryDuration >= 0) return summaryDuration
    if (!startedAt) return null
    return Math.max(0, (finishedAt ?? now) - startedAt)
  }, [job.summary?.duration_ms, startedAt, finishedAt, now])

  async function runAction(kind: 'refresh' | 'reconcile' | 'revalidate', action?: () => Promise<void>) {
    if (!action) return
    setBusyAction(kind)
    setActionError('')
    try {
      await action()
    } catch (error) {
      setActionError(error instanceof Error ? error.message : 'Action failed')
    } finally {
      setBusyAction('')
    }
  }

  return (
    <article className={`rounded-lg border border-edge bg-surface ${compact ? 'p-3' : 'p-4'}`}>
      <div className="flex items-start justify-between gap-3">
        <div>
          <Link to={`/voices/${job.voice_id}/training`} className="text-primary text-sm font-semibold hover:text-accent transition-colors">{voiceName ?? job.voice_id}</Link>
          <div className="mt-1 flex flex-wrap gap-x-3 gap-y-1 text-[10px] font-mono text-muted">
            <span>{job.job_id.slice(0, 12)}</span>
            <span>created={formatDateTime(job.created_at)}</span>
            {startedAt !== null && <span>started={formatTime(startedAt)}</span>}
            {durationMs !== null && <span>duration={formatDurationMs(durationMs)}</span>}
          </div>
        </div>
        <div className="flex items-center gap-2">
          <span className={`rounded-full px-2 py-0.5 text-[10px] font-mono uppercase tracking-wider ${statusClass}`}>
            {statusLabel}
          </span>
          {isActive && onCancel && (
            <button
              type="button"
              onClick={() => { void onCancel(job.job_id) }}
              className="text-xs text-muted hover:text-error"
            >
              Cancel
            </button>
          )}
        </div>
      </div>

      {isPolling && (
        <div className="mt-3 h-2 overflow-hidden rounded-full bg-edge">
          <div className="h-full bg-accent transition-all duration-500" style={{ width: `${isValidationPending ? 100 : progressPct}%` }} />
        </div>
      )}

      {!compact && (
        <>
          <div className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-4">
            <Metric label="Epoch" value={`${Number(job.progress.epoch) || 0}/${Number(job.progress.total_epochs) || '—'}`} />
            <Metric label="Step" value={`${Number(job.progress.step) || 0}`} />
            <Metric label="Loss" value={typeof job.progress.loss === 'number' ? job.progress.loss.toFixed(4) : '—'} />
            <Metric
              label="Score"
              value={typeof checkout.champion?.score === 'number' ? checkout.champion.score.toFixed(3) : '—'}
            />
          </div>

          {(checkout.message || checkout.last_message) && (
            <div className="mt-3 rounded-lg border border-edge px-3 py-2 text-[11px] text-subtle">
              {checkout.message ?? checkout.last_message}
            </div>
          )}
        </>
      )}

      <div className="mt-3 flex flex-wrap gap-2">
        {onRefresh && (
          <button
            type="button"
            onClick={() => { void runAction('refresh', () => onRefresh(job.job_id)) }}
            disabled={busyAction !== ''}
            className="rounded-lg border border-edge px-2.5 py-1.5 text-[11px] font-semibold text-primary hover:border-accent hover:text-accent disabled:opacity-50"
          >
            {busyAction === 'refresh' ? 'Refreshing...' : 'Refresh'}
          </button>
        )}
        {onReconcile && isPolling && (
          <button
            type="button"
            onClick={() => { void runAction('reconcile', () => onReconcile(job.job_id)) }}
            disabled={busyAction !== ''}
            className="rounded-lg border border-edge px-2.5 py-1.5 text-[11px] font-semibold text-primary hover:border-accent hover:text-accent disabled:opacity-50"
          >
            {busyAction === 'reconcile' ? 'Advancing...' : 'Advance Status'}
          </button>
        )}
        {onRevalidate && (job.status === 'completed' || job.status === 'failed') && (
          <button
            type="button"
            onClick={() => { void runAction('revalidate', () => onRevalidate(job.job_id)) }}
            disabled={busyAction !== ''}
            className="rounded-lg border border-edge px-2.5 py-1.5 text-[11px] font-semibold text-primary hover:border-accent hover:text-accent disabled:opacity-50"
          >
            {busyAction === 'revalidate' ? 'Restarting...' : 'Revalidate'}
          </button>
        )}
        <button
          type="button"
          onClick={() => setShowLogs(true)}
          className="rounded-lg border border-edge px-2.5 py-1.5 text-[11px] font-semibold text-primary hover:border-accent hover:text-accent"
        >
          Inspect Run
        </button>
        <Link
          to={`/voices/${job.voice_id}/compare?job=${job.job_id}`}
          className="rounded-lg border border-edge px-2.5 py-1.5 text-[11px] font-semibold text-primary hover:border-accent hover:text-accent"
        >
          Compare
        </Link>
        <Link
          to={`/voices/${job.voice_id}/generate`}
          className="rounded-lg border border-edge px-2.5 py-1.5 text-[11px] font-semibold text-primary hover:border-accent hover:text-accent"
        >
          Open Voice
        </Link>
      </div>

      {actionError && (
        <div className="mt-3 rounded-lg border border-error/20 bg-error-dim px-3 py-2 text-xs text-error">
          {actionError}
        </div>
      )}

      {showLogs && <JobLogsModal job={job} onClose={() => setShowLogs(false)} />}
    </article>
  )
}

function JobLogsModal({ job, onClose }: { job: TrainingJob; onClose: () => void }) {
  const [activeTab, setActiveTab] = useState<'logs' | 'checkout' | 'transcripts'>('logs')
  const [chunks, setChunks] = useState<TrainingLogChunk[]>([])
  const [selectedSeq, setSelectedSeq] = useState<number | null>(null)
  const [content, setContent] = useState('')
  const [loadingLogs, setLoadingLogs] = useState(true)
  const [logError, setLogError] = useState('')

  const [checkoutLedger, setCheckoutLedger] = useState<TrainingCheckoutLedgerEntry[]>([])
  const [loadingLedger, setLoadingLedger] = useState(true)
  const [ledgerError, setLedgerError] = useState('')

  const [preprocess, setPreprocess] = useState<TrainingPreprocessCacheResponse | null>(null)
  const [loadingPreprocess, setLoadingPreprocess] = useState(true)
  const [preprocessError, setPreprocessError] = useState('')
  const [selectedEntryId, setSelectedEntryId] = useState<string | null>(null)
  const [entryDraft, setEntryDraft] = useState('')
  const [entryIncluded, setEntryIncluded] = useState(true)
  const [referenceDraft, setReferenceDraft] = useState('')
  const [transcriptQuery, setTranscriptQuery] = useState('')
  const [saving, setSaving] = useState<'entry' | 'reference' | ''>('')
  const [saveMessage, setSaveMessage] = useState('')

  const loadLogs = useCallback(async (options?: { keepSelection?: boolean }) => {
    setLoadingLogs(true)
    setLogError('')
    try {
      const response = await fetchTrainingLogs(job.job_id, 20)
      setChunks(response.chunks)
      const targetSeq =
        options?.keepSelection && selectedSeq !== null && response.chunks.some((chunk) => chunk.seq === selectedSeq)
          ? selectedSeq
          : response.chunks[0]?.seq ?? null
      setSelectedSeq(targetSeq)
      if (targetSeq !== null) {
        setContent(await fetchTrainingLogChunkText(job.job_id, targetSeq))
      } else {
        setContent('')
      }
    } catch (error) {
      setLogError(error instanceof Error ? error.message : 'Failed to load logs')
    } finally {
      setLoadingLogs(false)
    }
  }, [job.job_id, selectedSeq])

  const loadPreprocess = useCallback(async (options?: { keepSelection?: boolean }) => {
    setLoadingPreprocess(true)
    setPreprocessError('')
    try {
      const response = await fetchTrainingPreprocessCache(job.job_id)
      setPreprocess(response)
      const keepSelection =
        options?.keepSelection &&
        selectedEntryId !== null &&
        response.entries.some((entry) => entry.entry_id === selectedEntryId)
      const targetEntry = keepSelection
        ? response.entries.find((entry) => entry.entry_id === selectedEntryId) ?? null
        : response.entries[0] ?? null
      setSelectedEntryId(targetEntry?.entry_id ?? null)
      setReferenceDraft(response.reference_text ?? '')
    } catch (error) {
      setPreprocessError(error instanceof Error ? error.message : 'Failed to load cached transcripts')
    } finally {
      setLoadingPreprocess(false)
    }
  }, [job.job_id, selectedEntryId])

  const loadCheckoutLedger = useCallback(async () => {
    setLoadingLedger(true)
    setLedgerError('')
    try {
      const response = await fetchTrainingCheckoutLedger(job.job_id)
      setCheckoutLedger(response.entries)
    } catch (error) {
      setLedgerError(error instanceof Error ? error.message : 'Failed to load checkout ledger')
    } finally {
      setLoadingLedger(false)
    }
  }, [job.job_id])

  useEffect(() => {
    void Promise.all([loadLogs(), loadPreprocess(), loadCheckoutLedger()])
  }, [loadCheckoutLedger, loadLogs, loadPreprocess])

  useEffect(() => {
    if (!shouldWatchTrainingJob(job)) return
    const interval = setInterval(() => {
      void loadLogs({ keepSelection: true })
      void loadPreprocess({ keepSelection: true })
      void loadCheckoutLedger()
    }, 10000)
    return () => clearInterval(interval)
  }, [job, loadCheckoutLedger, loadLogs, loadPreprocess])

  async function handleSelectSeq(seq: number) {
    setSelectedSeq(seq)
    setLoadingLogs(true)
    setLogError('')
    try {
      setContent(await fetchTrainingLogChunkText(job.job_id, seq))
    } catch (error) {
      setLogError(error instanceof Error ? error.message : 'Failed to load log chunk')
    } finally {
      setLoadingLogs(false)
    }
  }

  const entries = preprocess?.entries ?? []
  const includedEntries = entries.filter((entry) => entry.included)
  const filteredEntries = entries.filter((entry) => {
    const query = transcriptQuery.trim().toLowerCase()
    if (!query) return true
    return entry.text.toLowerCase().includes(query) || entry.audio_path.toLowerCase().includes(query)
  })
  const selectedEntry: DatasetPreprocessCacheEntry | null =
    entries.find((entry) => entry.entry_id === selectedEntryId) ?? filteredEntries[0] ?? null
  const totalLogBytes = chunks.reduce((sum, chunk) => sum + (chunk.bytes ?? 0), 0)
  const totalLogLines = chunks.reduce((sum, chunk) => sum + (chunk.lines ?? 0), 0)

  useEffect(() => {
    if (!selectedEntry) {
      setEntryDraft('')
      setEntryIncluded(true)
      return
    }
    setEntryDraft(selectedEntry.text)
    setEntryIncluded(selectedEntry.included)
  }, [selectedEntry])

  async function handleSaveEntry() {
    if (!selectedEntry) return
    setSaving('entry')
    setSaveMessage('')
    try {
      const response = await updateTrainingPreprocessEntry(job.job_id, selectedEntry.entry_id, {
        text: entryDraft,
        included: entryIncluded,
      })
      setPreprocess((previous) => (
        previous
          ? {
              ...previous,
              entries: previous.entries.map((entry) =>
                entry.entry_id === response.entry.entry_id ? response.entry : entry,
              ),
            }
          : previous
      ))
      setSaveMessage('Transcript entry saved to the reusable cache.')
    } catch (error) {
      setPreprocessError(error instanceof Error ? error.message : 'Failed to save transcript entry')
    } finally {
      setSaving('')
    }
  }

  async function handleSaveReference() {
    setSaving('reference')
    setSaveMessage('')
    try {
      const response = await updateTrainingPreprocessCache(job.job_id, {
        reference_text: referenceDraft,
      })
      setPreprocess((previous) => (
        previous
          ? {
              ...previous,
              reference_text: response.reference_text,
            }
          : previous
      ))
      setSaveMessage('Reference text updated.')
    } catch (error) {
      setPreprocessError(error instanceof Error ? error.message : 'Failed to save reference text')
    } finally {
      setSaving('')
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4">
      <div className="flex h-[88vh] w-full max-w-7xl flex-col rounded-2xl border border-edge bg-surface p-4">
        <div className="mb-4 flex items-start justify-between gap-4">
          <div>
            <div className="text-heading text-sm font-semibold">Run Inspector</div>
            <div className="text-[10px] font-mono text-muted">{job.job_id.slice(0, 12)}</div>
            <div className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-5">
              <Metric label="Status" value={job.status} />
              <Metric label="Logs" value={String(chunks.length)} />
              <Metric label="Entries" value={preprocess?.cache ? `${includedEntries.length}/${entries.length}` : '—'} />
              <Metric label="Ledger" value={String(checkoutLedger.length)} />
              <Metric label="Bytes" value={formatBytes(totalLogBytes)} />
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => {
                void loadLogs({ keepSelection: true })
                void loadPreprocess({ keepSelection: true })
                void loadCheckoutLedger()
              }}
              className="rounded-lg border border-edge px-3 py-2 text-[11px] font-semibold text-primary hover:border-accent hover:text-accent"
              type="button"
            >
              Refresh
            </button>
            <button onClick={onClose} className="text-muted hover:text-primary" type="button">Close</button>
          </div>
        </div>

        <div className="mb-4 flex gap-2">
          <button
            onClick={() => setActiveTab('logs')}
            className={`rounded-lg px-3 py-2 text-[11px] font-semibold ${activeTab === 'logs' ? 'bg-accent text-void' : 'border border-edge text-primary hover:border-accent hover:text-accent'}`}
            type="button"
          >
            Logs
          </button>
          <button
            onClick={() => setActiveTab('checkout')}
            className={`rounded-lg px-3 py-2 text-[11px] font-semibold ${activeTab === 'checkout' ? 'bg-accent text-void' : 'border border-edge text-primary hover:border-accent hover:text-accent'}`}
            type="button"
          >
            Checkout Ledger
          </button>
          <button
            onClick={() => setActiveTab('transcripts')}
            className={`rounded-lg px-3 py-2 text-[11px] font-semibold ${activeTab === 'transcripts' ? 'bg-accent text-void' : 'border border-edge text-primary hover:border-accent hover:text-accent'}`}
            type="button"
          >
            Cached Transcripts
          </button>
        </div>

        {activeTab === 'logs' ? (
          <div className="grid min-h-0 flex-1 grid-cols-[280px_1fr] gap-4">
            <div className="flex min-h-0 flex-col rounded-xl border border-edge bg-raised">
              <div className="border-b border-edge px-4 py-3">
                <div className="text-heading text-sm font-semibold">Log Index</div>
                <div className="mt-1 flex flex-wrap gap-3 text-[10px] font-mono text-muted">
                  <span>bytes={formatBytes(totalLogBytes)}</span>
                  <span>lines={totalLogLines}</span>
                </div>
              </div>
              <div className="flex-1 space-y-2 overflow-y-auto p-3">
                {chunks.length === 0 ? (
                  <div className="rounded-lg border border-dashed border-edge bg-surface px-3 py-6 text-center text-sm text-subtle">
                    No log chunks yet.
                  </div>
                ) : (
                  chunks.map((chunk) => (
                    <button
                      key={chunk.seq}
                      onClick={() => { void handleSelectSeq(chunk.seq) }}
                      className={`w-full rounded-lg border px-3 py-2 text-left ${selectedSeq === chunk.seq ? 'border-accent bg-accent-dim/20' : 'border-edge bg-surface hover:border-accent/40'}`}
                      type="button"
                    >
                      <div className="flex items-center justify-between gap-2">
                        <div className="text-[11px] font-mono text-primary">chunk #{chunk.seq}</div>
                        {typeof chunk.lines === 'number' && <div className="text-[10px] font-mono text-muted">{chunk.lines} lines</div>}
                      </div>
                      <div className="mt-1 text-[10px] font-mono text-muted">{formatDateTime(chunk.created_at)}</div>
                    </button>
                  ))
                )}
              </div>
            </div>

            <div className="flex min-h-0 flex-col rounded-xl border border-edge bg-raised">
              <div className="flex items-center justify-between border-b border-edge px-4 py-3">
                <div className="text-heading text-sm font-semibold">Chunk Content</div>
                {selectedSeq !== null && <div className="text-[10px] font-mono text-muted">seq={selectedSeq}</div>}
              </div>
              <div className="min-h-0 flex-1 overflow-auto p-4">
                {logError ? (
                  <div className="rounded-lg border border-error/20 bg-error-dim px-3 py-2 text-error text-sm">{logError}</div>
                ) : loadingLogs ? (
                  <div className="text-sm text-muted">Loading...</div>
                ) : (
                  <pre className="whitespace-pre-wrap break-words text-[12px] leading-relaxed text-primary">{content || 'No log content.'}</pre>
                )}
              </div>
            </div>
          </div>
        ) : activeTab === 'checkout' ? (
          <div className="min-h-0 flex-1 overflow-y-auto rounded-xl border border-edge bg-raised p-4">
            {ledgerError ? (
              <div className="rounded-lg border border-error/20 bg-error-dim px-3 py-2 text-error text-sm">{ledgerError}</div>
            ) : loadingLedger ? (
              <div className="text-sm text-muted">Loading checkout ledger...</div>
            ) : checkoutLedger.length === 0 ? (
              <div className="rounded-lg border border-dashed border-edge bg-surface px-3 py-6 text-center text-sm text-subtle">
                No persisted checkout entries for this run yet.
              </div>
            ) : (
              <div className="space-y-3">
                {checkoutLedger.map((entry) => (
                  <div key={entry.entry_id} className="rounded-xl border border-edge bg-surface p-4">
                    <div className="flex items-start justify-between gap-3">
                      <div>
                        <div className="text-primary text-sm font-semibold">{entry.role.replaceAll('_', ' ')}</div>
                        <div className="mt-1 text-[10px] font-mono text-muted">
                          {entry.run_name ?? 'checkpoint'} · epoch={entry.epoch ?? 'n/a'} · score={typeof entry.score === 'number' ? entry.score.toFixed(3) : 'n/a'}
                        </div>
                      </div>
                      <div className="text-right text-[10px] font-mono">
                        <div className={entry.ok === true ? 'text-accent' : entry.ok === false ? 'text-error' : 'text-muted'}>
                          {entry.ok === true ? 'passed' : entry.ok === false ? 'failed' : 'n/a'}
                        </div>
                        <div className="text-muted">{formatDateTime(entry.created_at)}</div>
                      </div>
                    </div>
                    {entry.message && <div className="mt-3 text-[11px] leading-relaxed text-subtle">{entry.message}</div>}
                  </div>
                ))}
              </div>
            )}
          </div>
        ) : (
          <div className="grid min-h-0 flex-1 grid-cols-[320px_1fr] gap-4">
            <div className="flex min-h-0 flex-col rounded-xl border border-edge bg-raised">
              <div className="border-b border-edge px-4 py-3">
                <div className="text-heading text-sm font-semibold">Transcript Cache</div>
                <input
                  value={transcriptQuery}
                  onChange={(event) => setTranscriptQuery(event.target.value)}
                  className="mt-3 w-full rounded-lg border border-edge bg-surface px-3 py-2 text-sm text-primary outline-none focus:border-accent"
                  placeholder="Search transcripts or segment names"
                />
              </div>
              <div className="flex-1 space-y-2 overflow-y-auto p-3">
                {loadingPreprocess ? (
                  <div className="text-sm text-muted">Loading cached transcripts...</div>
                ) : preprocessError ? (
                  <div className="rounded-lg border border-error/20 bg-error-dim px-3 py-2 text-error text-sm">{preprocessError}</div>
                ) : !preprocess?.cache ? (
                  <div className="rounded-lg border border-dashed border-edge bg-surface px-3 py-6 text-center text-sm text-subtle">
                    This run has no reusable preprocess cache yet.
                  </div>
                ) : filteredEntries.length === 0 ? (
                  <div className="rounded-lg border border-dashed border-edge bg-surface px-3 py-6 text-center text-sm text-subtle">
                    No transcript entries match this filter.
                  </div>
                ) : (
                  filteredEntries.map((entry) => (
                    <button
                      key={entry.entry_id}
                      onClick={() => setSelectedEntryId(entry.entry_id)}
                      className={`w-full rounded-lg border px-3 py-2 text-left ${selectedEntry?.entry_id === entry.entry_id ? 'border-accent bg-accent-dim/20' : 'border-edge bg-surface hover:border-accent/40'}`}
                      type="button"
                    >
                      <div className="flex items-center justify-between gap-2">
                        <div className="text-[11px] font-mono text-primary">#{entry.seq}</div>
                        <div className={`text-[10px] font-mono ${entry.included ? 'text-accent' : 'text-muted'}`}>
                          {entry.included ? 'included' : 'excluded'}
                        </div>
                      </div>
                      <div className="mt-1 truncate text-[11px] text-primary">{getPathLabel(entry.audio_path)}</div>
                      <div className="mt-1 line-clamp-2 text-[11px] text-subtle">{entry.text}</div>
                    </button>
                  ))
                )}
              </div>
            </div>

            <div className="flex min-h-0 flex-col rounded-xl border border-edge bg-raised">
              <div className="border-b border-edge px-4 py-3">
                <div className="text-heading text-sm font-semibold">Transcript Editor</div>
                {saveMessage && (
                  <div className="mt-2 rounded-lg border border-accent/20 bg-accent-dim/20 px-3 py-2 text-[11px] text-primary">
                    {saveMessage}
                  </div>
                )}
              </div>
              <div className="min-h-0 flex-1 overflow-y-auto p-4">
                {!preprocess?.cache ? (
                  <div className="text-sm text-muted">Long-form runs populate a reusable transcript cache here after preprocessing.</div>
                ) : (
                  <div className="space-y-6">
                    <section className="rounded-xl border border-edge bg-surface p-4">
                      <div className="flex items-center justify-between gap-3">
                        <div>
                          <div className="text-primary text-sm font-semibold">Reference Text</div>
                          <div className="text-[11px] text-subtle">Used with cached reference audio for future retrains.</div>
                        </div>
                        <button
                          onClick={() => { void handleSaveReference() }}
                          disabled={saving !== ''}
                          className="rounded-lg border border-edge px-3 py-2 text-[11px] font-semibold text-primary hover:border-accent hover:text-accent disabled:opacity-50"
                          type="button"
                        >
                          {saving === 'reference' ? 'Saving...' : 'Save Reference'}
                        </button>
                      </div>
                      <textarea
                        value={referenceDraft}
                        onChange={(event) => setReferenceDraft(event.target.value)}
                        className="mt-3 min-h-24 w-full rounded-lg border border-edge bg-raised px-3 py-2 text-sm text-primary outline-none focus:border-accent"
                      />
                    </section>

                    <section className="rounded-xl border border-edge bg-surface p-4">
                      {!selectedEntry ? (
                        <div className="text-sm text-muted">Select a transcript entry to review and edit it.</div>
                      ) : (
                        <>
                          <div className="flex flex-wrap items-start justify-between gap-3">
                            <div>
                              <div className="text-primary text-sm font-semibold">Segment #{selectedEntry.seq}</div>
                              <div className="mt-1 text-[11px] font-mono text-muted">{selectedEntry.audio_path}</div>
                            </div>
                            <label className="inline-flex items-center gap-2 text-[11px] font-semibold text-primary">
                              <input
                                type="checkbox"
                                checked={entryIncluded}
                                onChange={(event) => setEntryIncluded(event.target.checked)}
                                className="h-4 w-4 rounded border-edge bg-surface"
                              />
                              Include In Future Retrains
                            </label>
                          </div>

                          <textarea
                            value={entryDraft}
                            onChange={(event) => setEntryDraft(event.target.value)}
                            className="mt-4 min-h-40 w-full rounded-lg border border-edge bg-raised px-3 py-2 text-sm text-primary outline-none focus:border-accent"
                          />

                          <div className="mt-4 flex items-center justify-between gap-3">
                            <div className="text-[11px] text-subtle">Editing here rewrites cached train_raw data used by next retrain.</div>
                            <button
                              onClick={() => { void handleSaveEntry() }}
                              disabled={saving !== ''}
                              className="rounded-lg bg-accent px-3 py-2 text-[11px] font-semibold text-void hover:bg-accent-light disabled:opacity-50"
                              type="button"
                            >
                              {saving === 'entry' ? 'Saving...' : 'Save Entry'}
                            </button>
                          </div>
                        </>
                      )}
                    </section>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
