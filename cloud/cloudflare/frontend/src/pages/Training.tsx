import { useState, useEffect, useRef } from 'react'
import { Link } from 'react-router'
import {
  fetchVoices,
  fetchTrainingJobs,
  startTraining,
  fetchTrainingJob,
  cancelTrainingJob,
  type Voice,
  type TrainingJob,
  type TrainingConfig,
  formatDate,
} from '../lib/api'

const ACTIVE_JOB_STATUSES = new Set([
  'pending',
  'running',
  'provisioning',
  'downloading',
  'preprocessing',
  'preparing',
  'training',
  'uploading',
])

function isJobActiveStatus(status: string): boolean {
  return ACTIVE_JOB_STATUSES.has(status)
}

export function Training() {
  const [voices, setVoices] = useState<Voice[]>([])
  const [loadingVoices, setLoadingVoices] = useState(true)

  // Form state
  const [selectedVoiceId, setSelectedVoiceId] = useState('')
  const [batchSize, setBatchSize] = useState(4)
  const [epochs, setEpochs] = useState(8)
  const [learningRate, setLearningRate] = useState(0.00001)
  const [starting, setStarting] = useState(false)
  const [formError, setFormError] = useState('')

  // Jobs
  const [jobs, setJobs] = useState<TrainingJob[]>([])
  const [loadingJobs, setLoadingJobs] = useState(false)
  const jobsRef = useRef<TrainingJob[]>([])

  // Keep ref in sync
  useEffect(() => {
    jobsRef.current = jobs
  }, [jobs])

  // Load voices
  useEffect(() => {
    let cancelled = false

    async function load() {
      try {
        const data = await fetchVoices()
        if (cancelled) return
        setVoices(data.voices)
        if (data.voices.length > 0 && !selectedVoiceId) {
          setSelectedVoiceId(data.voices[0].voice_id)
        }
      } catch {
        // silently fail
      } finally {
        if (!cancelled) setLoadingVoices(false)
      }
    }

    load()
    return () => { cancelled = true }
  }, [selectedVoiceId])

  useEffect(() => {
    let cancelled = false

    async function loadJobs() {
      setLoadingJobs(true)
      try {
        const data = await fetchTrainingJobs(undefined, 50)
        if (cancelled) return
        setJobs(data.jobs)
      } catch {
        // silently fail
      } finally {
        if (!cancelled) setLoadingJobs(false)
      }
    }

    loadJobs()
    return () => {
      cancelled = true
    }
  }, [])

  // Poll active jobs every 5 seconds
  const hasActiveJobs = jobs.some((j) => isJobActiveStatus(j.status))

  useEffect(() => {
    if (!hasActiveJobs) return

    const interval = setInterval(async () => {
      const currentJobs = jobsRef.current
      const activeJobIds = currentJobs
        .filter((j) => isJobActiveStatus(j.status))
        .map((j) => j.job_id)

      if (activeJobIds.length === 0) return

      try {
        const updates = await Promise.all(
          activeJobIds.map((id) => fetchTrainingJob(id)),
        )
        setJobs((prev) =>
          prev.map((j) => {
            const updated = updates.find((u) => u.job_id === j.job_id)
            return updated ?? j
          }),
        )
      } catch {
        // silently ignore polling errors
      }
    }, 5000)

    return () => clearInterval(interval)
  }, [hasActiveJobs])

  async function handleStartTraining(e: { preventDefault: () => void }) {
    e.preventDefault()
    if (!selectedVoiceId) return

    setStarting(true)
    setFormError('')

    const config: TrainingConfig = {
      batch_size: batchSize,
      num_epochs: epochs,
      learning_rate: learningRate,
    }

    try {
      const result = await startTraining(selectedVoiceId, config)

      // Add the new job to the list
      const newJob: TrainingJob = {
        job_id: result.job_id,
        status: result.status as TrainingJob['status'],
        voice_id: selectedVoiceId,
        created_at: Date.now(),
        last_heartbeat_at: null,
        summary: {},
        metrics: {},
        config,
        progress: {
          epoch: 0,
          total_epochs: epochs,
          loss: 0,
          step: 0,
          total_steps: 0,
        },
      }
      setJobs((prev) => [newJob, ...prev])
    } catch (err) {
      setFormError(err instanceof Error ? err.message : 'Failed to start training')
    } finally {
      setStarting(false)
    }
  }

  async function handleCancel(jobId: string) {
    try {
      await cancelTrainingJob(jobId)
      setJobs((prev) =>
        prev.map((j) =>
          j.job_id === jobId ? { ...j, status: 'cancelled' as const } : j,
        ),
      )
    } catch {
      // silently fail
    }
  }

  const activeJobs = jobs.filter((j) => isJobActiveStatus(j.status))
  const completedJobs = jobs.filter(
    (j) => j.status === 'completed' || j.status === 'failed',
  )

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-heading text-2xl font-bold">Training</h1>
        <p className="text-subtle text-sm mt-1">Fine-tune voice models</p>
      </div>

      <div className="grid lg:grid-cols-[380px_1fr] gap-6">
        {/* Start Training Form */}
        <div className="bg-raised border border-edge rounded-xl p-5">
          <h2 className="text-heading font-semibold text-sm mb-5">Start Training</h2>

          <div className="mb-4 rounded-lg border border-edge bg-surface px-3 py-2.5">
            <p className="text-subtle text-xs">
              Upload a real training set on the Voices page first. The defaults here are conservative and tuned for voice similarity over aggressive overfitting.
            </p>
            <Link
              to="/voices"
              className="inline-flex items-center gap-1 mt-2 text-accent text-xs font-medium hover:text-accent-light"
            >
              Go to Voices upload
            </Link>
          </div>

          {voices.length === 0 && !loadingVoices && (
            <div className="mb-4 rounded-lg border border-warning/20 bg-warning-dim px-3 py-2 text-warning text-xs">
              No voices found yet. Create a voice and upload audio first.
            </div>
          )}

          <form onSubmit={handleStartTraining} className="space-y-4">
            {/* Voice */}
            <div>
              <label htmlFor="training-voice" className="text-subtle text-xs font-medium mb-1.5 block">Voice</label>
              {loadingVoices ? (
                <div className="h-10 bg-surface rounded-lg animate-pulse" />
              ) : (
                <select
                  id="training-voice"
                  value={selectedVoiceId}
                  onChange={(e) => setSelectedVoiceId(e.target.value)}
                  className="w-full bg-surface border border-edge rounded-lg px-3 py-2.5 text-sm text-primary focus:border-accent transition-colors appearance-none"
                >
                  {voices.length === 0 && (
                    <option value="">No voices available</option>
                  )}
                  {voices.map((v) => (
                    <option key={v.voice_id} value={v.voice_id}>
                      {v.name}
                    </option>
                  ))}
                </select>
              )}
            </div>

            {/* Batch Size */}
            <div>
              <label htmlFor="training-batch-size" className="text-subtle text-xs font-medium mb-1.5 block">Batch Size</label>
              <input
                id="training-batch-size"
                type="number"
                value={batchSize}
                onChange={(e) => setBatchSize(parseInt(e.target.value) || 1)}
                min={1}
                max={32}
                className="w-full bg-surface border border-edge rounded-lg px-3 py-2.5 text-sm text-primary font-mono focus:border-accent transition-colors"
              />
            </div>

            {/* Epochs */}
            <div>
              <label htmlFor="training-epochs" className="text-subtle text-xs font-medium mb-1.5 block">Epochs</label>
              <input
                id="training-epochs"
                type="number"
                value={epochs}
                onChange={(e) => setEpochs(parseInt(e.target.value) || 1)}
                min={1}
                max={50}
                className="w-full bg-surface border border-edge rounded-lg px-3 py-2.5 text-sm text-primary font-mono focus:border-accent transition-colors"
              />
              <p className="text-muted text-[10px] font-mono mt-1">Recommended: 5–15 (more epochs = higher similarity, risk of overfitting)</p>
            </div>

            {/* Learning Rate */}
            <div>
              <label htmlFor="training-learning-rate" className="text-subtle text-xs font-medium mb-1.5 block">Learning Rate</label>
              <input
                id="training-learning-rate"
                type="text"
                value={learningRate}
                onChange={(e) => {
                  const val = parseFloat(e.target.value)
                  if (!isNaN(val)) setLearningRate(val)
                }}
                className="w-full bg-surface border border-edge rounded-lg px-3 py-2.5 text-sm text-primary font-mono focus:border-accent transition-colors"
              />
              <p className="text-muted text-[10px] font-mono mt-1">Recommended: 1e-5 to 5e-5</p>
            </div>

            {/* Error */}
            {formError && (
              <div className="bg-error-dim border border-error/20 rounded-lg px-3 py-2 text-error text-xs">
                {formError}
              </div>
            )}

            {/* Submit */}
            <button
              type="submit"
              disabled={!selectedVoiceId || starting || voices.length === 0}
              className="w-full bg-accent hover:bg-accent-light text-void font-semibold text-sm py-2.5 rounded-lg disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            >
              {starting ? 'Starting...' : 'Start Training'}
            </button>
          </form>
        </div>

        {/* Jobs Panel */}
        <div className="space-y-6">
          {/* Active Jobs */}
          <div className="bg-raised border border-edge rounded-xl p-5">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-heading font-semibold text-sm">
                Active Jobs
                {activeJobs.length > 0 && (
                  <span className="ml-2 text-accent font-mono text-xs">({activeJobs.length})</span>
                )}
              </h2>
              {hasActiveJobs && (
                <span className="inline-flex items-center gap-1.5 text-accent text-[10px] font-mono">
                  <span className="w-1.5 h-1.5 rounded-full bg-accent animate-pulse-glow" />
                  POLLING
                </span>
              )}
            </div>

            {activeJobs.length === 0 ? (
              <p className="text-muted text-sm text-center py-8">No active training jobs</p>
            ) : (
              <div className="space-y-3">
                {activeJobs.map((job) => (
                  <JobCard key={job.job_id} job={job} onCancel={handleCancel} />
                ))}
              </div>
            )}
          </div>

          {completedJobs.length > 0 && (
            <div className="bg-raised border border-edge rounded-xl p-5">
              <h2 className="text-heading font-semibold text-sm mb-4">
                History
                <span className="ml-2 text-muted font-mono text-xs">({completedJobs.length})</span>
              </h2>
              <div className="space-y-3">
                {completedJobs.map((job) => (
                  <JobCard key={job.job_id} job={job} onCancel={handleCancel} />
                ))}
              </div>
            </div>
          )}

          {/* Empty state */}
          {jobs.length === 0 && !loadingJobs && (
            <div className="text-center py-12">
              <div className="w-16 h-16 mx-auto mb-4 rounded-2xl bg-raised border border-edge flex items-center justify-center">
                <svg className="w-8 h-8 text-muted" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                  <title>Training jobs icon</title>
                  <polyline points="22,12 18,12 15,21 9,3 6,12 2,12" />
                </svg>
              </div>
              <h3 className="text-heading font-semibold mb-1">No training jobs</h3>
              <p className="text-subtle text-sm">Start a training job to fine-tune a voice model</p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

// ── Job Card ───────────────────────────────────────────────────────────────────

function JobCard({
  job,
  onCancel,
}: {
  job: TrainingJob
  onCancel: (jobId: string) => void
}) {
  const isActive = isJobActiveStatus(job.status)
  const epoch = typeof job.progress.epoch === 'number' ? job.progress.epoch : 0
  const totalEpochs = typeof job.progress.total_epochs === 'number' ? job.progress.total_epochs : 0
  const loss = typeof job.progress.loss === 'number' ? job.progress.loss : null
  const step = typeof job.progress.step === 'number' ? job.progress.step : 0
  const progressPct =
    totalEpochs > 0
      ? (epoch / totalEpochs) * 100
      : 0
  const durationMs = typeof job.summary?.duration_ms === 'number' ? job.summary.duration_ms : null
  const finalLoss = typeof job.summary?.final_loss === 'number' ? job.summary.final_loss : null
  const finalEpoch = typeof job.summary?.final_epoch === 'number' ? job.summary.final_epoch : null
  const summaryTotalEpochs = typeof job.summary?.total_epochs === 'number' ? job.summary.total_epochs : null

  const statusStyles: Record<string, { bg: string; text: string }> = {
    pending: { bg: 'bg-warning-dim', text: 'text-warning' },
    running: { bg: 'bg-accent-dim', text: 'text-accent' },
    provisioning: { bg: 'bg-warning-dim', text: 'text-warning' },
    downloading: { bg: 'bg-accent-dim', text: 'text-accent' },
    preprocessing: { bg: 'bg-accent-dim', text: 'text-accent' },
    preparing: { bg: 'bg-accent-dim', text: 'text-accent' },
    training: { bg: 'bg-accent-dim', text: 'text-accent' },
    uploading: { bg: 'bg-accent-dim', text: 'text-accent' },
    completed: { bg: 'bg-accent-dim', text: 'text-accent' },
    failed: { bg: 'bg-error-dim', text: 'text-error' },
    cancelled: { bg: 'bg-raised', text: 'text-muted' },
  }

  const style = statusStyles[job.status] ?? statusStyles.pending

  return (
    <div className="bg-surface rounded-lg p-4">
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <span className="text-primary text-xs font-medium font-mono">
            {job.job_id.slice(0, 12)}
          </span>
          <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-mono uppercase tracking-wider ${style.bg} ${style.text}`}>
            {job.status}
          </span>
        </div>
        {isActive && (
          <button
            onClick={() => onCancel(job.job_id)}
            className="text-muted hover:text-error text-xs font-medium transition-colors"
            type="button"
          >
            Cancel
          </button>
        )}
      </div>

      {/* Progress Bar */}
      {isActive && (
        <div className="mb-3">
          <div className="w-full h-2 bg-edge rounded-full overflow-hidden">
            <div
              className="h-full bg-accent rounded-full transition-[width] duration-500"
              style={{ width: `${progressPct}%` }}
            />
          </div>
        </div>
      )}

      {/* Metrics */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <Metric label="Epoch" value={`${epoch}/${totalEpochs || '—'}`} />
        <Metric label="Step" value={String(step)} />
        <Metric label="Loss" value={loss !== null && loss > 0 ? loss.toFixed(4) : '—'} />
        <Metric label="Created" value={formatDate(job.created_at)} />
      </div>

      {job.status === 'completed' && (durationMs !== null || finalLoss !== null || finalEpoch !== null) && (
        <div className="mt-3 grid grid-cols-3 gap-3">
          <Metric label="Duration" value={durationMs !== null ? `${Math.round(durationMs / 1000)}s` : '—'} />
          <Metric label="Final Loss" value={finalLoss !== null ? finalLoss.toFixed(4) : '—'} />
          <Metric label="Epochs" value={finalEpoch !== null ? `${finalEpoch}/${summaryTotalEpochs ?? '—'}` : '—'} />
        </div>
      )}

      {/* Config */}
      <div className="mt-3 pt-3 border-t border-edge flex gap-4 text-muted text-[10px] font-mono">
        <span>batch={job.config.batch_size}</span>
        <span>epochs={job.config.num_epochs}</span>
        <span>lr={job.config.learning_rate}</span>
      </div>
    </div>
  )
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-muted text-[10px] font-mono uppercase tracking-wider mb-0.5">
        {label}
      </div>
      <div className="text-primary text-sm font-mono tabular-nums">{value}</div>
    </div>
  )
}
