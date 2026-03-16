import { useEffect, useMemo, useState } from 'react'
import {
  fetchAllTrainingJobs,
  fetchVoices,
  type TrainingJob,
  type Voice,
} from '../lib/api'
import { shouldWatchTrainingJob } from '../lib/trainingCheckout'
import { TrainingJobRow } from '../components/training/TrainingJobRow'

type QueueStats = {
  active_workers: number
  max_workers: number
  active_jobs: number
  queued_jobs: number
  running_jobs: number
}

export function QueuePage() {
  const [jobs, setJobs] = useState<TrainingJob[]>([])
  const [voices, setVoices] = useState<Voice[]>([])
  const [stats, setStats] = useState<QueueStats | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    let cancelled = false

    async function load() {
      setLoading(true)
      setError('')
      try {
        const [jobsResponse, voicesResponse] = await Promise.all([
          fetchAllTrainingJobs(200),
          fetchVoices(),
        ])
        if (cancelled) return
        setJobs(jobsResponse.jobs)
        setVoices(voicesResponse.voices)
        const activeStatuses = new Set(['running', 'provisioning', 'downloading', 'preprocessing', 'preparing', 'training', 'uploading'])
        const running = jobsResponse.jobs.filter((j) => activeStatuses.has(j.status)).length
        const queued = jobsResponse.jobs.filter((j) => j.status === 'queued' || j.status === 'pending').length
        setStats({ active_workers: running, max_workers: Math.max(running, 3), active_jobs: running + queued, queued_jobs: queued, running_jobs: running })
      } catch (loadError) {
        if (!cancelled) {
          setError(loadError instanceof Error ? loadError.message : 'Failed to load queue')
        }
      } finally {
        if (!cancelled) {
          setLoading(false)
        }
      }
    }

    void load()
    const interval = setInterval(() => {
      void load()
    }, 5000)

    return () => {
      cancelled = true
      clearInterval(interval)
    }
  }, [])

  const voiceNames = useMemo(() => new Map(voices.map((voice) => [voice.voice_id, voice.name])), [voices])
  const activeJobs = useMemo(() => jobs.filter((job) => shouldWatchTrainingJob(job)), [jobs])
  const queuedJobs = useMemo(() => jobs.filter((job) => job.status === 'queued' || job.status === 'pending'), [jobs])
  const recentCompletedJobs = useMemo(() => (
    [...jobs]
      .filter((job) => job.status === 'completed' || job.status === 'failed' || job.status === 'cancelled')
      .sort((a, b) => (b.updated_at ?? b.created_at) - (a.updated_at ?? a.created_at))
      .slice(0, 12)
  ), [jobs])

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-heading text-2xl font-bold">Queue Monitor</h1>
          <p className="text-subtle text-sm mt-1">Global training throughput across all voices.</p>
        </div>
        <div className="rounded-lg border border-edge bg-raised px-3 py-2 text-sm font-mono text-primary">
          {stats ? `${stats.active_workers}/${stats.max_workers} workers active` : 'Loading capacity...'}
        </div>
      </div>

      {error && (
        <div className="rounded-lg border border-error/20 bg-error-dim px-3 py-2 text-sm text-error">{error}</div>
      )}

      <section className="rounded-xl border border-edge bg-raised p-4">
        <h2 className="text-heading text-sm font-semibold">Active Training Jobs ({activeJobs.length})</h2>
        {loading ? (
          <p className="mt-4 text-sm text-muted">Loading queue...</p>
        ) : activeJobs.length === 0 ? (
          <p className="mt-4 rounded-lg border border-dashed border-edge bg-surface px-3 py-6 text-center text-sm text-subtle">
            No active training jobs.
          </p>
        ) : (
          <div className="mt-4 space-y-3">
            {activeJobs.map((job) => (
              <TrainingJobRow key={job.job_id} job={job} voiceName={voiceNames.get(job.voice_id)} compact />
            ))}
          </div>
        )}
      </section>

      <section className="rounded-xl border border-edge bg-raised p-4">
        <h2 className="text-heading text-sm font-semibold">Queued Jobs ({queuedJobs.length})</h2>
        {queuedJobs.length === 0 ? (
          <p className="mt-4 text-sm text-muted">No jobs waiting for slots.</p>
        ) : (
          <div className="mt-4 space-y-3">
            {queuedJobs.map((job) => (
              <TrainingJobRow key={job.job_id} job={job} voiceName={voiceNames.get(job.voice_id)} compact />
            ))}
          </div>
        )}
      </section>

      <section className="rounded-xl border border-edge bg-raised p-4">
        <h2 className="text-heading text-sm font-semibold">Recent Completed Jobs</h2>
        {recentCompletedJobs.length === 0 ? (
          <p className="mt-4 text-sm text-muted">No recent completed jobs.</p>
        ) : (
          <div className="mt-4 space-y-3">
            {recentCompletedJobs.map((job) => (
              <TrainingJobRow key={job.job_id} job={job} voiceName={voiceNames.get(job.voice_id)} compact />
            ))}
          </div>
        )}
      </section>
    </div>
  )
}
