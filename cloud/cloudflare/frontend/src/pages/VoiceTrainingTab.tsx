import { type FormEvent, useEffect, useMemo, useRef, useState } from 'react'
import { Link, useParams, useSearchParams } from 'react-router'
import {
  cancelTrainingCampaign,
  cancelTrainingJob,
  createTrainingCampaign,
  fetchTrainingAdvice,
  fetchTrainingCampaign,
  fetchTrainingJob,
  fetchTrainingJobs,
  fetchVoice,
  fetchVoiceDatasets,
  reconcileTrainingJob,
  revalidateTrainingJob,
  startTraining,
  type CampaignDirection,
  type DatasetInfo,
  type TrainingAdvice,
  type TrainingCampaign,
  type TrainingConfig,
  type TrainingJob,
  type Voice,
} from '../lib/api'
import { TrainingAdviceCard } from '../components/TrainingAdviceCard'
import { AutopilotCard } from '../components/training/AutopilotCard'
import { TrainingHistoryList } from '../components/training/TrainingHistoryList'
import { TrainingJobRow } from '../components/training/TrainingJobRow'
import { buildTrainingAdvice } from '../lib/trainingAdvisor'
import { shouldWatchTrainingJob } from '../lib/trainingCheckout'

function getRecommendedTrainingPreset(modelSize: string) {
  if (modelSize.includes('0.6')) {
    return {
      batchSize: 2,
      epochs: 12,
      learningRate: 0.0000025,
      gradientAccumulationSteps: 4,
      subtalkerLossWeight: 0.3,
      saveEveryNEpochs: 1,
      seed: 303,
      gpuTypeId: 'NVIDIA L40S',
    }
  }

  return {
    batchSize: 2,
    epochs: 15,
    learningRate: 0.00002,
    gradientAccumulationSteps: 4,
    subtalkerLossWeight: 0.3,
    saveEveryNEpochs: 5,
    seed: 42,
    gpuTypeId: 'NVIDIA A100-SXM4-80GB',
  }
}

function inferDatasetNameFromRefAudioKey(refAudioKey: string | null | undefined): string | null {
  if (!refAudioKey || !/\/ref_audio\.[^/]+$/i.test(refAudioKey)) {
    return null
  }
  const parts = refAudioKey.split('/').filter(Boolean)
  if (parts.length < 4 || parts[0] !== 'datasets') {
    return null
  }
  return parts[2] || null
}

function getTrainingResetAt(voice: Voice | null | undefined): number | null {
  const raw = voice?.labels?.training_reset_at
  if (!raw) return null
  const parsed = Number(raw)
  return Number.isFinite(parsed) && parsed > 0 ? parsed : null
}

function getActiveCampaignId(jobs: TrainingJob[]): string | null {
  const sorted = [...jobs].sort((a, b) => (b.updated_at ?? b.created_at) - (a.updated_at ?? a.created_at))
  for (const job of sorted) {
    if (!job.campaign_id) continue
    if (shouldWatchTrainingJob(job)) {
      return job.campaign_id
    }
  }
  return null
}

export function VoiceTrainingTab() {
  const { voiceId = '' } = useParams()
  const [searchParams] = useSearchParams()

  const [voice, setVoice] = useState<Voice | null>(null)
  const [jobs, setJobs] = useState<TrainingJob[]>([])
  const [datasets, setDatasets] = useState<DatasetInfo[]>([])
  const [selectedDatasetName, setSelectedDatasetName] = useState('')
  const [loading, setLoading] = useState(true)

  const [serverTrainingAdvice, setServerTrainingAdvice] = useState<TrainingAdvice | null>(null)
  const [showManual, setShowManual] = useState(false)
  const [error, setError] = useState('')

  const [batchSize, setBatchSize] = useState(2)
  const [epochs, setEpochs] = useState(12)
  const [learningRate, setLearningRate] = useState(0.0000025)
  const [trainingLanguage, setTrainingLanguage] = useState('ko')
  const [trainingSeed, setTrainingSeed] = useState(303)
  const [gradientAccumulationSteps, setGradientAccumulationSteps] = useState(4)
  const [subtalkerLossWeight, setSubtalkerLossWeight] = useState(0.3)
  const [saveEveryNEpochs, setSaveEveryNEpochs] = useState(1)
  const [gpuTypeId, setGpuTypeId] = useState('NVIDIA L40S')
  const [startingManual, setStartingManual] = useState(false)

  const [campaignAttempts, setCampaignAttempts] = useState(6)
  const [campaignParallelism, setCampaignParallelism] = useState(3)
  const [campaignDirection, setCampaignDirection] = useState<CampaignDirection>('balanced')
  const [activeCampaignId, setActiveCampaignId] = useState<string | null>(null)
  const [activeCampaign, setActiveCampaign] = useState<TrainingCampaign | null>(null)
  const [startingCampaign, setStartingCampaign] = useState(false)

  const jobsRef = useRef<TrainingJob[]>([])

  useEffect(() => {
    jobsRef.current = jobs
  }, [jobs])

  const linkedDatasetName = inferDatasetNameFromRefAudioKey(voice?.ref_audio_r2_key)
  const effectiveDatasetName = selectedDatasetName || linkedDatasetName || ''
  const selectedVoiceResetAt = getTrainingResetAt(voice)
  const scopedJobs = useMemo(() => {
    return jobs.filter((job) => selectedVoiceResetAt === null || job.created_at >= selectedVoiceResetAt)
  }, [jobs, selectedVoiceResetAt])
  const activeJobs = useMemo(() => scopedJobs.filter((job) => shouldWatchTrainingJob(job)), [scopedJobs])
  const trainingAdvice = serverTrainingAdvice ?? buildTrainingAdvice(voice, scopedJobs)

  useEffect(() => {
    if (!voiceId) return
    let cancelled = false
    async function load() {
      setLoading(true)
      setError('')
      try {
        const [voiceData, datasetsData, jobsData] = await Promise.all([
          fetchVoice(voiceId),
          fetchVoiceDatasets(voiceId),
          fetchTrainingJobs(voiceId, 100),
        ])
        if (cancelled) return
        setVoice(voiceData)
        setDatasets(datasetsData.datasets)
        setJobs(jobsData.jobs)
        const linked = inferDatasetNameFromRefAudioKey(voiceData.ref_audio_r2_key)
        const paramDataset = searchParams.get('datasetName')
        setSelectedDatasetName(paramDataset ?? linked ?? datasetsData.datasets[0]?.name ?? '')
        setTrainingLanguage(voiceData.labels.language ?? 'ko')
        const preset = getRecommendedTrainingPreset(voiceData.model_size)
        setBatchSize(preset.batchSize)
        setEpochs(preset.epochs)
        setLearningRate(preset.learningRate)
        setTrainingSeed(preset.seed)
        setGradientAccumulationSteps(preset.gradientAccumulationSteps)
        setSubtalkerLossWeight(preset.subtalkerLossWeight)
        setSaveEveryNEpochs(preset.saveEveryNEpochs)
        setGpuTypeId(preset.gpuTypeId)
      } catch (loadError) {
        if (!cancelled) {
          setError(loadError instanceof Error ? loadError.message : 'Failed to load training data')
        }
      } finally {
        if (!cancelled) {
          setLoading(false)
        }
      }
    }
    void load()
    return () => {
      cancelled = true
    }
  }, [voiceId, searchParams])

  const adviceSignature = useMemo(() => (
    scopedJobs
      .map((job) => `${job.job_id}:${job.status}:${job.updated_at ?? job.created_at}`)
      .join('|')
  ), [scopedJobs])

  useEffect(() => {
    if (!voiceId) return
    const signatureAtRequest = adviceSignature
    let cancelled = false
    async function loadAdvice() {
      try {
        const response = await fetchTrainingAdvice(voiceId, 100)
        if (!cancelled && signatureAtRequest === adviceSignature) {
          setServerTrainingAdvice(response.advice)
        }
      } catch {
        if (!cancelled) {
          setServerTrainingAdvice(null)
        }
      }
    }
    void loadAdvice()
    return () => {
      cancelled = true
    }
  }, [voiceId, adviceSignature])

  useEffect(() => {
    const inferredId = getActiveCampaignId(activeJobs)
    setActiveCampaignId((current) => current ?? inferredId)
  }, [activeJobs])

  useEffect(() => {
    if (!activeCampaignId) return
    const campaignId = activeCampaignId
    let cancelled = false

    async function pollCampaign() {
      try {
        const response = await fetchTrainingCampaign(campaignId)
        if (cancelled) return
        setActiveCampaign(response.campaign)
        setJobs((previous) => {
          const byId = new Map(previous.map((job) => [job.job_id, job]))
          for (const job of response.attempts) {
            byId.set(job.job_id, job)
          }
          return [...byId.values()]
        })
        if (
          response.campaign.status === 'completed' ||
          response.campaign.status === 'failed' ||
          response.campaign.status === 'blocked_dataset' ||
          response.campaign.status === 'blocked_budget' ||
          response.campaign.status === 'cancelled'
        ) {
          setActiveCampaignId(null)
        }
      } catch {
        if (!cancelled) {
          setActiveCampaign(null)
        }
      }
    }

    void pollCampaign()
    const interval = setInterval(() => {
      void pollCampaign()
    }, 5000)

    return () => {
      cancelled = true
      clearInterval(interval)
    }
  }, [activeCampaignId])

  useEffect(() => {
    if (activeJobs.length === 0) return
    const interval = setInterval(async () => {
      const ids = jobsRef.current.filter((job) => shouldWatchTrainingJob(job)).map((job) => job.job_id)
      if (ids.length === 0) return
      try {
        const updates = await Promise.all(ids.map((jobId) => fetchTrainingJob(jobId)))
        setJobs((previous) => previous.map((job) => updates.find((candidate) => candidate.job_id === job.job_id) ?? job))
      } catch {
      }
    }, 5000)

    return () => clearInterval(interval)
  }, [activeJobs.length])

  async function handleStartCampaign() {
    if (!voiceId || !voice) return
    setStartingCampaign(true)
    setError('')

    const baseConfigOverrides: TrainingConfig = {
      batch_size: batchSize,
      num_epochs: epochs,
      learning_rate: learningRate,
      model_size: voice.model_size,
      gradient_accumulation_steps: gradientAccumulationSteps,
      subtalker_loss_weight: subtalkerLossWeight,
      save_every_n_epochs: saveEveryNEpochs,
      seed: trainingSeed,
      whisper_language: trainingLanguage,
      gpu_type_id: gpuTypeId,
    }

    try {
      const response = await createTrainingCampaign(voiceId, {
        datasetName: effectiveDatasetName || undefined,
        attemptCount: Math.max(1, Math.min(Math.trunc(campaignAttempts), 12)),
        parallelism: Math.max(1, Math.min(Math.trunc(campaignParallelism), 6)),
        direction: campaignDirection,
        baseConfigOverrides,
      })
      setActiveCampaign(response.campaign)
      setActiveCampaignId(response.campaign.campaign_id)
      setJobs((previous) => {
        const byId = new Map(previous.map((job) => [job.job_id, job]))
        for (const job of response.attempts) {
          byId.set(job.job_id, job)
        }
        return [...byId.values()]
      })
    } catch (startError) {
      setError(startError instanceof Error ? startError.message : 'Failed to start campaign')
    } finally {
      setStartingCampaign(false)
    }
  }

  async function handleStopCampaign() {
    if (!activeCampaignId) return
    try {
      const response = await cancelTrainingCampaign(activeCampaignId)
      setActiveCampaign(response.campaign)
      setJobs((previous) => {
        const byId = new Map(previous.map((job) => [job.job_id, job]))
        for (const job of response.attempts) {
          byId.set(job.job_id, job)
        }
        return [...byId.values()]
      })
      setActiveCampaignId(null)
    } catch (cancelError) {
      setError(cancelError instanceof Error ? cancelError.message : 'Failed to cancel campaign')
    }
  }

  async function handleStartManualTraining(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (!voiceId || !voice) return
    setStartingManual(true)
    setError('')

    const config: TrainingConfig = {
      batch_size: batchSize,
      num_epochs: epochs,
      learning_rate: learningRate,
      model_size: voice.model_size,
      gradient_accumulation_steps: gradientAccumulationSteps,
      subtalker_loss_weight: subtalkerLossWeight,
      save_every_n_epochs: saveEveryNEpochs,
      seed: trainingSeed,
      whisper_language: trainingLanguage,
      gpu_type_id: gpuTypeId,
    }

    try {
      const response = await startTraining(voiceId, config, {
        datasetName: effectiveDatasetName || undefined,
      })
      setJobs((previous) => [
        {
          job_id: response.job_id,
          status: response.status as TrainingJob['status'],
          voice_id: voiceId,
          round_id: response.round_id,
          dataset_snapshot_id: response.dataset_snapshot_id,
          created_at: Date.now(),
          config,
          progress: {},
          summary: {},
        },
        ...previous,
      ])
      setShowManual(false)
    } catch (startError) {
      setError(startError instanceof Error ? startError.message : 'Failed to start manual training')
    } finally {
      setStartingManual(false)
    }
  }

  async function handleCancel(jobId: string) {
    await cancelTrainingJob(jobId)
    setJobs((previous) => previous.map((job) => (job.job_id === jobId ? { ...job, status: 'cancelled' } : job)))
  }

  async function handleRefresh(jobId: string) {
    const updated = await fetchTrainingJob(jobId)
    setJobs((previous) => previous.map((job) => (job.job_id === jobId ? updated : job)))
  }

  async function handleReconcile(jobId: string) {
    await reconcileTrainingJob(jobId)
    await handleRefresh(jobId)
  }

  async function handleRevalidate(jobId: string) {
    const response = await revalidateTrainingJob(jobId)
    setJobs((previous) => previous.map((job) => (job.job_id === jobId ? response.job : job)))
  }

  if (loading) {
    return <div className="rounded-xl border border-edge bg-raised p-6 text-sm text-muted">Loading training workspace...</div>
  }

  const datasetReady = datasets.length > 0 || Boolean(linkedDatasetName)

  return (
    <div className="space-y-6">
      <AutopilotCard
        datasetReady={datasetReady}
        datasetName={effectiveDatasetName || null}
        direction={campaignDirection}
        attempts={campaignAttempts}
        parallelism={campaignParallelism}
        starting={startingCampaign}
        campaign={activeCampaign}
        onDirectionChange={setCampaignDirection}
        onAttemptsChange={setCampaignAttempts}
        onParallelismChange={setCampaignParallelism}
        onStart={() => { void handleStartCampaign() }}
        onStop={() => { void handleStopCampaign() }}
      />

      {activeCampaign && (
        <section className="rounded-xl border border-edge bg-raised p-4">
          <h3 className="text-heading text-sm font-semibold">Current Campaign Status</h3>
          <div className="mt-3 grid gap-3 sm:grid-cols-3">
            <div className="rounded-lg border border-edge bg-surface px-3 py-2">
              <p className="text-[10px] font-mono text-muted">STATUS</p>
              <p className="text-sm font-semibold text-primary">{activeCampaign.status}</p>
            </div>
            <div className="rounded-lg border border-edge bg-surface px-3 py-2">
              <p className="text-[10px] font-mono text-muted">BEST SCORE</p>
              <p className="text-sm font-semibold text-accent">
                {typeof activeCampaign.planner_state.best_score === 'number'
                  ? activeCampaign.planner_state.best_score.toFixed(3)
                  : 'n/a'}
              </p>
            </div>
            <div className="rounded-lg border border-edge bg-surface px-3 py-2">
              <p className="text-[10px] font-mono text-muted">ATTEMPTS</p>
              <p className="text-sm font-semibold text-primary">
                {Number(activeCampaign.summary.attempts_created ?? 0)}/{activeCampaign.attempt_count}
              </p>
            </div>
          </div>
        </section>
      )}

      {voice && trainingAdvice && (
        <TrainingAdviceCard
          voiceId={voice.voice_id}
          advice={trainingAdvice}
          onApplyConfig={(config) => {
            const fallback = getRecommendedTrainingPreset(config.model_size ?? voice.model_size)
            setBatchSize(config.batch_size ?? fallback.batchSize)
            setEpochs(config.num_epochs ?? fallback.epochs)
            setLearningRate(config.learning_rate ?? fallback.learningRate)
            setTrainingSeed(config.seed ?? fallback.seed)
            setGradientAccumulationSteps(config.gradient_accumulation_steps ?? fallback.gradientAccumulationSteps)
            setSubtalkerLossWeight(config.subtalker_loss_weight ?? fallback.subtalkerLossWeight)
            setSaveEveryNEpochs(config.save_every_n_epochs ?? fallback.saveEveryNEpochs)
            setTrainingLanguage(config.whisper_language ?? trainingLanguage)
            setGpuTypeId(config.gpu_type_id ?? fallback.gpuTypeId)
          }}
        />
      )}

      <section className="rounded-xl border border-edge bg-raised p-4">
        <div className="mb-3 flex items-center justify-between">
          <h3 className="text-heading text-sm font-semibold">Active Jobs</h3>
          <span className="text-[10px] font-mono text-muted">{activeJobs.length}</span>
        </div>
        {activeJobs.length === 0 ? (
          <p className="rounded-lg border border-dashed border-edge bg-surface px-3 py-6 text-center text-sm text-subtle">
            No active jobs for this voice.
          </p>
        ) : (
          <div className="space-y-3">
            {activeJobs.map((job) => (
              <TrainingJobRow
                key={job.job_id}
                job={job}
                voiceName={voice?.name}
                onCancel={handleCancel}
                onRefresh={handleRefresh}
                onReconcile={handleReconcile}
                onRevalidate={handleRevalidate}
              />
            ))}
          </div>
        )}
      </section>

      <TrainingHistoryList
        jobs={scopedJobs}
        voiceName={voice?.name}
        onCancel={handleCancel}
        onRefresh={handleRefresh}
        onReconcile={handleReconcile}
        onRevalidate={handleRevalidate}
        defaultCollapsed
      />

      <section className="rounded-xl border border-edge bg-raised p-4">
        <button
          type="button"
          onClick={() => setShowManual((value) => !value)}
          className="flex w-full items-center justify-between text-left"
        >
          <h3 className="text-heading text-sm font-semibold">Advanced / Manual Run</h3>
          <span className="text-muted text-xs font-mono">{showManual ? 'hide' : 'show'}</span>
        </button>

        {showManual && (
          <form className="mt-4 space-y-3 border-t border-edge pt-4" onSubmit={handleStartManualTraining}>
            <div className="grid gap-3 sm:grid-cols-2">
              <label className="text-xs text-subtle">Dataset
                <select
                  value={effectiveDatasetName}
                  onChange={(event) => setSelectedDatasetName(event.target.value)}
                  className="mt-1 w-full rounded-lg border border-edge bg-surface px-3 py-2 text-sm text-primary"
                >
                  {datasets.length === 0 && <option value="">No finalized dataset</option>}
                  {datasets.map((dataset) => (
                    <option key={dataset.name} value={dataset.name}>{dataset.name}</option>
                  ))}
                </select>
              </label>
              <label className="text-xs text-subtle">Language
                <select
                  value={trainingLanguage}
                  onChange={(event) => setTrainingLanguage(event.target.value)}
                  className="mt-1 w-full rounded-lg border border-edge bg-surface px-3 py-2 text-sm text-primary"
                >
                  <option value="ko">Korean</option>
                  <option value="en">English</option>
                  <option value="ja">Japanese</option>
                  <option value="zh">Chinese</option>
                </select>
              </label>
            </div>

            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
              <label className="text-xs text-subtle">Batch Size
                <input type="number" min={1} max={32} value={batchSize} onChange={(e) => setBatchSize(parseInt(e.target.value, 10) || 1)} className="mt-1 w-full rounded-lg border border-edge bg-surface px-3 py-2 text-sm text-primary" />
              </label>
              <label className="text-xs text-subtle">Epochs
                <input type="number" min={1} max={50} value={epochs} onChange={(e) => setEpochs(parseInt(e.target.value, 10) || 1)} className="mt-1 w-full rounded-lg border border-edge bg-surface px-3 py-2 text-sm text-primary" />
              </label>
              <label className="text-xs text-subtle">Learning Rate
                <input type="text" value={learningRate} onChange={(e) => setLearningRate(parseFloat(e.target.value) || 0)} className="mt-1 w-full rounded-lg border border-edge bg-surface px-3 py-2 text-sm text-primary font-mono" />
              </label>
              <label className="text-xs text-subtle">Seed
                <input type="number" value={trainingSeed} onChange={(e) => setTrainingSeed(parseInt(e.target.value, 10) || 1)} className="mt-1 w-full rounded-lg border border-edge bg-surface px-3 py-2 text-sm text-primary font-mono" />
              </label>
            </div>

            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
              <label className="text-xs text-subtle">Grad Accum
                <input type="number" min={1} value={gradientAccumulationSteps} onChange={(e) => setGradientAccumulationSteps(parseInt(e.target.value, 10) || 1)} className="mt-1 w-full rounded-lg border border-edge bg-surface px-3 py-2 text-sm text-primary" />
              </label>
              <label className="text-xs text-subtle">Subtalker Weight
                <input type="number" step="0.05" value={subtalkerLossWeight} onChange={(e) => setSubtalkerLossWeight(parseFloat(e.target.value) || 0)} className="mt-1 w-full rounded-lg border border-edge bg-surface px-3 py-2 text-sm text-primary" />
              </label>
              <label className="text-xs text-subtle">Save Every N Epochs
                <input type="number" min={1} value={saveEveryNEpochs} onChange={(e) => setSaveEveryNEpochs(parseInt(e.target.value, 10) || 1)} className="mt-1 w-full rounded-lg border border-edge bg-surface px-3 py-2 text-sm text-primary" />
              </label>
              <label className="text-xs text-subtle">GPU Type
                <input type="text" value={gpuTypeId} onChange={(e) => setGpuTypeId(e.target.value)} className="mt-1 w-full rounded-lg border border-edge bg-surface px-3 py-2 text-sm text-primary" />
              </label>
            </div>

            <div className="flex flex-wrap items-center justify-between gap-2">
              <Link to={`/voices/${voiceId}/dataset`} className="text-xs text-accent hover:text-accent-light">
                Need dataset changes? Open Dataset tab
              </Link>
              <button
                type="submit"
                disabled={startingManual || !datasetReady}
                className="rounded-lg bg-accent px-4 py-2 text-xs font-semibold text-void disabled:opacity-50"
              >
                {startingManual ? 'Starting...' : 'Start Single Training'}
              </button>
            </div>
          </form>
        )}
      </section>

      {error && (
        <div className="rounded-lg border border-error/20 bg-error-dim px-3 py-2 text-sm text-error">
          {error}
        </div>
      )}
    </div>
  )
}
