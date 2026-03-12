// ── Types ──────────────────────────────────────────────────────────────────────

export interface VoiceSettings {
  stability: number
  similarity_boost: number
  style: number
  speed: number
}

export interface Voice {
  voice_id: string
  name: string
  description: string
  status: 'ready' | 'training' | 'created'
  model_size: string
  run_name?: string | null
  epoch?: number | null
  checkpoint_preset?: string | null
  checkpoint_score?: number | null
  checkpoint_job_id?: string | null
  checkpoint_r2_prefix?: string | null
  candidate_checkpoint_r2_prefix?: string | null
  candidate_run_name?: string | null
  candidate_epoch?: number | null
  candidate_preset?: string | null
  candidate_score?: number | null
  candidate_job_id?: string | null
  active_round_id?: string | null
  ref_audio_r2_key?: string | null
  settings: VoiceSettings
  labels: Record<string, string>
  created_at: string | number
  updated_at?: string | number
}

export type VoiceModelSize = '1.7B' | '0.6B'

export interface VoicesResponse {
  voices: Voice[]
  has_more: boolean
  total_count: number
}

export interface Model {
  model_id: string
  name: string
  description: string
}

export interface AsyncSpeechJob {
  job_id: string
  status: string
}

export interface AsyncSpeechStatus {
  status: string
  audio?: string
  sample_rate?: number
  duration_ms?: number | null
  error?: string
}

export interface SpeechGenerationOptions {
  stylePrompt?: string
  instruct?: string
  checkpointPrefix?: string
  checkpointEpoch?: number
  seed?: number
}

export interface TrainingStartOptions {
  datasetName?: string
}

export type TrainingCampaignStatus =
  | 'planning'
  | 'running'
  | 'completed'
  | 'failed'
  | 'blocked_dataset'
  | 'blocked_budget'
  | 'cancelled'

export interface TrainingCampaignStopRules {
  max_infra_failures?: number
  max_asr_failures?: number
  min_score_improvement?: number
  stagnation_window?: number
}

export interface TrainingCampaign {
  campaign_id: string
  voice_id: string
  dataset_name: string | null
  dataset_r2_prefix: string | null
  dataset_snapshot_id: string | null
  attempt_count: number
  parallelism: number
  status: TrainingCampaignStatus
  base_config: TrainingConfig
  stop_rules: TrainingCampaignStopRules
  planner_state: Record<string, unknown>
  summary: Record<string, unknown>
  created_at: number
  updated_at: number
  completed_at: number | null
}

export interface CreateTrainingCampaignOptions {
  datasetName?: string
  attemptCount: number
  parallelism?: number
  baseConfigOverrides?: TrainingConfig
  stopRules?: TrainingCampaignStopRules
}

export interface TrainingProgress {
  epoch?: number
  total_epochs?: number
  loss?: number
  step?: number
  total_steps?: number
  [key: string]: unknown
}

export type TrainingCheckoutAdoptionMode = 'promote' | 'candidate' | 'keep_current'

export type TrainingCheckoutSearchStatus =
  | 'pending'
  | 'validating'
  | 'promoted'
  | 'candidate_ready'
  | 'kept_current'
  | 'manual_promoted'
  | 'rejected'
  | 'failed'

export interface TrainingCheckoutTarget {
  prefix: string
  epoch: number | null
  preset: string | null
  score: number | null
  run_name: string | null
}

export interface TrainingCheckoutEvaluation {
  epoch: number
  prefix: string
  ok: boolean
  score: number
  message: string
  preset: string
  passed_samples: number
  total_samples: number
  run_name: string | null
  is_champion: boolean
  is_selected: boolean
}

export interface TrainingCheckoutLedgerEntry {
  entry_id: string
  round_id: string | null
  job_id: string
  voice_id: string
  checkpoint_r2_prefix: string
  run_name: string | null
  epoch: number | null
  preset: string | null
  score: number | null
  ok: boolean | null
  passed_samples: number | null
  total_samples: number | null
  message: string | null
  role: string
  source: string
  adoption_mode: string | null
  created_at: number
  updated_at: number
}

export interface TrainingCheckoutSearch {
  status: TrainingCheckoutSearchStatus
  validation_checked: boolean
  validation_passed: boolean
  validation_in_progress: boolean
  has_candidates: boolean
  compare_ready: boolean
  adoption_mode: TrainingCheckoutAdoptionMode | null
  message: string | null
  last_message: string | null
  champion: TrainingCheckoutTarget | null
  selected: TrainingCheckoutTarget | null
  manual_promoted: TrainingCheckoutTarget | null
  evaluated: TrainingCheckoutEvaluation[]
}

export interface TrainingConfig {
  batch_size?: number
  num_epochs?: number
  learning_rate?: number
  model_size?: string
  gradient_accumulation_steps?: number
  subtalker_loss_weight?: number
  save_every_n_epochs?: number
  seed?: number
  whisper_language?: string
  gpu_type_id?: string
}

export type TrainingAdviceMode =
  | 'compare-first'
  | 'dataset-first'
  | 'tone-explore'
  | 'stability-reset'
  | 'checkpoint-window'
  | 'hold-current'

export interface TrainingAdvice {
  mode: TrainingAdviceMode
  title: string
  summary: string
  confidence: 'high' | 'medium'
  reasons: string[]
  suggestedConfig: TrainingConfig | null
  compareFirst: boolean
  reviewDatasetFirst: boolean
  primaryActionLabel?: string
  analysisProvider?: 'heuristic' | 'llm'
}

export interface TrainingJob {
  job_id: string
  campaign_id?: string | null
  attempt_index?: number | null
  round_id?: string | null
  dataset_snapshot_id?: string | null
  status:
    | 'queued'
    | 'pending'
    | 'running'
    | 'provisioning'
    | 'downloading'
    | 'preprocessing'
    | 'preparing'
    | 'training'
    | 'uploading'
    | 'completed'
    | 'failed'
    | 'cancelled'
  progress: TrainingProgress
  voice_id: string
  created_at: number
  updated_at?: number
  completed_at?: number | null
  started_at?: number | null
  last_heartbeat_at?: number | null
  error_message?: string | null
  config: TrainingConfig
  summary?: Record<string, unknown>
  checkout_search?: TrainingCheckoutSearch
  metrics?: Record<string, unknown>
  supervisor?: Record<string, unknown>
}

export interface DatasetSnapshot {
  snapshot_id: string
  voice_id: string
  dataset_name?: string | null
  dataset_r2_prefix: string
  dataset_signature: string
  status: string
  source_cache_id?: string | null
  cache_r2_prefix?: string | null
  train_raw_r2_key?: string | null
  ref_audio_r2_key?: string | null
  reference_profile_r2_key?: string | null
  reference_text?: string | null
  source_file_count?: number | null
  segments_created?: number | null
  segments_accepted?: number | null
  accepted_duration_min?: number | null
  created_from_job_id?: string | null
  created_at: number
  updated_at: number
}

export interface TrainingRound {
  round_id: string
  voice_id: string
  dataset_snapshot_id?: string | null
  round_index: number
  status: string
  production_checkpoint_r2_prefix?: string | null
  production_run_name?: string | null
  production_epoch?: number | null
  production_preset?: string | null
  production_score?: number | null
  production_job_id?: string | null
  champion_checkpoint_r2_prefix?: string | null
  champion_run_name?: string | null
  champion_epoch?: number | null
  champion_preset?: string | null
  champion_score?: number | null
  champion_job_id?: string | null
  selected_checkpoint_r2_prefix?: string | null
  selected_run_name?: string | null
  selected_epoch?: number | null
  selected_preset?: string | null
  selected_score?: number | null
  selected_job_id?: string | null
  adoption_mode?: string | null
  candidate_checkpoint_r2_prefix?: string | null
  candidate_run_name?: string | null
  candidate_epoch?: number | null
  candidate_score?: number | null
  candidate_job_id?: string | null
  summary?: Record<string, unknown>
  created_at: number
  updated_at: number
  started_at?: number | null
  completed_at?: number | null
}

export interface TrainingLogChunk {
  job_id: string
  seq: number
  r2_key: string
  created_at: number
  bytes?: number | null
  lines?: number | null
}

export interface DatasetPreprocessCache {
  cache_id: string
  voice_id: string
  dataset_r2_prefix: string
  dataset_signature: string
  cache_r2_prefix: string
  train_raw_r2_key: string
  ref_audio_r2_key?: string | null
  reference_profile_r2_key?: string | null
  source_file_count?: number | null
  segments_created?: number | null
  segments_accepted?: number | null
  accepted_duration_min?: number | null
  created_at: number
  updated_at: number
}

export interface DatasetPreprocessCacheEntry {
  entry_id: string
  cache_id: string
  seq: number
  audio_path: string
  audio_r2_key: string
  text: string
  included: boolean
  created_at: number
  updated_at: number
}

export interface TrainingPreprocessCacheResponse {
  job_id: string
  cache: DatasetPreprocessCache | null
  entries: DatasetPreprocessCacheEntry[]
  reference_text: string | null
  hydrated_from_r2: boolean
}

export interface DatasetInfo {
  name: string
  r2_prefix: string
  file_count: number
}

export interface RawDatasetFile {
  key: string
  filename: string
  size: number
  uploaded: string
  content_type?: string | null
}

export interface DatasetDraftItem {
  audio_r2_key: string
  text: string
}

export interface DatasetRetranscribeResult {
  audio_r2_key: string
  provider?: string
  asr_text?: string
  source_text?: string
  asr_score?: number | null
  error?: string
}

export interface DatasetReviewResult {
  segment?: string
  original_text: string
  corrected: string
  score: number
  issues: string[]
}

export interface PresignedUpload {
  upload_url: string
  r2_key: string
}

export interface UploadProgress {
  loadedBytes: number
  totalBytes: number
}

interface MultipartUploadSession {
  upload_id: string
  r2_key: string
  chunk_size_bytes?: number
}

interface MultipartUploadPart {
  partNumber: number
  etag: string
}

// ── Error ──────────────────────────────────────────────────────────────────────

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
    public detail?: string,
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

// ── Helpers ────────────────────────────────────────────────────────────────────

const API_URL = import.meta.env.VITE_API_URL ?? ''
const MULTIPART_UPLOAD_THRESHOLD_BYTES = 32 * 1024 * 1024
const DEFAULT_MULTIPART_CHUNK_SIZE_BYTES = 8 * 1024 * 1024

function getApiKey(): string {
  return (localStorage.getItem('xi-api-key') ?? '').trim()
}

function authHeaders(): Record<string, string> {
  const key = getApiKey()
  if (!key) return {}
  return { 'xi-api-key': key }
}

async function request<T>(
  path: string,
  options: RequestInit = {},
): Promise<T> {
  const url = `${API_URL}${path}`
  const response = await fetch(url, {
    ...options,
    headers: {
      ...authHeaders(),
      ...options.headers,
    },
  })

  if (!response.ok) {
    let detail = `Request failed: ${response.status}`
    try {
      const body = await response.json() as Record<string, unknown>
      if (typeof body.detail === 'string') {
        detail = body.detail
      } else if (
        typeof body.detail === 'object' &&
        body.detail !== null &&
        typeof (body.detail as { message?: unknown }).message === 'string'
      ) {
        detail = (body.detail as { message: string }).message
      }
    } catch {
      // ignore parse errors
    }

    if (response.status === 401) {
      detail = 'Access denied by the API deployment.'
    }

    throw new ApiError(detail, response.status, detail)
  }

  return response.json() as Promise<T>
}

async function requestText(
  path: string,
  options: RequestInit = {},
): Promise<string> {
  const url = `${API_URL}${path}`
  const response = await fetch(url, {
    ...options,
    headers: {
      ...authHeaders(),
      ...options.headers,
    },
  })

  if (!response.ok) {
    let detail = `Request failed: ${response.status}`
    try {
      const body = await response.json() as Record<string, unknown>
      if (typeof body.detail === 'string') {
        detail = body.detail
      } else if (
        typeof body.detail === 'object' &&
        body.detail !== null &&
        typeof (body.detail as { message?: unknown }).message === 'string'
      ) {
        detail = (body.detail as { message: string }).message
      }
    } catch {
      // ignore parse errors
    }
    throw new ApiError(detail, response.status, detail)
  }

  return response.text()
}

// ── Voices ─────────────────────────────────────────────────────────────────────

export async function fetchVoices(): Promise<VoicesResponse> {
  const response = await request<VoicesResponse>('/v1/voices')
  return {
    ...response,
    voices: [...response.voices].sort((a, b) => {
      const aTime = new Date(a.updated_at ?? a.created_at).getTime()
      const bTime = new Date(b.updated_at ?? b.created_at).getTime()
      return bTime - aTime
    }),
  }
}

export async function fetchVoice(voiceId: string): Promise<Voice> {
  return request<Voice>(`/v1/voices/${voiceId}`)
}

export async function createVoice(
  name: string,
  description: string,
  audioFiles: File[],
  modelSize: VoiceModelSize = '0.6B',
): Promise<{ voice_id: string }> {
  if (audioFiles.length === 0) {
    throw new ApiError('At least one training audio file is required', 400)
  }

  const formData = new FormData()
  formData.append('name', name)
  formData.append('description', description)
  formData.append('model_size', modelSize)
  for (const audioFile of audioFiles) {
    formData.append('files', audioFile)
  }

  const url = `${API_URL}/v1/voices/add`
  const response = await fetch(url, {
    method: 'POST',
    headers: authHeaders(),
    body: formData,
  })

  if (!response.ok) {
    let detail = 'Failed to create voice'
    try {
      const body = await response.json() as Record<string, unknown>
      if (typeof body.detail === 'string') detail = body.detail
    } catch {
      // ignore
    }
    throw new ApiError(detail, response.status, detail)
  }

  return response.json() as Promise<{ voice_id: string }>
}

export async function createVoiceDraft(
  name: string,
  description: string,
  modelSize: VoiceModelSize = '0.6B',
): Promise<{ voice_id: string }> {
  const formData = new FormData()
  formData.append('name', name)
  formData.append('description', description)
  formData.append('model_size', modelSize)

  const url = `${API_URL}/v1/voices/add`
  const response = await fetch(url, {
    method: 'POST',
    headers: authHeaders(),
    body: formData,
  })

  if (!response.ok) {
    let detail = 'Failed to create voice'
    try {
      const body = await response.json() as Record<string, unknown>
      if (typeof body.detail === 'string') detail = body.detail
    } catch {
      // ignore
    }
    throw new ApiError(detail, response.status, detail)
  }

  return response.json() as Promise<{ voice_id: string }>
}

export async function deleteVoice(
  voiceId: string,
): Promise<{ status: string }> {
  return request<{ status: string }>(`/v1/voices/${voiceId}`, {
    method: 'DELETE',
  })
}

// ── Text-to-Speech ─────────────────────────────────────────────────────────────

export async function generateSpeech(
  voiceId: string,
  text: string,
  voiceSettings?: Partial<VoiceSettings>,
  options?: SpeechGenerationOptions,
): Promise<Blob> {
  const url = `${API_URL}/v1/text-to-speech/${voiceId}`
  const response = await fetch(url, {
    method: 'POST',
    headers: {
      ...authHeaders(),
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      text,
      voice_settings: voiceSettings,
      style_prompt: options?.stylePrompt,
      instruct: options?.instruct,
      checkpoint_prefix: options?.checkpointPrefix,
      checkpoint_epoch: options?.checkpointEpoch,
      seed: options?.seed,
    }),
  })

  if (!response.ok) {
    let detail = 'Speech generation failed'
    try {
      const body = await response.json() as Record<string, unknown>
      if (typeof body.detail === 'string') detail = body.detail
    } catch {
      // binary response, can't parse
    }
    throw new ApiError(detail, response.status, detail)
  }

  return response.blob()
}

export async function startSpeechGenerationAsync(
  voiceId: string,
  text: string,
  voiceSettings?: Partial<VoiceSettings>,
  options?: SpeechGenerationOptions,
): Promise<AsyncSpeechJob> {
  return request<AsyncSpeechJob>(`/v1/text-to-speech/${voiceId}/async`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      text,
      voice_settings: voiceSettings,
      style_prompt: options?.stylePrompt,
      instruct: options?.instruct,
      checkpoint_prefix: options?.checkpointPrefix,
      checkpoint_epoch: options?.checkpointEpoch,
      seed: options?.seed,
    }),
  })
}

export async function getSpeechGenerationStatus(jobId: string): Promise<AsyncSpeechStatus> {
  return request<AsyncSpeechStatus>(`/v1/text-to-speech/jobs/${jobId}`)
}

// ── Models ─────────────────────────────────────────────────────────────────────

export async function fetchModels(): Promise<Model[]> {
  return request<Model[]>('/v1/models')
}

// ── Training ───────────────────────────────────────────────────────────────────

export async function startTraining(
  voiceId: string,
  config: TrainingConfig,
  options?: TrainingStartOptions,
): Promise<{ job_id: string; round_id: string; dataset_snapshot_id: string; status: string }> {
  return request<{ job_id: string; round_id: string; dataset_snapshot_id: string; status: string }>('/v1/training/start', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      voice_id: voiceId,
      dataset_name: options?.datasetName,
      config,
    }),
  })
}

export async function fetchTrainingJob(jobId: string): Promise<TrainingJob> {
  return request<TrainingJob>(`/v1/training/${jobId}`)
}

export async function fetchTrainingCheckoutLedger(jobId: string): Promise<{ entries: TrainingCheckoutLedgerEntry[] }> {
  return request<{ entries: TrainingCheckoutLedgerEntry[] }>(`/v1/training/${jobId}/checkout-ledger`)
}

export async function fetchTrainingJobs(
  voiceId?: string,
  limit = 20,
): Promise<{ jobs: TrainingJob[] }> {
  const params = new URLSearchParams()
  params.set('limit', String(limit))
  if (voiceId) params.set('voice_id', voiceId)
  const query = params.toString()
  const path = query ? `/v1/training/jobs?${query}` : '/v1/training/jobs'
  return request<{ jobs: TrainingJob[] }>(path)
}

export async function fetchTrainingRounds(
  voiceId?: string,
  limit = 20,
): Promise<{ rounds: TrainingRound[] }> {
  const params = new URLSearchParams()
  params.set('limit', String(limit))
  if (voiceId) params.set('voice_id', voiceId)
  return request<{ rounds: TrainingRound[] }>(`/v1/training/rounds?${params.toString()}`)
}

export async function fetchDatasetSnapshots(
  voiceId?: string,
  limit = 20,
): Promise<{ snapshots: DatasetSnapshot[] }> {
  const params = new URLSearchParams()
  params.set('limit', String(limit))
  if (voiceId) params.set('voice_id', voiceId)
  return request<{ snapshots: DatasetSnapshot[] }>(`/v1/training/snapshots?${params.toString()}`)
}

export async function fetchTrainingAdvice(
  voiceId: string,
  limit = 40,
): Promise<{ advice: TrainingAdvice | null; voice_id: string; jobs_considered: number }> {
  const params = new URLSearchParams()
  params.set('voice_id', voiceId)
  params.set('limit', String(limit))
  return request<{ advice: TrainingAdvice | null; voice_id: string; jobs_considered: number }>(
    `/v1/training/advice?${params.toString()}`,
  )
}

export async function createTrainingCampaign(
  voiceId: string,
  options: CreateTrainingCampaignOptions,
): Promise<{ campaign: TrainingCampaign; attempts: TrainingJob[] }> {
  return request<{ campaign: TrainingCampaign; attempts: TrainingJob[] }>('/v1/training/campaigns', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      voice_id: voiceId,
      dataset_name: options.datasetName,
      attempt_count: options.attemptCount,
      parallelism: options.parallelism,
      base_config_overrides: options.baseConfigOverrides,
      stop_rules: options.stopRules,
    }),
  })
}

export async function fetchTrainingCampaign(
  campaignId: string,
): Promise<{ campaign: TrainingCampaign; attempts: TrainingJob[] }> {
  return request<{ campaign: TrainingCampaign; attempts: TrainingJob[] }>(`/v1/training/campaigns/${campaignId}`)
}

export async function cancelTrainingCampaign(
  campaignId: string,
): Promise<{ campaign: TrainingCampaign; attempts: TrainingJob[] }> {
  return request<{ campaign: TrainingCampaign; attempts: TrainingJob[] }>(`/v1/training/campaigns/${campaignId}/cancel`, {
    method: 'POST',
  })
}

export async function cancelTrainingJob(
  jobId: string,
): Promise<{ status: string }> {
  return request<{ status: string }>(`/v1/training/${jobId}/cancel`, {
    method: 'POST',
  })
}

export async function reconcileTrainingJob(
  jobId: string,
): Promise<{ status: string; validation_started?: boolean; job_id?: string }> {
  return request<{ status: string; validation_started?: boolean; job_id?: string }>(`/v1/training/${jobId}/reconcile`, {
    method: 'POST',
  })
}

export async function revalidateTrainingJob(
  jobId: string,
): Promise<{ status: string; job: TrainingJob }> {
  return request<{ status: string; job: TrainingJob }>(`/v1/training/${jobId}/revalidate`, {
    method: 'POST',
  })
}

export async function promoteTrainingCheckpoint(
  jobId: string,
  checkpointPrefix: string,
): Promise<{ status: string; job: TrainingJob; voice: Voice }> {
  return request<{ status: string; job: TrainingJob; voice: Voice }>(`/v1/training/${jobId}/promote`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ checkpoint_prefix: checkpointPrefix }),
  })
}

export async function fetchTrainingLogs(
  jobId: string,
  limit = 20,
  cursor?: number,
): Promise<{ job_id: string; chunks: TrainingLogChunk[]; next_cursor: number | null }> {
  const params = new URLSearchParams()
  params.set('limit', String(limit))
  if (typeof cursor === 'number') params.set('cursor', String(cursor))
  return request<{ job_id: string; chunks: TrainingLogChunk[]; next_cursor: number | null }>(
    `/v1/training/${jobId}/logs?${params.toString()}`,
  )
}

export async function fetchTrainingLogChunkText(
  jobId: string,
  seq: number,
): Promise<string> {
  return requestText(`/v1/training/${jobId}/logs/${seq}`)
}

export async function fetchTrainingPreprocessCache(
  jobId: string,
): Promise<TrainingPreprocessCacheResponse> {
  return request<TrainingPreprocessCacheResponse>(`/v1/training/${jobId}/preprocess-cache`)
}

export async function updateTrainingPreprocessCache(
  jobId: string,
  input: { reference_text: string },
): Promise<{ status: string; reference_text: string; updated_at: number }> {
  return request<{ status: string; reference_text: string; updated_at: number }>(
    `/v1/training/${jobId}/preprocess-cache`,
    {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(input),
    },
  )
}

export async function updateTrainingPreprocessEntry(
  jobId: string,
  entryId: string,
  input: { text?: string; included?: boolean },
): Promise<{
  status: string
  entry: DatasetPreprocessCacheEntry
  included_entries: number
  updated_at: number
}> {
  return request<{
    status: string
    entry: DatasetPreprocessCacheEntry
    included_entries: number
    updated_at: number
  }>(`/v1/training/${jobId}/preprocess-cache/entries/${entryId}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(input),
  })
}

export async function fetchVoiceDatasets(
  voiceId: string,
): Promise<{ datasets: DatasetInfo[] }> {
  return request<{ datasets: DatasetInfo[] }>(`/v1/dataset/${voiceId}`)
}

export async function selectVoiceDataset(
  voiceId: string,
  datasetName: string,
): Promise<{ status: string; dataset_name: string; ref_audio_r2_key: string }> {
  return request<{ status: string; dataset_name: string; ref_audio_r2_key: string }>(`/v1/dataset/${voiceId}/select`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ dataset_name: datasetName }),
  })
}

export async function fetchVoiceRawDatasetFiles(
  voiceId: string,
  limit = 500,
): Promise<{ voice_id: string; prefix: string; files: RawDatasetFile[] }> {
  return request<{ voice_id: string; prefix: string; files: RawDatasetFile[] }>(`/v1/dataset/${voiceId}/raw-files?limit=${limit}`)
}

export async function retranscribeDatasetEntries(
  voiceId: string,
  entries: Array<{ audio_r2_key: string; text?: string }>,
  languageCode?: string,
): Promise<{ voice_id: string; language_code: string; results: DatasetRetranscribeResult[] }> {
  return request<{ voice_id: string; language_code: string; results: DatasetRetranscribeResult[] }>(`/v1/dataset/${voiceId}/retranscribe`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      language_code: languageCode,
      entries,
    }),
  })
}

export async function reviewDatasetTexts(
  voiceId: string,
  entries: Array<{ segment?: string; text: string; duration?: number }>,
): Promise<{ voice_id: string; provider: string; results: DatasetReviewResult[] }> {
  return request<{ voice_id: string; provider: string; results: DatasetReviewResult[] }>(`/v1/dataset/${voiceId}/review-texts`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ entries }),
  })
}

export async function createFinalizedDataset(
  voiceId: string,
  input: {
    dataset_name: string
    items: DatasetDraftItem[]
    ref_audio_r2_key: string
    ref_text?: string
  },
): Promise<{
  dataset_name: string
  dataset_r2_prefix: string
  items_count: number
  ref_audio_r2_key: string
}> {
  return request(`/v1/dataset/${voiceId}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(input),
  })
}

// ── Upload ─────────────────────────────────────────────────────────────────────

export async function getPresignedUpload(input: {
  filename: string
  contentType: string
  voiceId: string
}): Promise<PresignedUpload> {
  return request<PresignedUpload>('/v1/upload/presigned', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      filename: input.filename,
      content_type: input.contentType,
      voice_id: input.voiceId,
    }),
  })
}

function getUploadErrorDetail(responseText: string, fallback: string): string {
  try {
    const body = JSON.parse(responseText) as Record<string, unknown>
    if (typeof body.detail === 'string') {
      return body.detail
    }
    if (
      typeof body.detail === 'object' &&
      body.detail !== null &&
      typeof (body.detail as { message?: unknown }).message === 'string'
    ) {
      return (body.detail as { message: string }).message
    }
  } catch {
    // ignore parse errors
  }
  return fallback
}

async function startMultipartUpload(
  voiceId: string,
  file: File,
): Promise<MultipartUploadSession> {
  return request<MultipartUploadSession>('/v1/upload/multipart/start', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      voice_id: voiceId,
      filename: file.name,
      content_type: file.type || 'application/octet-stream',
    }),
  })
}

async function uploadMultipartPart(
  input: {
    r2Key: string
    uploadId: string
    partNumber: number
    chunk: Blob
  },
  onProgress?: (progress: UploadProgress) => void,
): Promise<MultipartUploadPart> {
  const apiKey = getApiKey()

  return new Promise<MultipartUploadPart>((resolve, reject) => {
    const url = new URL(`${API_URL}/v1/upload/multipart/part`)
    url.searchParams.set('key', input.r2Key)
    url.searchParams.set('upload_id', input.uploadId)
    url.searchParams.set('part_number', String(input.partNumber))

    const xhr = new XMLHttpRequest()
    xhr.open('POST', url.toString())
    if (apiKey) {
      xhr.setRequestHeader('xi-api-key', apiKey)
    }

    xhr.upload.addEventListener('progress', (event) => {
      const totalBytes = input.chunk.size > 0 ? input.chunk.size : 1
      if (event.lengthComputable && event.total > 0) {
        onProgress?.({
          loadedBytes: Math.min(totalBytes, event.loaded),
          totalBytes,
        })
        return
      }
      onProgress?.({
        loadedBytes: Math.min(totalBytes, event.loaded),
        totalBytes,
      })
    })

    xhr.addEventListener('load', () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          const part = JSON.parse(xhr.responseText) as MultipartUploadPart
          onProgress?.({
            loadedBytes: input.chunk.size,
            totalBytes: input.chunk.size,
          })
          resolve(part)
        } catch {
          reject(new ApiError(`Upload failed for part ${input.partNumber}`, xhr.status || 500))
        }
        return
      }
      const fallback = `Upload failed for part ${input.partNumber}`
      const detail = getUploadErrorDetail(xhr.responseText, fallback)
      reject(new ApiError(detail, xhr.status || 500, detail))
    })

    xhr.addEventListener('error', () => {
      reject(new ApiError(`Upload failed for part ${input.partNumber}`, xhr.status || 0))
    })

    xhr.addEventListener('abort', () => {
      reject(new ApiError(`Upload cancelled for part ${input.partNumber}`, xhr.status || 0))
    })

    xhr.send(input.chunk)
  })
}

async function completeMultipartUpload(
  r2Key: string,
  uploadId: string,
  parts: MultipartUploadPart[],
): Promise<void> {
  await request<{ r2_key: string; size: number }>('/v1/upload/multipart/complete', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      r2_key: r2Key,
      upload_id: uploadId,
      parts,
    }),
  })
}

async function abortMultipartUpload(
  r2Key: string,
  uploadId: string,
): Promise<void> {
  try {
    await request<{ status: string }>('/v1/upload/multipart/abort', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        r2_key: r2Key,
        upload_id: uploadId,
      }),
    })
  } catch {
    // best-effort cleanup only
  }
}

async function uploadFileViaWorkerMultipart(
  voiceId: string,
  file: File,
  onProgress?: (progress: UploadProgress) => void,
): Promise<void> {
  const session = await startMultipartUpload(voiceId, file)
  const chunkSize = Math.max(5 * 1024 * 1024, session.chunk_size_bytes ?? DEFAULT_MULTIPART_CHUNK_SIZE_BYTES)
  const parts: MultipartUploadPart[] = []
  let completedBytes = 0

  try {
    for (let offset = 0, partNumber = 1; offset < file.size; offset += chunkSize, partNumber += 1) {
      const chunk = file.slice(offset, Math.min(offset + chunkSize, file.size))
      const part = await uploadMultipartPart(
        {
          r2Key: session.r2_key,
          uploadId: session.upload_id,
          partNumber,
          chunk,
        },
        (progress) => {
          onProgress?.({
            loadedBytes: Math.min(file.size, completedBytes + progress.loadedBytes),
            totalBytes: file.size,
          })
        },
      )
      parts.push(part)
      completedBytes = Math.min(file.size, offset + chunk.size)
      onProgress?.({
        loadedBytes: completedBytes,
        totalBytes: file.size,
      })
    }

    await completeMultipartUpload(session.r2_key, session.upload_id, parts)
  } catch (error) {
    await abortMultipartUpload(session.r2_key, session.upload_id)
    throw error
  }
}

async function uploadFileViaWorker(
  voiceId: string,
  file: File,
  onProgress?: (progress: UploadProgress) => void,
): Promise<void> {
  if (file.size >= MULTIPART_UPLOAD_THRESHOLD_BYTES) {
    await uploadFileViaWorkerMultipart(voiceId, file, onProgress)
    return
  }

  if (typeof XMLHttpRequest !== 'undefined') {
    const apiKey = getApiKey()
    await new Promise<void>((resolve, reject) => {
      const url = new URL(`${API_URL}/v1/upload/raw`)
      url.searchParams.set('voice_id', voiceId)
      url.searchParams.set('filename', file.name)
      url.searchParams.set('content_type', file.type || 'application/octet-stream')

      const xhr = new XMLHttpRequest()
      xhr.open('POST', url.toString())
      if (apiKey) {
        xhr.setRequestHeader('xi-api-key', apiKey)
      }
      if (file.type) {
        xhr.setRequestHeader('Content-Type', file.type)
      }

      xhr.upload.addEventListener('progress', (event) => {
        const totalBytes = file.size > 0 ? file.size : 1
        if (event.lengthComputable && event.total > 0) {
          const ratio = Math.max(0, Math.min(1, event.loaded / event.total))
          onProgress?.({
            loadedBytes: Math.min(totalBytes, Math.round(totalBytes * ratio)),
            totalBytes,
          })
          return
        }
        onProgress?.({
          loadedBytes: Math.min(totalBytes, event.loaded),
          totalBytes,
        })
      })

      xhr.addEventListener('load', () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          onProgress?.({
            loadedBytes: file.size,
            totalBytes: file.size,
          })
          resolve()
          return
        }
        const fallback = `Upload failed for ${file.name}`
        const detail = getUploadErrorDetail(xhr.responseText, fallback)
        reject(new ApiError(detail, xhr.status || 500, detail))
      })

      xhr.addEventListener('error', () => {
        reject(new ApiError(`Upload failed for ${file.name}`, xhr.status || 0))
      })

      xhr.addEventListener('abort', () => {
        reject(new ApiError(`Upload cancelled for ${file.name}`, xhr.status || 0))
      })

      xhr.send(file)
    })
    return
  }

  onProgress?.({
    loadedBytes: 0,
    totalBytes: file.size,
  })

  const url = new URL(`${API_URL}/v1/upload/raw`)
  url.searchParams.set('voice_id', voiceId)
  url.searchParams.set('filename', file.name)
  url.searchParams.set('content_type', file.type || 'application/octet-stream')

  const response = await fetch(url.toString(), {
    method: 'POST',
    headers: {
      ...authHeaders(),
      ...(file.type ? { 'Content-Type': file.type } : {}),
    },
    body: file,
  })

  if (!response.ok) {
    let detail = `Upload failed for ${file.name}`
    detail = getUploadErrorDetail(await response.text(), detail)
    throw new ApiError(detail, response.status, detail)
  }

  onProgress?.({
    loadedBytes: file.size,
    totalBytes: file.size,
  })
}

export async function uploadVoiceDatasetFile(
  voiceId: string,
  file: File,
  onProgress?: (progress: UploadProgress) => void,
): Promise<void> {
  await uploadFileViaWorker(voiceId, file, onProgress)
}

export async function uploadFileToR2(
  upload: PresignedUpload,
  file: File,
  voiceId?: string,
): Promise<void> {
  try {
    const response = await fetch(upload.upload_url, {
      method: 'PUT',
      headers: {
        'Content-Type': file.type || 'application/octet-stream',
      },
      body: file,
    })

    if (!response.ok) {
      throw new ApiError(`Upload failed for ${file.name}`, response.status)
    }
  } catch (error) {
    if (!voiceId) {
      throw error
    }
    await uploadFileViaWorker(voiceId, file)
  }
}

// ── Utility ────────────────────────────────────────────────────────────────────

export function formatDate(input: string | number): string {
  const d = new Date(input)
  return d.toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  })
}

export function formatTime(input: string | number): string {
  const d = new Date(input)
  return d.toLocaleTimeString('en-US', {
    hour: 'numeric',
    minute: '2-digit',
  })
}

export function formatDateTime(input: string | number): string {
  return `${formatDate(input)} ${formatTime(input)}`
}

export function formatDurationMs(input: number | null | undefined): string {
  if (typeof input !== 'number' || !Number.isFinite(input) || input < 0) {
    return '—'
  }

  const totalSeconds = Math.round(input / 1000)
  const hours = Math.floor(totalSeconds / 3600)
  const minutes = Math.floor((totalSeconds % 3600) / 60)
  const seconds = totalSeconds % 60

  if (hours > 0) {
    return `${hours}h ${minutes.toString().padStart(2, '0')}m`
  }
  if (minutes > 0) {
    return `${minutes}m ${seconds.toString().padStart(2, '0')}s`
  }
  return `${seconds}s`
}

export function formatDuration(seconds: number): string {
  const m = Math.floor(seconds / 60)
  const s = Math.floor(seconds % 60)
  return `${m}:${s.toString().padStart(2, '0')}`
}

export const DEFAULT_VOICE_SETTINGS: VoiceSettings = {
  stability: 0.85,
  similarity_boost: 0.85,
  style: 0.05,
  speed: 1.0,
}
