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
  settings: VoiceSettings
  labels: Record<string, string>
  created_at: string
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

export interface TrainingProgress {
  epoch?: number
  total_epochs?: number
  loss?: number
  step?: number
  total_steps?: number
  [key: string]: unknown
}

export interface TrainingConfig {
  batch_size: number
  num_epochs: number
  learning_rate: number
}

export interface TrainingJob {
  job_id: string
  status:
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
  config: TrainingConfig
  summary?: Record<string, unknown>
  metrics?: Record<string, unknown>
}

export interface PresignedUpload {
  upload_url: string
  r2_key: string
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
      const hasKey = getApiKey().length > 0
      detail = hasKey
        ? 'Invalid API key. Check the key in the left sidebar.'
        : 'Missing API key. Enter your xi-api-key in the left sidebar.'
    }

    throw new ApiError(detail, response.status, detail)
  }

  return response.json() as Promise<T>
}

// ── Voices ─────────────────────────────────────────────────────────────────────

export async function fetchVoices(): Promise<VoicesResponse> {
  return request<VoicesResponse>('/v1/voices')
}

export async function fetchVoice(voiceId: string): Promise<Voice> {
  return request<Voice>(`/v1/voices/${voiceId}`)
}

export async function createVoice(
  name: string,
  description: string,
  audioFile: File,
  modelSize: VoiceModelSize = '1.7B',
): Promise<{ voice_id: string }> {
  const formData = new FormData()
  formData.append('name', name)
  formData.append('description', description)
  formData.append('model_size', modelSize)
  formData.append('files', audioFile)

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
): Promise<AsyncSpeechJob> {
  return request<AsyncSpeechJob>(`/v1/text-to-speech/${voiceId}/async`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, voice_settings: voiceSettings }),
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
): Promise<{ job_id: string; status: string }> {
  return request<{ job_id: string; status: string }>('/v1/training/start', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ voice_id: voiceId, config }),
  })
}

export async function fetchTrainingJob(jobId: string): Promise<TrainingJob> {
  return request<TrainingJob>(`/v1/training/${jobId}`)
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

export async function cancelTrainingJob(
  jobId: string,
): Promise<{ status: string }> {
  return request<{ status: string }>(`/v1/training/${jobId}/cancel`, {
    method: 'POST',
  })
}

// ── Upload ─────────────────────────────────────────────────────────────────────

export async function getPresignedUpload(): Promise<PresignedUpload> {
  return request<PresignedUpload>('/v1/upload/presigned', {
    method: 'POST',
  })
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

export function formatDuration(seconds: number): string {
  const m = Math.floor(seconds / 60)
  const s = Math.floor(seconds % 60)
  return `${m}:${s.toString().padStart(2, '0')}`
}

export const DEFAULT_VOICE_SETTINGS: VoiceSettings = {
  stability: 0.5,
  similarity_boost: 0.75,
  style: 0.0,
  speed: 1.0,
}
