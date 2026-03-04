export interface Env {
  R2: R2Bucket;
  DB: D1Database;
  RUNPOD_API_KEY: string;
  RUNPOD_ENDPOINT_ID: string;
  RUNPOD_TRAINING_TEMPLATE_ID: string;
  API_KEY: string;
  R2_ACCESS_KEY_ID: string;
  R2_SECRET_ACCESS_KEY: string;
  R2_ENDPOINT_URL: string;
  CORS_ORIGIN: string;
}

export interface VoiceSettings {
  stability?: number;
  similarity_boost?: number;
  style?: number;
  use_speaker_boost?: boolean;
  speed?: number;
}

export interface Voice {
  voice_id: string;
  name: string;
  description: string;
  speaker_name: string;
  model_size: string;
  model_id: string;
  category: string;
  status: string;
  checkpoint_r2_prefix: string | null;
  run_name: string | null;
  epoch: number | null;
  sample_audio_r2_key: string | null;
  ref_audio_r2_key: string | null;
  labels: Record<string, string>;
  settings: VoiceSettings;
  preview_url: string | null;
  created_at: number;
  updated_at: number;
}

export interface TTSRequest {
  text: string;
  model_id?: string;
  voice_settings?: VoiceSettings;
  seed?: number;
  language_code?: string;
}

export interface Model {
  model_id: string;
  name: string;
  description: string | null;
  can_do_text_to_speech: boolean;
  can_be_finetuned: boolean;
  max_characters_request: number;
  languages: Array<{ language_id: string; name: string }>;
}

export interface Generation {
  generation_id: string;
  voice_id: string;
  model_id: string;
  text: string;
  audio_r2_key: string | null;
  output_format: string;
  duration_ms: number | null;
  latency_ms: number | null;
  settings: VoiceSettings;
  created_at: number;
}

export interface TrainingConfig {
  batch_size?: number;
  learning_rate?: number;
  num_epochs?: number;
  model_size?: string;
  gpu_type_id?: string;
  [key: string]: unknown;
}

export interface TrainingProgress {
  epoch?: number;
  step?: number;
  loss?: number;
  eta?: string;
  [key: string]: unknown;
}

export interface TrainingJob {
  job_id: string;
  voice_id: string;
  runpod_pod_id: string | null;
  job_token?: string | null;
  status: string;
  config: TrainingConfig;
  progress: TrainingProgress;
  summary: Record<string, unknown>;
  metrics: Record<string, unknown>;
  dataset_r2_prefix: string;
  log_r2_prefix: string | null;
  error_message: string | null;
  last_heartbeat_at: number | null;
  started_at: number | null;
  completed_at: number | null;
  created_at: number;
  updated_at: number;
}

export interface TTSResponse {
  audio: string;
  sample_rate: number;
  duration_ms: number;
}

export interface VoicesListResponse {
  voices: Voice[];
  has_more: boolean;
  total_count: number;
}

export interface ErrorResponse {
  detail: {
    status?: "error";
    message: string;
  };
}

export interface AppContext {
  Bindings: Env;
}

export interface DatasetItem {
  audio_r2_key: string;
  text: string;
}

export interface CreateDatasetRequest {
  dataset_name: string;
  items: DatasetItem[];
  ref_audio_r2_key: string;
}

export interface DatasetInfo {
  name: string;
  r2_prefix: string;
  file_count: number;
}
