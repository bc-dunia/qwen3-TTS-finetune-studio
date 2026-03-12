export interface Env {
  R2: R2Bucket;
  DB: D1Database;
  AI?: {
    run(model: string, input: Record<string, unknown>): Promise<unknown>;
  };
  RUNPOD_API_KEY: string;
  RUNPOD_ENDPOINT_ID: string;
  RUNPOD_TRAINING_TEMPLATE_ID: string;
  RUNPOD_TRAINING_IMAGE_NAME?: string;
  RUNPOD_TRAINING_FALLBACK_IMAGE_NAMES?: string;
  RUNPOD_TRAINING_DOCKER_ARGS?: string;
  RUNPOD_TRAINING_VOLUME_MOUNT_PATH?: string;
  API_KEY: string;
  R2_ACCESS_KEY_ID: string;
  R2_SECRET_ACCESS_KEY: string;
  R2_ENDPOINT_URL: string;
  CORS_ORIGIN: string;
  ALLOW_ANONYMOUS_ACCESS?: string;
  OPENAI_API_KEY?: string;
  OPENAI_TRANSCRIBE_MODEL?: string;
  OPENAI_REVIEW_MODEL?: string;
  OPENAI_ADVISOR_MODEL?: string;
  WORKER_PUBLIC_URL?: string;
  TRAINING_MAX_ACTIVE_JOBS_PER_VOICE?: string;
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
  checkpoint_preset: string | null;
  checkpoint_score: number | null;
  checkpoint_job_id: string | null;
  candidate_checkpoint_r2_prefix: string | null;
  candidate_run_name: string | null;
  candidate_epoch: number | null;
  candidate_preset: string | null;
  candidate_score: number | null;
  candidate_job_id: string | null;
  active_round_id: string | null;
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
  style_prompt?: string;
  instruct?: string;
  checkpoint_prefix?: string;
  checkpoint_epoch?: number;
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

export type TrainingAdviceMode =
  | "compare-first"
  | "dataset-first"
  | "tone-explore"
  | "stability-reset"
  | "checkpoint-window"
  | "hold-current";

export interface TrainingAdvice {
  mode: TrainingAdviceMode;
  title: string;
  summary: string;
  confidence: "high" | "medium";
  reasons: string[];
  suggestedConfig: TrainingConfig | null;
  compareFirst: boolean;
  reviewDatasetFirst: boolean;
  primaryActionLabel?: string;
  analysisProvider?: "heuristic" | "llm";
}

export interface TrainingProgress {
  epoch?: number;
  step?: number;
  loss?: number;
  eta?: string;
  [key: string]: unknown;
}

export type TrainingCheckoutAdoptionMode = "promote" | "candidate" | "keep_current";

export type TrainingCheckoutSearchStatus =
  | "pending"
  | "validating"
  | "promoted"
  | "candidate_ready"
  | "kept_current"
  | "manual_promoted"
  | "rejected"
  | "failed";

export interface TrainingCheckoutTarget {
  prefix: string;
  epoch: number | null;
  preset: string | null;
  score: number | null;
  run_name: string | null;
}

export interface TrainingCheckoutEvaluation {
  epoch: number;
  prefix: string;
  ok: boolean;
  score: number;
  message: string;
  preset: string;
  passed_samples: number;
  total_samples: number;
  run_name: string | null;
  is_champion: boolean;
  is_selected: boolean;
}

export interface TrainingCheckoutSearch {
  status: TrainingCheckoutSearchStatus;
  validation_checked: boolean;
  validation_passed: boolean;
  validation_in_progress: boolean;
  has_candidates: boolean;
  compare_ready: boolean;
  adoption_mode: TrainingCheckoutAdoptionMode | null;
  message: string | null;
  last_message: string | null;
  champion: TrainingCheckoutTarget | null;
  selected: TrainingCheckoutTarget | null;
  manual_promoted: TrainingCheckoutTarget | null;
  evaluated: TrainingCheckoutEvaluation[];
}

export interface TrainingJob {
  job_id: string;
  voice_id: string;
  campaign_id?: string | null;
  attempt_index?: number | null;
  round_id: string | null;
  dataset_snapshot_id: string | null;
  runpod_pod_id: string | null;
  job_token?: string | null;
  status: string;
  config: TrainingConfig;
  progress: TrainingProgress;
  summary: Record<string, unknown>;
  checkout_search?: TrainingCheckoutSearch;
  metrics: Record<string, unknown>;
  supervisor: Record<string, unknown>;
  dataset_r2_prefix: string;
  log_r2_prefix: string | null;
  error_message: string | null;
  last_heartbeat_at: number | null;
  started_at: number | null;
  completed_at: number | null;
  created_at: number;
  updated_at: number;
}

export type TrainingCampaignStatus =
  | "planning"
  | "running"
  | "completed"
  | "failed"
  | "blocked_dataset"
  | "blocked_budget"
  | "cancelled";

export interface TrainingCampaignStopRules {
  max_infra_failures?: number;
  max_asr_failures?: number;
  min_score_improvement?: number;
  stagnation_window?: number;
}

export interface TrainingCampaign {
  campaign_id: string;
  voice_id: string;
  dataset_name: string | null;
  dataset_r2_prefix: string | null;
  dataset_snapshot_id: string | null;
  attempt_count: number;
  parallelism: number;
  status: TrainingCampaignStatus;
  base_config: TrainingConfig;
  stop_rules: TrainingCampaignStopRules;
  planner_state: Record<string, unknown>;
  summary: Record<string, unknown>;
  created_at: number;
  updated_at: number;
  completed_at: number | null;
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
  ref_text?: string;
}

export interface DatasetInfo {
  name: string;
  r2_prefix: string;
  file_count: number;
}

export interface DatasetSnapshot {
  snapshot_id: string;
  voice_id: string;
  dataset_name: string | null;
  dataset_r2_prefix: string;
  dataset_signature: string;
  status: string;
  source_cache_id: string | null;
  cache_r2_prefix: string | null;
  train_raw_r2_key: string | null;
  ref_audio_r2_key: string | null;
  reference_profile_r2_key: string | null;
  reference_text: string | null;
  source_file_count: number | null;
  segments_created: number | null;
  segments_accepted: number | null;
  accepted_duration_min: number | null;
  created_from_job_id: string | null;
  created_at: number;
  updated_at: number;
}

export interface TrainingRound {
  round_id: string;
  voice_id: string;
  dataset_snapshot_id: string | null;
  round_index: number;
  status: string;
  production_checkpoint_r2_prefix: string | null;
  production_run_name: string | null;
  production_epoch: number | null;
  production_preset: string | null;
  production_score: number | null;
  production_job_id: string | null;
  champion_checkpoint_r2_prefix: string | null;
  champion_run_name: string | null;
  champion_epoch: number | null;
  champion_preset: string | null;
  champion_score: number | null;
  champion_job_id: string | null;
  selected_checkpoint_r2_prefix: string | null;
  selected_run_name: string | null;
  selected_epoch: number | null;
  selected_preset: string | null;
  selected_score: number | null;
  selected_job_id: string | null;
  adoption_mode: string | null;
  candidate_checkpoint_r2_prefix: string | null;
  candidate_run_name: string | null;
  candidate_epoch: number | null;
  candidate_score: number | null;
  candidate_job_id: string | null;
  summary: Record<string, unknown>;
  created_at: number;
  updated_at: number;
  started_at: number | null;
  completed_at: number | null;
}
