import type {
  DatasetSnapshot,
  Generation,
  TrainingCampaign,
  TrainingCampaignStatus,
  TrainingCampaignStopRules,
  TrainingConfig,
  TrainingJob,
  TrainingProgress,
  TrainingRound,
  Voice,
  VoiceSettings,
} from "../types";

type DbVoiceRow = {
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
  labels_json: string;
  settings_json: string;
  created_at: number;
  updated_at: number;
};

type DbTrainingRow = {
  job_id: string;
  voice_id: string;
  campaign_id: string | null;
  attempt_index: number | null;
  round_id: string | null;
  dataset_snapshot_id: string | null;
  runpod_pod_id: string | null;
  job_token: string | null;
  status: string;
  config_json: string;
  progress_json: string;
  summary_json: string | null;
  metrics_json: string | null;
  supervisor_json: string | null;
  dataset_r2_prefix: string;
  log_r2_prefix: string | null;
  error_message: string | null;
  last_heartbeat_at: number | null;
  started_at: number | null;
  completed_at: number | null;
  created_at: number;
  updated_at: number;
};

type DbTrainingCampaignRow = {
  campaign_id: string;
  voice_id: string;
  dataset_name: string | null;
  dataset_r2_prefix: string | null;
  dataset_snapshot_id: string | null;
  attempt_count: number;
  parallelism: number;
  status: TrainingCampaignStatus;
  base_config_json: string;
  stop_rules_json: string;
  planner_state_json: string;
  summary_json: string;
  created_at: number;
  updated_at: number;
  completed_at: number | null;
};

type DbTrainingLogChunkRow = {
  job_id: string;
  seq: number;
  r2_key: string;
  created_at: number;
  bytes: number | null;
  lines: number | null;
};

type DbDatasetPreprocessCacheRow = {
  cache_id: string;
  voice_id: string;
  dataset_r2_prefix: string;
  dataset_signature: string;
  cache_r2_prefix: string;
  train_raw_r2_key: string;
  ref_audio_r2_key: string | null;
  reference_profile_r2_key: string | null;
  source_file_count: number | null;
  segments_created: number | null;
  segments_accepted: number | null;
  accepted_duration_min: number | null;
  created_at: number;
  updated_at: number;
};

type DbDatasetPreprocessCacheEntryRow = {
  entry_id: string;
  cache_id: string;
  seq: number;
  audio_path: string;
  audio_r2_key: string;
  text: string;
  included: number;
  created_at: number;
  updated_at: number;
};

type DbDatasetSnapshotRow = {
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
};

type DbTrainingRoundRow = {
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
  summary_json: string | null;
  created_at: number;
  updated_at: number;
  started_at: number | null;
  completed_at: number | null;
};

export type TrainingCheckoutLedgerEntry = {
  entry_id: string;
  round_id: string | null;
  job_id: string;
  voice_id: string;
  checkpoint_r2_prefix: string;
  run_name: string | null;
  epoch: number | null;
  preset: string | null;
  score: number | null;
  ok: boolean | null;
  passed_samples: number | null;
  total_samples: number | null;
  message: string | null;
  role: string;
  source: string;
  adoption_mode: string | null;
  created_at: number;
  updated_at: number;
};

type DbTrainingCheckoutLedgerRow = {
  entry_id: string;
  round_id: string | null;
  job_id: string;
  voice_id: string;
  checkpoint_r2_prefix: string;
  run_name: string | null;
  epoch: number | null;
  preset: string | null;
  score: number | null;
  ok: number | null;
  passed_samples: number | null;
  total_samples: number | null;
  message: string | null;
  role: string;
  source: string;
  adoption_mode: string | null;
  created_at: number;
  updated_at: number;
};

const parseJson = <T>(value: string | null | undefined, fallback: T): T => {
  if (!value) {
    return fallback;
  }
  try {
    return JSON.parse(value) as T;
  } catch {
    return fallback;
  }
};

const mapVoice = (row: DbVoiceRow): Voice => ({
  voice_id: row.voice_id,
  name: row.name,
  description: row.description,
  speaker_name: row.speaker_name,
  model_size: row.model_size,
  model_id: row.model_id,
  category: row.category,
  status: row.status,
  checkpoint_r2_prefix: row.checkpoint_r2_prefix,
  run_name: row.run_name,
  epoch: row.epoch,
  checkpoint_preset: row.checkpoint_preset,
  checkpoint_score: row.checkpoint_score,
  checkpoint_job_id: row.checkpoint_job_id,
  candidate_checkpoint_r2_prefix: row.candidate_checkpoint_r2_prefix,
  candidate_run_name: row.candidate_run_name,
  candidate_epoch: row.candidate_epoch,
  candidate_preset: row.candidate_preset,
  candidate_score: row.candidate_score,
  candidate_job_id: row.candidate_job_id,
  active_round_id: row.active_round_id,
  sample_audio_r2_key: row.sample_audio_r2_key,
  ref_audio_r2_key: row.ref_audio_r2_key,
  labels: parseJson<Record<string, string>>(row.labels_json, {}),
  settings: parseJson<VoiceSettings>(row.settings_json, {}),
  preview_url: row.sample_audio_r2_key
    ? `/v1/audio/${row.voice_id}/${row.sample_audio_r2_key.split("/").pop()}`
    : null,
  created_at: row.created_at,
  updated_at: row.updated_at,
});

const mapTrainingJob = (row: DbTrainingRow): TrainingJob => ({
  job_id: row.job_id,
  voice_id: row.voice_id,
  campaign_id: row.campaign_id,
  attempt_index: row.attempt_index,
  round_id: row.round_id,
  dataset_snapshot_id: row.dataset_snapshot_id,
  runpod_pod_id: row.runpod_pod_id,
  job_token: row.job_token,
  status: row.status,
  config: parseJson<TrainingConfig>(row.config_json, {}),
  progress: parseJson<TrainingProgress>(row.progress_json, {}),
  summary: parseJson<Record<string, unknown>>(row.summary_json, {}),
  metrics: parseJson<Record<string, unknown>>(row.metrics_json, {}),
  supervisor: parseJson<Record<string, unknown>>(row.supervisor_json, {}),
  dataset_r2_prefix: row.dataset_r2_prefix,
  log_r2_prefix: row.log_r2_prefix,
  error_message: row.error_message,
  last_heartbeat_at: row.last_heartbeat_at,
  started_at: row.started_at,
  completed_at: row.completed_at,
  created_at: row.created_at,
  updated_at: row.updated_at,
});

export type TrainingLogChunk = {
  job_id: string;
  seq: number;
  r2_key: string;
  created_at: number;
  bytes: number | null;
  lines: number | null;
};

export type DatasetPreprocessCache = {
  cache_id: string;
  voice_id: string;
  dataset_r2_prefix: string;
  dataset_signature: string;
  cache_r2_prefix: string;
  train_raw_r2_key: string;
  ref_audio_r2_key: string | null;
  reference_profile_r2_key: string | null;
  source_file_count: number | null;
  segments_created: number | null;
  segments_accepted: number | null;
  accepted_duration_min: number | null;
  created_at: number;
  updated_at: number;
};

export type DatasetPreprocessCacheEntry = {
  entry_id: string;
  cache_id: string;
  seq: number;
  audio_path: string;
  audio_r2_key: string;
  text: string;
  included: boolean;
  created_at: number;
  updated_at: number;
};

const mapTrainingLogChunk = (row: DbTrainingLogChunkRow): TrainingLogChunk => ({
  job_id: row.job_id,
  seq: row.seq,
  r2_key: row.r2_key,
  created_at: row.created_at,
  bytes: row.bytes,
  lines: row.lines,
});

const mapDatasetPreprocessCache = (
  row: DbDatasetPreprocessCacheRow
): DatasetPreprocessCache => ({
  cache_id: row.cache_id,
  voice_id: row.voice_id,
  dataset_r2_prefix: row.dataset_r2_prefix,
  dataset_signature: row.dataset_signature,
  cache_r2_prefix: row.cache_r2_prefix,
  train_raw_r2_key: row.train_raw_r2_key,
  ref_audio_r2_key: row.ref_audio_r2_key,
  reference_profile_r2_key: row.reference_profile_r2_key,
  source_file_count: row.source_file_count,
  segments_created: row.segments_created,
  segments_accepted: row.segments_accepted,
  accepted_duration_min: row.accepted_duration_min,
  created_at: row.created_at,
  updated_at: row.updated_at,
});

const mapDatasetPreprocessCacheEntry = (
  row: DbDatasetPreprocessCacheEntryRow
): DatasetPreprocessCacheEntry => ({
  entry_id: row.entry_id,
  cache_id: row.cache_id,
  seq: row.seq,
  audio_path: row.audio_path,
  audio_r2_key: row.audio_r2_key,
  text: row.text,
  included: row.included === 1,
  created_at: row.created_at,
  updated_at: row.updated_at,
});

const mapDatasetSnapshot = (row: DbDatasetSnapshotRow): DatasetSnapshot => ({
  snapshot_id: row.snapshot_id,
  voice_id: row.voice_id,
  dataset_name: row.dataset_name,
  dataset_r2_prefix: row.dataset_r2_prefix,
  dataset_signature: row.dataset_signature,
  status: row.status,
  source_cache_id: row.source_cache_id,
  cache_r2_prefix: row.cache_r2_prefix,
  train_raw_r2_key: row.train_raw_r2_key,
  ref_audio_r2_key: row.ref_audio_r2_key,
  reference_profile_r2_key: row.reference_profile_r2_key,
  reference_text: row.reference_text,
  source_file_count: row.source_file_count,
  segments_created: row.segments_created,
  segments_accepted: row.segments_accepted,
  accepted_duration_min: row.accepted_duration_min,
  created_from_job_id: row.created_from_job_id,
  created_at: row.created_at,
  updated_at: row.updated_at,
});

const mapTrainingRound = (row: DbTrainingRoundRow): TrainingRound => ({
  round_id: row.round_id,
  voice_id: row.voice_id,
  dataset_snapshot_id: row.dataset_snapshot_id,
  round_index: row.round_index,
  status: row.status,
  production_checkpoint_r2_prefix: row.production_checkpoint_r2_prefix,
  production_run_name: row.production_run_name,
  production_epoch: row.production_epoch,
  production_preset: row.production_preset,
  production_score: row.production_score,
  production_job_id: row.production_job_id,
  champion_checkpoint_r2_prefix: row.champion_checkpoint_r2_prefix,
  champion_run_name: row.champion_run_name,
  champion_epoch: row.champion_epoch,
  champion_preset: row.champion_preset,
  champion_score: row.champion_score,
  champion_job_id: row.champion_job_id,
  selected_checkpoint_r2_prefix: row.selected_checkpoint_r2_prefix,
  selected_run_name: row.selected_run_name,
  selected_epoch: row.selected_epoch,
  selected_preset: row.selected_preset,
  selected_score: row.selected_score,
  selected_job_id: row.selected_job_id,
  adoption_mode: row.adoption_mode,
  candidate_checkpoint_r2_prefix: row.candidate_checkpoint_r2_prefix,
  candidate_run_name: row.candidate_run_name,
  candidate_epoch: row.candidate_epoch,
  candidate_score: row.candidate_score,
  candidate_job_id: row.candidate_job_id,
  summary: parseJson<Record<string, unknown>>(row.summary_json, {}),
  created_at: row.created_at,
  updated_at: row.updated_at,
  started_at: row.started_at,
  completed_at: row.completed_at,
});

const mapTrainingCampaign = (row: DbTrainingCampaignRow): TrainingCampaign => ({
  campaign_id: row.campaign_id,
  voice_id: row.voice_id,
  dataset_name: row.dataset_name,
  dataset_r2_prefix: row.dataset_r2_prefix,
  dataset_snapshot_id: row.dataset_snapshot_id,
  attempt_count: row.attempt_count,
  parallelism: row.parallelism,
  status: row.status,
  base_config: parseJson<TrainingConfig>(row.base_config_json, {}),
  stop_rules: parseJson<TrainingCampaignStopRules>(row.stop_rules_json, {}),
  planner_state: parseJson<Record<string, unknown>>(row.planner_state_json, {}),
  summary: parseJson<Record<string, unknown>>(row.summary_json, {}),
  created_at: row.created_at,
  updated_at: row.updated_at,
  completed_at: row.completed_at,
});

export const getVoice = async (db: D1Database, voiceId: string): Promise<Voice | null> => {
  const row = await db
    .prepare("SELECT * FROM voices WHERE voice_id = ? LIMIT 1")
    .bind(voiceId)
    .first<DbVoiceRow>();

  if (!row) {
    return null;
  }

  return mapVoice(row);
};

export const listVoices = async (
  db: D1Database,
  filters: { search?: string; category?: string; status?: string } = {}
): Promise<Voice[]> => {
  const conditions: string[] = [];
  const bindings: string[] = [];

  if (filters.search) {
    conditions.push("name LIKE ?");
    bindings.push(`%${filters.search}%`);
  }
  if (filters.category) {
    conditions.push("category = ?");
    bindings.push(filters.category);
  }
  if (filters.status) {
    conditions.push("status = ?");
    bindings.push(filters.status);
  }

  const whereClause = conditions.length > 0 ? `WHERE ${conditions.join(" AND ")}` : "";
  const sql = `SELECT * FROM voices ${whereClause} ORDER BY updated_at DESC, created_at DESC`;

  const result = await db.prepare(sql).bind(...bindings).all<DbVoiceRow>();
  return (result.results ?? []).map(mapVoice);
};

export const createVoice = async (db: D1Database, voice: Voice): Promise<void> => {
  await db
    .prepare(
      `INSERT INTO voices (
        voice_id, name, description, speaker_name, model_size, model_id, category, status,
        checkpoint_r2_prefix, run_name, epoch, checkpoint_preset, checkpoint_score, checkpoint_job_id,
        candidate_checkpoint_r2_prefix, candidate_run_name, candidate_epoch, candidate_preset, candidate_score, candidate_job_id, active_round_id,
        sample_audio_r2_key, ref_audio_r2_key, labels_json, settings_json, created_at, updated_at
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
    )
    .bind(
      voice.voice_id,
      voice.name,
      voice.description,
      voice.speaker_name,
      voice.model_size,
      voice.model_id,
      voice.category,
      voice.status,
      voice.checkpoint_r2_prefix,
      voice.run_name,
      voice.epoch,
      voice.checkpoint_preset,
      voice.checkpoint_score,
      voice.checkpoint_job_id,
      voice.candidate_checkpoint_r2_prefix,
      voice.candidate_run_name,
      voice.candidate_epoch,
      voice.candidate_preset,
      voice.candidate_score,
      voice.candidate_job_id,
      voice.active_round_id,
      voice.sample_audio_r2_key,
      voice.ref_audio_r2_key,
      JSON.stringify(voice.labels),
      JSON.stringify(voice.settings),
      voice.created_at,
      voice.updated_at
    )
    .run();
};

export const updateVoice = async (
  db: D1Database,
  voiceId: string,
  updates: {
    status?: string;
    checkpoint_r2_prefix?: string | null;
    run_name?: string | null;
    epoch?: number | null;
    checkpoint_preset?: string | null;
    checkpoint_score?: number | null;
    checkpoint_job_id?: string | null;
    candidate_checkpoint_r2_prefix?: string | null;
    candidate_run_name?: string | null;
    candidate_epoch?: number | null;
    candidate_preset?: string | null;
    candidate_score?: number | null;
    candidate_job_id?: string | null;
    active_round_id?: string | null;
    sample_audio_r2_key?: string | null;
    ref_audio_r2_key?: string | null;
    settings?: VoiceSettings;
    updated_at?: number;
  }
): Promise<void> => {
  const fields: string[] = [];
  const bindings: Array<string | number | null> = [];

  if (updates.status !== undefined) {
    fields.push("status = ?");
    bindings.push(updates.status);
  }
  if (updates.checkpoint_r2_prefix !== undefined) {
    fields.push("checkpoint_r2_prefix = ?");
    bindings.push(updates.checkpoint_r2_prefix);
  }
  if (updates.run_name !== undefined) {
    fields.push("run_name = ?");
    bindings.push(updates.run_name);
  }
  if (updates.epoch !== undefined) {
    fields.push("epoch = ?");
    bindings.push(updates.epoch);
  }
  if (updates.checkpoint_preset !== undefined) {
    fields.push("checkpoint_preset = ?");
    bindings.push(updates.checkpoint_preset);
  }
  if (updates.checkpoint_score !== undefined) {
    fields.push("checkpoint_score = ?");
    bindings.push(updates.checkpoint_score);
  }
  if (updates.checkpoint_job_id !== undefined) {
    fields.push("checkpoint_job_id = ?");
    bindings.push(updates.checkpoint_job_id);
  }
  if (updates.candidate_checkpoint_r2_prefix !== undefined) {
    fields.push("candidate_checkpoint_r2_prefix = ?");
    bindings.push(updates.candidate_checkpoint_r2_prefix);
  }
  if (updates.candidate_run_name !== undefined) {
    fields.push("candidate_run_name = ?");
    bindings.push(updates.candidate_run_name);
  }
  if (updates.candidate_epoch !== undefined) {
    fields.push("candidate_epoch = ?");
    bindings.push(updates.candidate_epoch);
  }
  if (updates.candidate_preset !== undefined) {
    fields.push("candidate_preset = ?");
    bindings.push(updates.candidate_preset);
  }
  if (updates.candidate_score !== undefined) {
    fields.push("candidate_score = ?");
    bindings.push(updates.candidate_score);
  }
  if (updates.candidate_job_id !== undefined) {
    fields.push("candidate_job_id = ?");
    bindings.push(updates.candidate_job_id);
  }
  if (updates.active_round_id !== undefined) {
    fields.push("active_round_id = ?");
    bindings.push(updates.active_round_id);
  }
  if (updates.sample_audio_r2_key !== undefined) {
    fields.push("sample_audio_r2_key = ?");
    bindings.push(updates.sample_audio_r2_key);
  }
  if (updates.ref_audio_r2_key !== undefined) {
    fields.push("ref_audio_r2_key = ?");
    bindings.push(updates.ref_audio_r2_key);
  }
  if (updates.settings !== undefined) {
    fields.push("settings_json = ?");
    bindings.push(JSON.stringify(updates.settings));
  }

  fields.push("updated_at = ?");
  bindings.push(updates.updated_at ?? Date.now());

  if (fields.length === 0) {
    return;
  }

  bindings.push(voiceId);
  await db
    .prepare(`UPDATE voices SET ${fields.join(", ")} WHERE voice_id = ?`)
    .bind(...bindings)
    .run();
};

export const deleteVoice = async (db: D1Database, voiceId: string): Promise<void> => {
  await db.prepare("DELETE FROM voices WHERE voice_id = ?").bind(voiceId).run();
};

export const createGeneration = async (db: D1Database, generation: Generation): Promise<void> => {
  await db
    .prepare(
      `INSERT INTO generations (
        generation_id, voice_id, model_id, text, audio_r2_key, output_format,
        duration_ms, latency_ms, settings_json, created_at
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
    )
    .bind(
      generation.generation_id,
      generation.voice_id,
      generation.model_id,
      generation.text,
      generation.audio_r2_key,
      generation.output_format,
      generation.duration_ms,
      generation.latency_ms,
      JSON.stringify(generation.settings),
      generation.created_at
    )
    .run();
};

export const getTrainingJob = async (
  db: D1Database,
  jobId: string
): Promise<TrainingJob | null> => {
  const row = await db
    .prepare("SELECT * FROM training_jobs WHERE job_id = ? LIMIT 1")
    .bind(jobId)
    .first<DbTrainingRow>();

  if (!row) {
    return null;
  }

  return mapTrainingJob(row);
};

export const getTrainingJobByToken = async (
  db: D1Database,
  jobId: string,
  jobToken: string
): Promise<TrainingJob | null> => {
  const row = await db
    .prepare("SELECT * FROM training_jobs WHERE job_id = ? AND job_token = ? LIMIT 1")
    .bind(jobId, jobToken)
    .first<DbTrainingRow>();

  if (!row) {
    return null;
  }

  return mapTrainingJob(row);
};

export const listTrainingJobs = async (
  db: D1Database,
  filters: { voice_id?: string; campaign_id?: string; limit?: number } = {}
): Promise<TrainingJob[]> => {
  const conditions: string[] = [];
  const bindings: Array<string | number> = [];

  if (filters.voice_id) {
    conditions.push("voice_id = ?");
    bindings.push(filters.voice_id);
  }
  if (filters.campaign_id) {
    conditions.push("campaign_id = ?");
    bindings.push(filters.campaign_id);
  }

  const whereClause = conditions.length > 0 ? `WHERE ${conditions.join(" AND ")}` : "";
  const limit = Math.max(1, Math.min(filters.limit ?? 20, 100));
  const sql = `SELECT * FROM training_jobs ${whereClause} ORDER BY created_at DESC LIMIT ?`;

  bindings.push(limit);
  const result = await db.prepare(sql).bind(...bindings).all<DbTrainingRow>();
  return (result.results ?? []).map(mapTrainingJob);
};

export const countActiveTrainingJobs = async (
  db: D1Database,
  statusList: string[]
): Promise<number> => {
  if (statusList.length === 0) return 0;
  const placeholders = statusList.map(() => "?").join(", ");
  const sql = `SELECT COUNT(*) as cnt FROM training_jobs WHERE status IN (${placeholders})`;
  const result = await db.prepare(sql).bind(...statusList).first<{ cnt: number }>();
  return result?.cnt ?? 0;
};

export const createTrainingJob = async (db: D1Database, job: TrainingJob): Promise<void> => {
  await db
    .prepare(
      `INSERT INTO training_jobs (
        job_id, voice_id, campaign_id, attempt_index, round_id, dataset_snapshot_id, runpod_pod_id, job_token, status, config_json, progress_json,
        summary_json, metrics_json, supervisor_json, dataset_r2_prefix, log_r2_prefix, error_message,
        last_heartbeat_at, started_at, completed_at, created_at, updated_at
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
    )
    .bind(
      job.job_id,
      job.voice_id,
      job.campaign_id ?? null,
      job.attempt_index ?? null,
      job.round_id ?? null,
      job.dataset_snapshot_id ?? null,
      job.runpod_pod_id ?? null,
      job.job_token ?? null,
      job.status,
      JSON.stringify(job.config),
      JSON.stringify(job.progress),
      JSON.stringify(job.summary),
      JSON.stringify(job.metrics),
      JSON.stringify(job.supervisor ?? {}),
      job.dataset_r2_prefix,
      job.log_r2_prefix ?? null,
      job.error_message ?? null,
      job.last_heartbeat_at ?? null,
      job.started_at ?? null,
      job.completed_at ?? null,
      job.created_at,
      job.updated_at
    )
    .run();
};

export const updateTrainingJob = async (
  db: D1Database,
  jobId: string,
  updates: {
    campaign_id?: string | null;
    attempt_index?: number | null;
    round_id?: string | null;
    dataset_snapshot_id?: string | null;
    runpod_pod_id?: string | null;
    job_token?: string | null;
    status?: string;
    config?: TrainingConfig;
    progress?: TrainingProgress;
    summary?: Record<string, unknown>;
    metrics?: Record<string, unknown>;
    supervisor?: Record<string, unknown>;
    log_r2_prefix?: string | null;
    error_message?: string | null;
    last_heartbeat_at?: number | null;
    started_at?: number | null;
    completed_at?: number | null;
    updated_at?: number;
    expected_updated_at?: number;
  }
): Promise<void> => {
  const fields: string[] = [];
  const bindings: Array<string | number | null> = [];

  if (updates.round_id !== undefined) {
    fields.push("round_id = ?");
    bindings.push(updates.round_id);
  }
  if (updates.campaign_id !== undefined) {
    fields.push("campaign_id = ?");
    bindings.push(updates.campaign_id);
  }
  if (updates.attempt_index !== undefined) {
    fields.push("attempt_index = ?");
    bindings.push(updates.attempt_index);
  }
  if (updates.dataset_snapshot_id !== undefined) {
    fields.push("dataset_snapshot_id = ?");
    bindings.push(updates.dataset_snapshot_id);
  }
  if (updates.runpod_pod_id !== undefined) {
    fields.push("runpod_pod_id = ?");
    bindings.push(updates.runpod_pod_id);
  }
  if (updates.job_token !== undefined) {
    fields.push("job_token = ?");
    bindings.push(updates.job_token);
  }
  if (updates.status !== undefined) {
    fields.push("status = ?");
    bindings.push(updates.status);
  }
  if (updates.config !== undefined) {
    fields.push("config_json = ?");
    bindings.push(JSON.stringify(updates.config));
  }
  if (updates.progress !== undefined) {
    fields.push("progress_json = ?");
    bindings.push(JSON.stringify(updates.progress));
  }
  if (updates.summary !== undefined) {
    fields.push("summary_json = ?");
    bindings.push(JSON.stringify(updates.summary));
  }
  if (updates.metrics !== undefined) {
    fields.push("metrics_json = ?");
    bindings.push(JSON.stringify(updates.metrics));
  }
  if (updates.supervisor !== undefined) {
    fields.push("supervisor_json = ?");
    bindings.push(JSON.stringify(updates.supervisor));
  }
  if (updates.log_r2_prefix !== undefined) {
    fields.push("log_r2_prefix = ?");
    bindings.push(updates.log_r2_prefix);
  }
  if (updates.error_message !== undefined) {
    fields.push("error_message = ?");
    bindings.push(updates.error_message);
  }
  if (updates.last_heartbeat_at !== undefined) {
    fields.push("last_heartbeat_at = ?");
    bindings.push(updates.last_heartbeat_at);
  }
  if (updates.started_at !== undefined) {
    fields.push("started_at = ?");
    bindings.push(updates.started_at);
  }
  if (updates.completed_at !== undefined) {
    fields.push("completed_at = ?");
    bindings.push(updates.completed_at);
  }

  const effectiveUpdatedAt = updates.updated_at ?? Date.now();
  fields.push("updated_at = ?");
  bindings.push(effectiveUpdatedAt);

  if (fields.length === 0) {
    return;
  }

  if (updates.expected_updated_at !== undefined) {
    bindings.push(jobId, updates.expected_updated_at);
    const result = await db
      .prepare(`UPDATE training_jobs SET ${fields.join(", ")} WHERE job_id = ? AND updated_at = ?`)
      .bind(...bindings)
      .run();
    if ((result.meta?.changes ?? 0) === 0) {
      throw new Error(`training_job_conflict:${jobId}`);
    }
    return;
  }

  bindings.push(jobId);
  await db
    .prepare(`UPDATE training_jobs SET ${fields.join(", ")} WHERE job_id = ?`)
    .bind(...bindings)
    .run();
};

export const getTrainingCampaign = async (
  db: D1Database,
  campaignId: string
): Promise<TrainingCampaign | null> => {
  const row = await db
    .prepare("SELECT * FROM training_campaigns WHERE campaign_id = ? LIMIT 1")
    .bind(campaignId)
    .first<DbTrainingCampaignRow>();

  return row ? mapTrainingCampaign(row) : null;
};

export const listTrainingCampaigns = async (
  db: D1Database,
  filters: { voice_id?: string; status_in?: TrainingCampaignStatus[]; limit?: number } = {}
): Promise<TrainingCampaign[]> => {
  const conditions: string[] = [];
  const bindings: Array<string | number> = [];

  if (filters.voice_id) {
    conditions.push("voice_id = ?");
    bindings.push(filters.voice_id);
  }
  if (filters.status_in && filters.status_in.length > 0) {
    conditions.push(`status IN (${filters.status_in.map(() => "?").join(", ")})`);
    bindings.push(...filters.status_in);
  }

  const whereClause = conditions.length > 0 ? `WHERE ${conditions.join(" AND ")}` : "";
  const limit = Math.max(1, Math.min(filters.limit ?? 20, 100));
  const result = await db
    .prepare(`SELECT * FROM training_campaigns ${whereClause} ORDER BY created_at DESC LIMIT ?`)
    .bind(...bindings, limit)
    .all<DbTrainingCampaignRow>();

  return (result.results ?? []).map(mapTrainingCampaign);
};

export const createTrainingCampaign = async (
  db: D1Database,
  campaign: TrainingCampaign
): Promise<void> => {
  await db
    .prepare(
      `INSERT INTO training_campaigns (
        campaign_id, voice_id, dataset_name, dataset_r2_prefix, dataset_snapshot_id,
        attempt_count, parallelism, status, base_config_json, stop_rules_json,
        planner_state_json, summary_json, created_at, updated_at, completed_at
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
    )
    .bind(
      campaign.campaign_id,
      campaign.voice_id,
      campaign.dataset_name,
      campaign.dataset_r2_prefix,
      campaign.dataset_snapshot_id,
      campaign.attempt_count,
      campaign.parallelism,
      campaign.status,
      JSON.stringify(campaign.base_config),
      JSON.stringify(campaign.stop_rules),
      JSON.stringify(campaign.planner_state),
      JSON.stringify(campaign.summary),
      campaign.created_at,
      campaign.updated_at,
      campaign.completed_at
    )
    .run();
};

export const updateTrainingCampaign = async (
  db: D1Database,
  campaignId: string,
  updates: {
    dataset_name?: string | null;
    dataset_r2_prefix?: string | null;
    dataset_snapshot_id?: string | null;
    attempt_count?: number;
    parallelism?: number;
    status?: TrainingCampaignStatus;
    base_config?: TrainingConfig;
    stop_rules?: TrainingCampaignStopRules;
    planner_state?: Record<string, unknown>;
    summary?: Record<string, unknown>;
    completed_at?: number | null;
    updated_at?: number;
    expected_updated_at?: number;
  }
): Promise<void> => {
  const fields: string[] = [];
  const bindings: Array<string | number | null> = [];

  if (updates.dataset_name !== undefined) {
    fields.push("dataset_name = ?");
    bindings.push(updates.dataset_name);
  }
  if (updates.dataset_r2_prefix !== undefined) {
    fields.push("dataset_r2_prefix = ?");
    bindings.push(updates.dataset_r2_prefix);
  }
  if (updates.dataset_snapshot_id !== undefined) {
    fields.push("dataset_snapshot_id = ?");
    bindings.push(updates.dataset_snapshot_id);
  }
  if (updates.attempt_count !== undefined) {
    fields.push("attempt_count = ?");
    bindings.push(updates.attempt_count);
  }
  if (updates.parallelism !== undefined) {
    fields.push("parallelism = ?");
    bindings.push(updates.parallelism);
  }
  if (updates.status !== undefined) {
    fields.push("status = ?");
    bindings.push(updates.status);
  }
  if (updates.base_config !== undefined) {
    fields.push("base_config_json = ?");
    bindings.push(JSON.stringify(updates.base_config));
  }
  if (updates.stop_rules !== undefined) {
    fields.push("stop_rules_json = ?");
    bindings.push(JSON.stringify(updates.stop_rules));
  }
  if (updates.planner_state !== undefined) {
    fields.push("planner_state_json = ?");
    bindings.push(JSON.stringify(updates.planner_state));
  }
  if (updates.summary !== undefined) {
    fields.push("summary_json = ?");
    bindings.push(JSON.stringify(updates.summary));
  }
  if (updates.completed_at !== undefined) {
    fields.push("completed_at = ?");
    bindings.push(updates.completed_at);
  }

  fields.push("updated_at = ?");
  bindings.push(updates.updated_at ?? Date.now());

  if (updates.expected_updated_at !== undefined) {
    bindings.push(campaignId, updates.expected_updated_at);
    const result = await db
      .prepare(`UPDATE training_campaigns SET ${fields.join(", ")} WHERE campaign_id = ? AND updated_at = ?`)
      .bind(...bindings)
      .run();
    if ((result.meta?.changes ?? 0) === 0) {
      throw new Error(`training_campaign_conflict:${campaignId}`);
    }
    return;
  }

  bindings.push(campaignId);
  await db
    .prepare(`UPDATE training_campaigns SET ${fields.join(", ")} WHERE campaign_id = ?`)
    .bind(...bindings)
    .run();
};

export const createTrainingLogChunk = async (
  db: D1Database,
  chunk: {
    job_id: string;
    seq: number;
    r2_key: string;
    created_at: number;
    bytes?: number;
    lines?: number;
  }
): Promise<void> => {
  await db
    .prepare(
      `INSERT OR REPLACE INTO training_log_chunks (job_id, seq, r2_key, created_at, bytes, lines)
      VALUES (?, ?, ?, ?, ?, ?)`
    )
    .bind(
      chunk.job_id,
      chunk.seq,
      chunk.r2_key,
      chunk.created_at,
      chunk.bytes ?? null,
      chunk.lines ?? null
    )
    .run();
};

export const listTrainingLogChunks = async (
  db: D1Database,
  jobId: string,
  limit = 50,
  cursor?: number
): Promise<TrainingLogChunk[]> => {
  const pageSize = Math.max(1, Math.min(limit, 200));
  const hasCursor = typeof cursor === "number";
  const sql = hasCursor
    ? "SELECT * FROM training_log_chunks WHERE job_id = ? AND seq < ? ORDER BY seq DESC LIMIT ?"
    : "SELECT * FROM training_log_chunks WHERE job_id = ? ORDER BY seq DESC LIMIT ?";
  const stmt = hasCursor ? db.prepare(sql).bind(jobId, cursor as number, pageSize) : db.prepare(sql).bind(jobId, pageSize);
  const result = await stmt.all<DbTrainingLogChunkRow>();
  return (result.results ?? []).map(mapTrainingLogChunk);
};

export const getTrainingLogChunk = async (
  db: D1Database,
  jobId: string,
  seq: number
): Promise<TrainingLogChunk | null> => {
  const row = await db
    .prepare("SELECT * FROM training_log_chunks WHERE job_id = ? AND seq = ? LIMIT 1")
    .bind(jobId, seq)
    .first<DbTrainingLogChunkRow>();

  if (!row) {
    return null;
  }

  return mapTrainingLogChunk(row);
};

export const deleteTrainingLogChunks = async (
  db: D1Database,
  jobId: string
): Promise<void> => {
  await db.prepare("DELETE FROM training_log_chunks WHERE job_id = ?").bind(jobId).run();
};

export const getDatasetPreprocessCache = async (
  db: D1Database,
  voiceId: string,
  datasetR2Prefix: string,
  datasetSignature: string
): Promise<DatasetPreprocessCache | null> => {
  const row = await db
    .prepare(
      `SELECT * FROM dataset_preprocess_caches
       WHERE voice_id = ? AND dataset_r2_prefix = ? AND dataset_signature = ?
       LIMIT 1`
    )
    .bind(voiceId, datasetR2Prefix, datasetSignature)
    .first<DbDatasetPreprocessCacheRow>();

  return row ? mapDatasetPreprocessCache(row) : null;
};

export const getDatasetPreprocessCacheById = async (
  db: D1Database,
  cacheId: string
): Promise<DatasetPreprocessCache | null> => {
  const row = await db
    .prepare("SELECT * FROM dataset_preprocess_caches WHERE cache_id = ? LIMIT 1")
    .bind(cacheId)
    .first<DbDatasetPreprocessCacheRow>();

  return row ? mapDatasetPreprocessCache(row) : null;
};

export const upsertDatasetPreprocessCache = async (
  db: D1Database,
  cache: DatasetPreprocessCache
): Promise<void> => {
  await db
    .prepare(
      `INSERT INTO dataset_preprocess_caches (
        cache_id, voice_id, dataset_r2_prefix, dataset_signature, cache_r2_prefix,
        train_raw_r2_key, ref_audio_r2_key, reference_profile_r2_key,
        source_file_count, segments_created, segments_accepted, accepted_duration_min,
        created_at, updated_at
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      ON CONFLICT(voice_id, dataset_r2_prefix, dataset_signature) DO UPDATE SET
        cache_r2_prefix = excluded.cache_r2_prefix,
        train_raw_r2_key = excluded.train_raw_r2_key,
        ref_audio_r2_key = excluded.ref_audio_r2_key,
        reference_profile_r2_key = excluded.reference_profile_r2_key,
        source_file_count = excluded.source_file_count,
        segments_created = excluded.segments_created,
        segments_accepted = excluded.segments_accepted,
        accepted_duration_min = excluded.accepted_duration_min,
        updated_at = excluded.updated_at`
    )
    .bind(
      cache.cache_id,
      cache.voice_id,
      cache.dataset_r2_prefix,
      cache.dataset_signature,
      cache.cache_r2_prefix,
      cache.train_raw_r2_key,
      cache.ref_audio_r2_key,
      cache.reference_profile_r2_key,
      cache.source_file_count,
      cache.segments_created,
      cache.segments_accepted,
      cache.accepted_duration_min,
      cache.created_at,
      cache.updated_at
    )
    .run();
};

export const getDatasetPreprocessCacheEntry = async (
  db: D1Database,
  entryId: string
): Promise<DatasetPreprocessCacheEntry | null> => {
  const row = await db
    .prepare("SELECT * FROM dataset_preprocess_cache_entries WHERE entry_id = ? LIMIT 1")
    .bind(entryId)
    .first<DbDatasetPreprocessCacheEntryRow>();

  return row ? mapDatasetPreprocessCacheEntry(row) : null;
};

export const listDatasetPreprocessCacheEntries = async (
  db: D1Database,
  cacheId: string
): Promise<DatasetPreprocessCacheEntry[]> => {
  const result = await db
    .prepare(
      `SELECT * FROM dataset_preprocess_cache_entries
       WHERE cache_id = ?
       ORDER BY seq ASC`
    )
    .bind(cacheId)
    .all<DbDatasetPreprocessCacheEntryRow>();

  return (result.results ?? []).map(mapDatasetPreprocessCacheEntry);
};

export const replaceDatasetPreprocessCacheEntries = async (
  db: D1Database,
  cacheId: string,
  entries: DatasetPreprocessCacheEntry[]
): Promise<void> => {
  await db
    .prepare("DELETE FROM dataset_preprocess_cache_entries WHERE cache_id = ?")
    .bind(cacheId)
    .run();

  if (entries.length === 0) {
    return;
  }

  const chunkSize = 100;
  for (let index = 0; index < entries.length; index += chunkSize) {
    const chunk = entries.slice(index, index + chunkSize);
    await db.batch(
      chunk.map((entry) =>
        db
          .prepare(
            `INSERT INTO dataset_preprocess_cache_entries (
              entry_id, cache_id, seq, audio_path, audio_r2_key, text, included, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`
          )
          .bind(
            entry.entry_id,
            cacheId,
            entry.seq,
            entry.audio_path,
            entry.audio_r2_key,
            entry.text,
            entry.included ? 1 : 0,
            entry.created_at,
            entry.updated_at
          )
      )
    );
  }
};

export const updateDatasetPreprocessCacheEntry = async (
  db: D1Database,
  entryId: string,
  updates: {
    text?: string;
    included?: boolean;
    updated_at?: number;
  }
): Promise<void> => {
  const fields: string[] = [];
  const bindings: Array<string | number> = [];

  if (updates.text !== undefined) {
    fields.push("text = ?");
    bindings.push(updates.text);
  }
  if (updates.included !== undefined) {
    fields.push("included = ?");
    bindings.push(updates.included ? 1 : 0);
  }

  fields.push("updated_at = ?");
  bindings.push(updates.updated_at ?? Date.now());

  bindings.push(entryId);
  await db
    .prepare(`UPDATE dataset_preprocess_cache_entries SET ${fields.join(", ")} WHERE entry_id = ?`)
    .bind(...bindings)
    .run();
};

export const getDatasetSnapshot = async (
  db: D1Database,
  voiceId: string,
  datasetR2Prefix: string,
  datasetSignature: string
): Promise<DatasetSnapshot | null> => {
  const row = await db
    .prepare(
      `SELECT * FROM dataset_snapshots
       WHERE voice_id = ? AND dataset_r2_prefix = ? AND dataset_signature = ?
       LIMIT 1`
    )
    .bind(voiceId, datasetR2Prefix, datasetSignature)
    .first<DbDatasetSnapshotRow>();

  return row ? mapDatasetSnapshot(row) : null;
};

export const getDatasetSnapshotById = async (
  db: D1Database,
  snapshotId: string
): Promise<DatasetSnapshot | null> => {
  const row = await db
    .prepare("SELECT * FROM dataset_snapshots WHERE snapshot_id = ? LIMIT 1")
    .bind(snapshotId)
    .first<DbDatasetSnapshotRow>();

  return row ? mapDatasetSnapshot(row) : null;
};

export const listDatasetSnapshots = async (
  db: D1Database,
  filters: { voice_id?: string; limit?: number } = {}
): Promise<DatasetSnapshot[]> => {
  const conditions: string[] = [];
  const bindings: Array<string | number> = [];

  if (filters.voice_id) {
    conditions.push("voice_id = ?");
    bindings.push(filters.voice_id);
  }

  const whereClause = conditions.length > 0 ? `WHERE ${conditions.join(" AND ")}` : "";
  const limit = Math.max(1, Math.min(filters.limit ?? 20, 100));
  const result = await db
    .prepare(`SELECT * FROM dataset_snapshots ${whereClause} ORDER BY updated_at DESC, created_at DESC LIMIT ?`)
    .bind(...bindings, limit)
    .all<DbDatasetSnapshotRow>();

  return (result.results ?? []).map(mapDatasetSnapshot);
};

export const upsertDatasetSnapshot = async (
  db: D1Database,
  snapshot: DatasetSnapshot
): Promise<void> => {
  await db
    .prepare(
      `INSERT INTO dataset_snapshots (
        snapshot_id, voice_id, dataset_name, dataset_r2_prefix, dataset_signature, status,
        source_cache_id, cache_r2_prefix, train_raw_r2_key, ref_audio_r2_key, reference_profile_r2_key,
        reference_text, source_file_count, segments_created, segments_accepted, accepted_duration_min,
        created_from_job_id, created_at, updated_at
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      ON CONFLICT(voice_id, dataset_r2_prefix, dataset_signature) DO UPDATE SET
        dataset_name = excluded.dataset_name,
        status = excluded.status,
        source_cache_id = excluded.source_cache_id,
        cache_r2_prefix = excluded.cache_r2_prefix,
        train_raw_r2_key = excluded.train_raw_r2_key,
        ref_audio_r2_key = excluded.ref_audio_r2_key,
        reference_profile_r2_key = excluded.reference_profile_r2_key,
        reference_text = excluded.reference_text,
        source_file_count = excluded.source_file_count,
        segments_created = excluded.segments_created,
        segments_accepted = excluded.segments_accepted,
        accepted_duration_min = excluded.accepted_duration_min,
        created_from_job_id = excluded.created_from_job_id,
        updated_at = excluded.updated_at`
    )
    .bind(
      snapshot.snapshot_id,
      snapshot.voice_id,
      snapshot.dataset_name ?? null,
      snapshot.dataset_r2_prefix,
      snapshot.dataset_signature,
      snapshot.status,
      snapshot.source_cache_id ?? null,
      snapshot.cache_r2_prefix ?? null,
      snapshot.train_raw_r2_key ?? null,
      snapshot.ref_audio_r2_key ?? null,
      snapshot.reference_profile_r2_key ?? null,
      snapshot.reference_text ?? null,
      snapshot.source_file_count ?? null,
      snapshot.segments_created ?? null,
      snapshot.segments_accepted ?? null,
      snapshot.accepted_duration_min ?? null,
      snapshot.created_from_job_id ?? null,
      snapshot.created_at,
      snapshot.updated_at
    )
    .run();
};

export const getTrainingRound = async (
  db: D1Database,
  roundId: string
): Promise<TrainingRound | null> => {
  const row = await db
    .prepare("SELECT * FROM training_rounds WHERE round_id = ? LIMIT 1")
    .bind(roundId)
    .first<DbTrainingRoundRow>();

  return row ? mapTrainingRound(row) : null;
};

export const listTrainingRounds = async (
  db: D1Database,
  filters: { voice_id?: string; limit?: number } = {}
): Promise<TrainingRound[]> => {
  const conditions: string[] = [];
  const bindings: Array<string | number> = [];

  if (filters.voice_id) {
    conditions.push("voice_id = ?");
    bindings.push(filters.voice_id);
  }

  const whereClause = conditions.length > 0 ? `WHERE ${conditions.join(" AND ")}` : "";
  const limit = Math.max(1, Math.min(filters.limit ?? 20, 100));
  const result = await db
    .prepare(`SELECT * FROM training_rounds ${whereClause} ORDER BY round_index DESC, created_at DESC LIMIT ?`)
    .bind(...bindings, limit)
    .all<DbTrainingRoundRow>();

  return (result.results ?? []).map(mapTrainingRound);
};

export const getLatestTrainingRoundForVoice = async (
  db: D1Database,
  voiceId: string
): Promise<TrainingRound | null> => {
  const row = await db
    .prepare(
      `SELECT * FROM training_rounds
       WHERE voice_id = ?
       ORDER BY round_index DESC, created_at DESC
       LIMIT 1`
    )
    .bind(voiceId)
    .first<DbTrainingRoundRow>();

  return row ? mapTrainingRound(row) : null;
};

export const createTrainingRound = async (
  db: D1Database,
  round: TrainingRound
): Promise<void> => {
  await db
    .prepare(
      `INSERT INTO training_rounds (
        round_id, voice_id, dataset_snapshot_id, round_index, status,
        production_checkpoint_r2_prefix, production_run_name, production_epoch,
        production_preset, production_score, production_job_id,
        champion_checkpoint_r2_prefix, champion_run_name, champion_epoch,
        champion_preset, champion_score, champion_job_id,
        selected_checkpoint_r2_prefix, selected_run_name, selected_epoch,
        selected_preset, selected_score, selected_job_id, adoption_mode,
        candidate_checkpoint_r2_prefix, candidate_run_name, candidate_epoch,
        candidate_score, candidate_job_id, summary_json, created_at, updated_at,
        started_at, completed_at
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
    )
    .bind(
      round.round_id,
      round.voice_id,
      round.dataset_snapshot_id ?? null,
      round.round_index,
      round.status,
      round.production_checkpoint_r2_prefix ?? null,
      round.production_run_name ?? null,
      round.production_epoch ?? null,
      round.production_preset ?? null,
      round.production_score ?? null,
      round.production_job_id ?? null,
      round.champion_checkpoint_r2_prefix ?? null,
      round.champion_run_name ?? null,
      round.champion_epoch ?? null,
      round.champion_preset ?? null,
      round.champion_score ?? null,
      round.champion_job_id ?? null,
      round.selected_checkpoint_r2_prefix ?? null,
      round.selected_run_name ?? null,
      round.selected_epoch ?? null,
      round.selected_preset ?? null,
      round.selected_score ?? null,
      round.selected_job_id ?? null,
      round.adoption_mode ?? null,
      round.candidate_checkpoint_r2_prefix ?? null,
      round.candidate_run_name ?? null,
      round.candidate_epoch ?? null,
      round.candidate_score ?? null,
      round.candidate_job_id ?? null,
      JSON.stringify(round.summary),
      round.created_at,
      round.updated_at,
      round.started_at ?? null,
      round.completed_at ?? null
    )
    .run();
};

export const updateTrainingRound = async (
  db: D1Database,
  roundId: string,
  updates: {
    dataset_snapshot_id?: string | null;
    status?: string;
    production_checkpoint_r2_prefix?: string | null;
    production_run_name?: string | null;
    production_epoch?: number | null;
    production_preset?: string | null;
    production_score?: number | null;
    production_job_id?: string | null;
    champion_checkpoint_r2_prefix?: string | null;
    champion_run_name?: string | null;
    champion_epoch?: number | null;
    champion_preset?: string | null;
    champion_score?: number | null;
    champion_job_id?: string | null;
    selected_checkpoint_r2_prefix?: string | null;
    selected_run_name?: string | null;
    selected_epoch?: number | null;
    selected_preset?: string | null;
    selected_score?: number | null;
    selected_job_id?: string | null;
    adoption_mode?: string | null;
    candidate_checkpoint_r2_prefix?: string | null;
    candidate_run_name?: string | null;
    candidate_epoch?: number | null;
    candidate_score?: number | null;
    candidate_job_id?: string | null;
    summary?: Record<string, unknown>;
    started_at?: number | null;
    completed_at?: number | null;
    updated_at?: number;
  }
): Promise<void> => {
  const fields: string[] = [];
  const bindings: Array<string | number | null> = [];

  if (updates.dataset_snapshot_id !== undefined) {
    fields.push("dataset_snapshot_id = ?");
    bindings.push(updates.dataset_snapshot_id);
  }
  if (updates.status !== undefined) {
    fields.push("status = ?");
    bindings.push(updates.status);
  }
  if (updates.production_checkpoint_r2_prefix !== undefined) {
    fields.push("production_checkpoint_r2_prefix = ?");
    bindings.push(updates.production_checkpoint_r2_prefix);
  }
  if (updates.production_run_name !== undefined) {
    fields.push("production_run_name = ?");
    bindings.push(updates.production_run_name);
  }
  if (updates.production_epoch !== undefined) {
    fields.push("production_epoch = ?");
    bindings.push(updates.production_epoch);
  }
  if (updates.production_preset !== undefined) {
    fields.push("production_preset = ?");
    bindings.push(updates.production_preset);
  }
  if (updates.production_score !== undefined) {
    fields.push("production_score = ?");
    bindings.push(updates.production_score);
  }
  if (updates.production_job_id !== undefined) {
    fields.push("production_job_id = ?");
    bindings.push(updates.production_job_id);
  }
  if (updates.champion_checkpoint_r2_prefix !== undefined) {
    fields.push("champion_checkpoint_r2_prefix = ?");
    bindings.push(updates.champion_checkpoint_r2_prefix);
  }
  if (updates.champion_run_name !== undefined) {
    fields.push("champion_run_name = ?");
    bindings.push(updates.champion_run_name);
  }
  if (updates.champion_epoch !== undefined) {
    fields.push("champion_epoch = ?");
    bindings.push(updates.champion_epoch);
  }
  if (updates.champion_preset !== undefined) {
    fields.push("champion_preset = ?");
    bindings.push(updates.champion_preset);
  }
  if (updates.champion_score !== undefined) {
    fields.push("champion_score = ?");
    bindings.push(updates.champion_score);
  }
  if (updates.champion_job_id !== undefined) {
    fields.push("champion_job_id = ?");
    bindings.push(updates.champion_job_id);
  }
  if (updates.selected_checkpoint_r2_prefix !== undefined) {
    fields.push("selected_checkpoint_r2_prefix = ?");
    bindings.push(updates.selected_checkpoint_r2_prefix);
  }
  if (updates.selected_run_name !== undefined) {
    fields.push("selected_run_name = ?");
    bindings.push(updates.selected_run_name);
  }
  if (updates.selected_epoch !== undefined) {
    fields.push("selected_epoch = ?");
    bindings.push(updates.selected_epoch);
  }
  if (updates.selected_preset !== undefined) {
    fields.push("selected_preset = ?");
    bindings.push(updates.selected_preset);
  }
  if (updates.selected_score !== undefined) {
    fields.push("selected_score = ?");
    bindings.push(updates.selected_score);
  }
  if (updates.selected_job_id !== undefined) {
    fields.push("selected_job_id = ?");
    bindings.push(updates.selected_job_id);
  }
  if (updates.adoption_mode !== undefined) {
    fields.push("adoption_mode = ?");
    bindings.push(updates.adoption_mode);
  }
  if (updates.candidate_checkpoint_r2_prefix !== undefined) {
    fields.push("candidate_checkpoint_r2_prefix = ?");
    bindings.push(updates.candidate_checkpoint_r2_prefix);
  }
  if (updates.candidate_run_name !== undefined) {
    fields.push("candidate_run_name = ?");
    bindings.push(updates.candidate_run_name);
  }
  if (updates.candidate_epoch !== undefined) {
    fields.push("candidate_epoch = ?");
    bindings.push(updates.candidate_epoch);
  }
  if (updates.candidate_score !== undefined) {
    fields.push("candidate_score = ?");
    bindings.push(updates.candidate_score);
  }
  if (updates.candidate_job_id !== undefined) {
    fields.push("candidate_job_id = ?");
    bindings.push(updates.candidate_job_id);
  }
  if (updates.summary !== undefined) {
    fields.push("summary_json = ?");
    bindings.push(JSON.stringify(updates.summary));
  }
  if (updates.started_at !== undefined) {
    fields.push("started_at = ?");
    bindings.push(updates.started_at);
  }
  if (updates.completed_at !== undefined) {
    fields.push("completed_at = ?");
    bindings.push(updates.completed_at);
  }

  fields.push("updated_at = ?");
  bindings.push(updates.updated_at ?? Date.now());

  bindings.push(roundId);
  await db
    .prepare(`UPDATE training_rounds SET ${fields.join(", ")} WHERE round_id = ?`)
    .bind(...bindings)
    .run();
};

const mapTrainingCheckoutLedgerEntry = (
  row: DbTrainingCheckoutLedgerRow
): TrainingCheckoutLedgerEntry => ({
  entry_id: row.entry_id,
  round_id: row.round_id,
  job_id: row.job_id,
  voice_id: row.voice_id,
  checkpoint_r2_prefix: row.checkpoint_r2_prefix,
  run_name: row.run_name,
  epoch: row.epoch,
  preset: row.preset,
  score: row.score,
  ok: row.ok === null ? null : row.ok === 1,
  passed_samples: row.passed_samples,
  total_samples: row.total_samples,
  message: row.message,
  role: row.role,
  source: row.source,
  adoption_mode: row.adoption_mode,
  created_at: row.created_at,
  updated_at: row.updated_at,
});

export const listTrainingCheckoutLedger = async (
  db: D1Database,
  filters: { job_id?: string; round_id?: string; voice_id?: string; limit?: number } = {}
): Promise<TrainingCheckoutLedgerEntry[]> => {
  const conditions: string[] = [];
  const bindings: Array<string | number> = [];

  if (filters.job_id) {
    conditions.push("job_id = ?");
    bindings.push(filters.job_id);
  }
  if (filters.round_id) {
    conditions.push("round_id = ?");
    bindings.push(filters.round_id);
  }
  if (filters.voice_id) {
    conditions.push("voice_id = ?");
    bindings.push(filters.voice_id);
  }

  const whereClause = conditions.length > 0 ? `WHERE ${conditions.join(" AND ")}` : "";
  const limit = Math.max(1, Math.min(filters.limit ?? 200, 1000));
  const result = await db
    .prepare(
      `SELECT * FROM training_checkout_ledger ${whereClause}
       ORDER BY created_at DESC, updated_at DESC
       LIMIT ?`
    )
    .bind(...bindings, limit)
    .all<DbTrainingCheckoutLedgerRow>();

  return (result.results ?? []).map(mapTrainingCheckoutLedgerEntry);
};

export const replaceTrainingCheckoutLedgerForJob = async (
  db: D1Database,
  jobId: string,
  entries: TrainingCheckoutLedgerEntry[]
): Promise<void> => {
  await db.prepare("DELETE FROM training_checkout_ledger WHERE job_id = ?").bind(jobId).run();

  for (const entry of entries) {
    await db
      .prepare(
        `INSERT INTO training_checkout_ledger (
          entry_id, round_id, job_id, voice_id, checkpoint_r2_prefix, run_name, epoch, preset,
          score, ok, passed_samples, total_samples, message, role, source, adoption_mode,
          created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
      )
      .bind(
        entry.entry_id,
        entry.round_id,
        entry.job_id,
        entry.voice_id,
        entry.checkpoint_r2_prefix,
        entry.run_name,
        entry.epoch,
        entry.preset,
        entry.score,
        entry.ok === null ? null : entry.ok ? 1 : 0,
        entry.passed_samples,
        entry.total_samples,
        entry.message,
        entry.role,
        entry.source,
        entry.adoption_mode,
        entry.created_at,
        entry.updated_at
      )
      .run();
  }
};
