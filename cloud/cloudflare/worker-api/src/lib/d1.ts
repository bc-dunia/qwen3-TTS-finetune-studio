import type {
  Generation,
  TrainingConfig,
  TrainingJob,
  TrainingProgress,
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
  runpod_pod_id: string | null;
  job_token: string | null;
  status: string;
  config_json: string;
  progress_json: string;
  summary_json: string | null;
  metrics_json: string | null;
  dataset_r2_prefix: string;
  log_r2_prefix: string | null;
  error_message: string | null;
  last_heartbeat_at: number | null;
  started_at: number | null;
  completed_at: number | null;
  created_at: number;
  updated_at: number;
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
  runpod_pod_id: row.runpod_pod_id,
  job_token: row.job_token,
  status: row.status,
  config: parseJson<TrainingConfig>(row.config_json, {}),
  progress: parseJson<TrainingProgress>(row.progress_json, {}),
  summary: parseJson<Record<string, unknown>>(row.summary_json, {}),
  metrics: parseJson<Record<string, unknown>>(row.metrics_json, {}),
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
        checkpoint_r2_prefix, run_name, epoch, sample_audio_r2_key, ref_audio_r2_key, labels_json, settings_json,
        created_at, updated_at
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
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
  filters: { voice_id?: string; limit?: number } = {}
): Promise<TrainingJob[]> => {
  const conditions: string[] = [];
  const bindings: Array<string | number> = [];

  if (filters.voice_id) {
    conditions.push("voice_id = ?");
    bindings.push(filters.voice_id);
  }

  const whereClause = conditions.length > 0 ? `WHERE ${conditions.join(" AND ")}` : "";
  const limit = Math.max(1, Math.min(filters.limit ?? 20, 100));
  const sql = `SELECT * FROM training_jobs ${whereClause} ORDER BY created_at DESC LIMIT ?`;

  bindings.push(limit);
  const result = await db.prepare(sql).bind(...bindings).all<DbTrainingRow>();
  return (result.results ?? []).map(mapTrainingJob);
};

export const createTrainingJob = async (db: D1Database, job: TrainingJob): Promise<void> => {
  await db
    .prepare(
      `INSERT INTO training_jobs (
        job_id, voice_id, runpod_pod_id, job_token, status, config_json, progress_json,
        summary_json, metrics_json, dataset_r2_prefix, log_r2_prefix, error_message,
        last_heartbeat_at, started_at, completed_at, created_at, updated_at
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
    )
    .bind(
      job.job_id,
      job.voice_id,
      job.runpod_pod_id,
      job.job_token ?? null,
      job.status,
      JSON.stringify(job.config),
      JSON.stringify(job.progress),
      JSON.stringify(job.summary),
      JSON.stringify(job.metrics),
      job.dataset_r2_prefix,
      job.log_r2_prefix,
      job.error_message,
      job.last_heartbeat_at,
      job.started_at,
      job.completed_at,
      job.created_at,
      job.updated_at
    )
    .run();
};

export const updateTrainingJob = async (
  db: D1Database,
  jobId: string,
  updates: {
    runpod_pod_id?: string | null;
    job_token?: string | null;
    status?: string;
    config?: TrainingConfig;
    progress?: TrainingProgress;
    summary?: Record<string, unknown>;
    metrics?: Record<string, unknown>;
    log_r2_prefix?: string | null;
    error_message?: string | null;
    last_heartbeat_at?: number | null;
    started_at?: number | null;
    completed_at?: number | null;
    updated_at?: number;
  }
): Promise<void> => {
  const fields: string[] = [];
  const bindings: Array<string | number | null> = [];

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

  bindings.push(jobId);
  await db
    .prepare(`UPDATE training_jobs SET ${fields.join(", ")} WHERE job_id = ?`)
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
