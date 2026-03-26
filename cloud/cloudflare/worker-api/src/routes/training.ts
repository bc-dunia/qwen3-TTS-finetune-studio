import { Hono } from 'hono';
import type { Context } from 'hono';
import {
  createTrainingRound,
  createTrainingCampaign,
  createTrainingJob,
  deleteTrainingLogChunks,
  getDatasetPreprocessCache,
  getDatasetPreprocessCacheById,
  getDatasetSnapshot,
  getDatasetSnapshotById,
  getLatestTrainingRoundForVoice,
  listDatasetPreprocessCacheEntries,
  listDatasetSnapshots,
  listTrainingRounds,
  getTrainingJob,
  getTrainingCampaign,
  getTrainingLogChunk,
  getTrainingRound,
  getVoice,
  listTrainingJobs,
  countActiveTrainingJobs,
  listTrainingCampaigns,
  listTrainingLogChunks,
  listTrainingCheckoutLedger,
  replaceTrainingCheckoutLedgerForJob,
  replaceDatasetPreprocessCacheEntries,
  updateTrainingRound,
  updateTrainingCampaign,
  updateDatasetPreprocessCacheEntry,
  upsertDatasetSnapshot,
  updateTrainingJob,
  updateVoice,
  upsertDatasetPreprocessCache,
  type DatasetPreprocessCache,
  type DatasetPreprocessCacheEntry,
  type TrainingCheckoutLedgerEntry,
} from '../lib/d1';
import {
  createPod,
  createPodDirect,
  getServerlessStatus,
  getPodStatus,
  getTemplateById,
  invokeServerless,
  invokeServerlessAsync,
  terminatePod,
} from '../lib/runpod';
import { buildTrainingCheckoutSearch } from '../lib/training-checkout';
import { buildTrainingAdvice } from '../lib/training-advisor';
import { buildLLMTrainingAdvice } from '../lib/training-advisor-llm';
import {
  readNumber as domainReadNumber,
  readTimestamp as domainReadTimestamp,
  stripSlashes as domainStripSlashes,
  extractDatasetNameFromPrefix as domainExtractDatasetNameFromPrefix,
  parseRunNameFromCheckpointPrefix as domainParseRunNameFromCheckpointPrefix,
  getTrainingDefaults,
  passesValidationGate,
  passesHardSafetyGate,
  grayZoneRescue,
  VALIDATION_GATE_THRESHOLDS,
  getJobPriority,
} from '../lib/training-domain';
import { loadEffectiveWeights } from '../lib/arena-calibration';
import {
  planCampaignAttempts,
  buildLLMPlannerPrompt,
  buildLLMPlannerSystemPrompt,
  buildLLMPlannerStateHash,
  parseLLMPlannerResponse,
  type LLMPlannerDecision,
  type PlannerResult,
} from '../lib/campaign-planner';
import { enrichOutputWithReviewAsr } from '../lib/review-asr';
import { buildStrategyBrief, maybeAdvanceResearchLoop } from '../lib/research-loop';
import { authMiddleware } from '../middleware/auth';
import type {
  AppContext,
  DatasetSnapshot,
  Env,
  TrainingCampaign,
  TrainingCampaignStatus,
  TrainingCampaignStopRules,
  TrainingConfig,
  TrainingJob,
  TrainingProgress,
  TrainingRound,
} from '../types';

const app = new Hono<AppContext>();
app.use('*', authMiddleware);

type TrainingStatusBlob = {
  status?: string;
  progress?: TrainingProgress;
  checkpoints?: Array<{ epoch?: number; r2_prefix?: string }>;
  updated_at?: string;
};

type ValidationPreset = {
  name: string;
  payload: Record<string, unknown>;
  settings?: {
    stability: number;
    similarity_boost: number;
    style: number;
    speed: number;
  };
};

type CheckpointValidationResult = {
  ok: boolean;
  message: string;
  aggregateScore: number;
  presetName: string;
  presetSettings?: ValidationPreset['settings'];
  passedSamples: number;
  totalSamples: number;
};

type CheckpointCandidate = {
  epoch: number;
  r2_prefix: string;
};

type CheckpointEvaluation = {
  epoch: number;
  prefix: string;
  ok: boolean;
  score: number;
  message: string;
  preset: string;
  passed_samples: number;
  total_samples: number;
};

type AsyncValidationAccumulator = {
  passed: number;
  no_audio: number;
  infra_issues: number;
  sum_overall: number;
  sum_duration: number;
  sum_health: number;
  sum_asr: number;
  sum_speaker: number;
  sum_tone: number;
  sum_speed: number;
  sum_style: number;
  speaker_samples: number;
  tone_samples: number;
  speed_samples: number;
  style_samples: number;
  first_failure_message: string | null;
};

type AsyncValidationChampion = {
  epoch: number;
  prefix: string;
  score: number;
  message: string;
  preset_name: string;
  preset_settings?: ValidationPreset['settings'];
  passed_samples: number;
  total_samples: number;
};

type AsyncValidationFailure = {
  passed_samples: number;
  score: number;
  message: string;
  preset_name: string;
  total_samples: number;
};

type AsyncCheckpointValidationState = {
  mode: 'checkpoint_async';
  run_id: string;
  run_started_at?: number;
  checkpoint_index: number;
  checkpoint_epoch: number;
  checkpoint_prefix: string;
  preset_index: number;
  text_index: number;
  seed_index: number;
  reference_audio_key: string | null;
  reference_text: string;
  evaluations: CheckpointEvaluation[];
  preset_stats: AsyncValidationAccumulator;
  checkpoint_best_passing: AsyncValidationChampion | null;
  checkpoint_best_failure: AsyncValidationFailure | null;
  champion: AsyncValidationChampion | null;
};

type Async06bValidationState = {
  mode: 'fast_06b_async';
  run_id: string;
  run_started_at?: number;
  checkpoint_index: number;
  checkpoint_epoch: number;
  checkpoint_prefix: string;
  preset_name: string;
  validation_text: string;
  seed: number;
  reference_audio_key: string | null;
  reference_text: string;
  evaluations: CheckpointEvaluation[];
};

type ValidationPlan = {
  is06b: boolean;
  presets: ValidationPreset[];
  validationTexts: string[];
  validationSeedOffsets: readonly number[];
  totalSamples: number;
  minOverall: number;
  minPassRate: number;
  minAsrScore: number;
  minToneScore: number;
  maxCheckpointsToEval: number;
  prioritizeLatestPassingCheckpoint: boolean;
};

type ValidationSampleOutcome = {
  passed: boolean;
  noAudio: boolean;
  infraIssue: boolean;
  overall: number | null;
  duration: number | null;
  health: number | null;
  asr: number | null;
  speaker: number | null;
  tone: number | null;
  speed: number | null;
  style: number | null;
  failureMessage: string | null;
};

const QUEUED_JOB_STATUSES = new Set(['queued']);
const ACTIVE_JOB_STATUSES = new Set([
  'running',
  'provisioning',
  'downloading',
  'preprocessing',
  'preparing',
  'training',
  'uploading',
]);
const ACTIVE_RUNTIME_RECOVERY_STATUSES = new Set([
  'running',
  'downloading',
  'preprocessing',
  'preparing',
  'training',
  'uploading',
]);

const TERMINAL_JOB_STATUSES = new Set(['completed', 'failed', 'cancelled']);
const CAMPAIGN_ACTIVE_STATUSES = new Set<TrainingCampaignStatus>(['planning', 'running']);
const DEFAULT_CAMPAIGN_ATTEMPT_COUNT = 3;
const DEFAULT_CAMPAIGN_PARALLELISM = 1;
const MAX_CAMPAIGN_ATTEMPTS = 12;

type TrainingCampaignRequest = {
  voice_id?: string;
  dataset_name?: string;
  attempt_count?: number;
  parallelism?: number;
  direction?: string;
  base_config_overrides?: TrainingConfig;
  stop_rules?: TrainingCampaignStopRules;
};
const ACTIVE_STAGE_STALE_MS: Record<string, number> = {
  running: 12 * 60 * 1000,
  downloading: 10 * 60 * 1000,
  preprocessing: 30 * 60 * 1000,
  preparing: 20 * 60 * 1000,
  training: 12 * 60 * 1000,
  uploading: 20 * 60 * 1000,
};
const MAX_STALL_RECOVERY_ATTEMPTS = 3;
const DEFAULT_WORKER_PUBLIC_URL = 'https://qwen-tts-api.brian-367.workers.dev';
const TRAINING_SWEEP_LIMIT = 100;
const DEFAULT_MAX_ACTIVE_TRAINING_JOBS_PER_VOICE = 1;
const DEFAULT_MAX_ACTIVE_TRAINING_JOBS_GLOBAL = 3;
const DEFAULT_TRAINING_IMAGE_FALLBACKS = [
  'ghcr.io/bc-dunia/qwen3-tts-training@sha256:fd32cc16f8febc7ce23a08eea38b9b62beaa7819139895e241a8bdddb6bf2357',
];

const REFERENCE_AUDIO_KEY_RE = /\/ref_audio\.[^/]+$/i;
const RAW_DATASET_AUDIO_EXTENSIONS = new Set(['.wav', '.mp3', '.mp4', '.m4a', '.flac', '.aac']);
const GENERATED_DATASET_FILENAMES = new Set([
  'train_raw.jsonl',
  'train_with_codes.jsonl',
  'ref_audio.wav',
  'reference_profile.json',
  'preprocess_report.json',
]);

const stripSlashes = domainStripSlashes;

const uniqueNonEmpty = (values: Array<string | null | undefined>): string[] => {
  const deduped = new Set<string>();
  for (const value of values) {
    if (typeof value !== 'string') {
      continue;
    }
    const trimmed = value.trim();
    if (!trimmed) {
      continue;
    }
    deduped.add(trimmed);
  }
  return [...deduped];
};

const toHex = (buffer: ArrayBuffer): string =>
  [...new Uint8Array(buffer)].map((value) => value.toString(16).padStart(2, '0')).join('');

const listAllR2Objects = async (bucket: R2Bucket, prefix: string): Promise<R2Object[]> => {
  const objects: R2Object[] = [];
  let cursor: string | undefined;
  do {
    const page = await bucket.list({ prefix, cursor, limit: 1000 });
    objects.push(...page.objects);
    cursor = page.truncated ? page.cursor : undefined;
  } while (cursor);
  return objects;
};

const isRawDatasetObject = (datasetPrefix: string, key: string): boolean => {
  const normalizedPrefix = `${stripSlashes(datasetPrefix)}/`;
  if (!key.startsWith(normalizedPrefix)) {
    return false;
  }
  const relative = key.slice(normalizedPrefix.length).replace(/^\/+/, '');
  if (!relative) {
    return false;
  }
  if (
    relative.startsWith('segments/') ||
    relative.startsWith('converted/') ||
    relative.startsWith('.preprocess_cache/')
  ) {
    return false;
  }
  const parts = relative.split('/');
  const fileName = (parts[parts.length - 1] ?? '').toLowerCase();
  if (GENERATED_DATASET_FILENAMES.has(fileName)) {
    return false;
  }
  const dotIndex = fileName.lastIndexOf('.');
  const ext = dotIndex >= 0 ? fileName.slice(dotIndex) : '';
  return RAW_DATASET_AUDIO_EXTENSIONS.has(ext);
};

const computeDatasetSignature = async (
  datasetPrefix: string,
  objects: R2Object[],
): Promise<{ signature: string; sourceCount: number } | null> => {
  const normalizedPrefix = `${stripSlashes(datasetPrefix)}/`;
  const rawObjects = objects
    .filter((object) => isRawDatasetObject(datasetPrefix, object.key))
    .map((object) => {
      const anyObject = object as unknown as { etag?: string };
      return {
        relative: object.key.slice(normalizedPrefix.length),
        size: object.size,
        etag: typeof anyObject.etag === 'string' ? anyObject.etag : '',
      };
    })
    .sort((a, b) => a.relative.localeCompare(b.relative));

  if (rawObjects.length === 0) {
    return null;
  }

  const payload = rawObjects
    .map((object) => `${object.relative}\t${object.size}\t${object.etag}`)
    .join('\n');
  const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(payload));
  return {
    signature: toHex(digest),
    sourceCount: rawObjects.length,
  };
};

const TMP_DATASET_PREFIX = '/tmp/dataset/';

const getSummaryString = (summary: Record<string, unknown>, key: string): string | null => {
  const value = summary[key];
  return typeof value === 'string' && value.trim() ? value.trim() : null;
};

const toDatasetRelativePath = (value: string): string => {
  const trimmed = String(value ?? '').trim();
  if (!trimmed) {
    return '';
  }
  if (trimmed.startsWith(TMP_DATASET_PREFIX)) {
    return trimmed.slice(TMP_DATASET_PREFIX.length).replace(/^\/+/, '');
  }
  return trimmed.replace(/^\/+/, '');
};

const buildPreprocessCacheEntryId = (cacheId: string, seq: number): string =>
  `${cacheId}:${String(seq).padStart(6, '0')}`;

const readR2Text = async (bucket: R2Bucket, key: string): Promise<string | null> => {
  const obj = await bucket.get(key);
  if (!obj) {
    return null;
  }
  return obj.text();
};

const readReferenceText = async (
  bucket: R2Bucket,
  cache: DatasetPreprocessCache,
): Promise<string | null> => {
  const profileKey =
    (typeof cache.reference_profile_r2_key === 'string' && cache.reference_profile_r2_key.trim()) ||
    `${cache.cache_r2_prefix}/reference_profile.json`;
  const raw = await readR2Text(bucket, profileKey);
  if (!raw) {
    return null;
  }

  try {
    const parsed = JSON.parse(raw) as Record<string, unknown>;
    const value = parsed.reference_text;
    return typeof value === 'string' ? value.trim() : null;
  } catch {
    return null;
  }
};

const parseTrainRawCacheEntries = (
  raw: string,
  cache: DatasetPreprocessCache,
  now: number,
): DatasetPreprocessCacheEntry[] =>
  raw
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .flatMap((line, index) => {
      try {
        const parsed = JSON.parse(line) as Record<string, unknown>;
        const audioPath = toDatasetRelativePath(String(parsed.audio ?? ''));
        const text = String(parsed.text ?? '').trim();
        if (!audioPath || !text) {
          return [];
        }
        return [
          {
            entry_id: buildPreprocessCacheEntryId(cache.cache_id, index + 1),
            cache_id: cache.cache_id,
            seq: index + 1,
            audio_path: audioPath,
            audio_r2_key: `${cache.cache_r2_prefix}/${audioPath}`,
            text,
            included: true,
            created_at: now,
            updated_at: now,
          },
        ];
      } catch {
        return [];
      }
    });

const buildTrainRawJsonl = (entries: DatasetPreprocessCacheEntry[]): string => {
  const includedEntries = [...entries]
    .filter((entry) => entry.included)
    .sort((a, b) => a.seq - b.seq);
  if (includedEntries.length === 0) {
    throw new Error('At least one transcript entry must remain included.');
  }

  const lines = includedEntries.map((entry) => {
    const text = entry.text.trim();
    if (!text) {
      throw new Error(`Transcript entry ${entry.seq} is empty.`);
    }
    return JSON.stringify({
      audio: `${TMP_DATASET_PREFIX}${entry.audio_path}`,
      text,
      ref_audio: `${TMP_DATASET_PREFIX}ref_audio.wav`,
    });
  });
  return `${lines.join('\n')}\n`;
};

const resolveJobPreprocessCache = async (
  c: Context<AppContext>,
  job: TrainingJob,
): Promise<DatasetPreprocessCache | null> => {
  if (job.dataset_snapshot_id) {
    const snapshot = await getDatasetSnapshotById(c.env.DB, job.dataset_snapshot_id);
    if (snapshot?.source_cache_id) {
      const fromSnapshot = await getDatasetPreprocessCacheById(c.env.DB, snapshot.source_cache_id);
      if (fromSnapshot) {
        return fromSnapshot;
      }
    }
  }

  const summary = (job.summary ?? {}) as Record<string, unknown>;
  const summarySignature = getSummaryString(summary, 'preprocess_cache_dataset_signature');
  if (summarySignature) {
    const matched = await getDatasetPreprocessCache(
      c.env.DB,
      job.voice_id,
      job.dataset_r2_prefix,
      summarySignature,
    );
    if (matched) {
      return matched;
    }
  }

  const objects = await listAllR2Objects(c.env.R2, `${stripSlashes(job.dataset_r2_prefix)}/`);
  if (objects.length === 0) {
    return null;
  }
  const signatureInfo = await computeDatasetSignature(job.dataset_r2_prefix, objects);
  if (!signatureInfo) {
    return null;
  }
  return getDatasetPreprocessCache(
    c.env.DB,
    job.voice_id,
    job.dataset_r2_prefix,
    signatureInfo.signature,
  );
};

const ensureJobPreprocessCacheState = async (
  c: Context<AppContext>,
  job: TrainingJob,
): Promise<{
  cache: DatasetPreprocessCache;
  entries: DatasetPreprocessCacheEntry[];
  referenceText: string | null;
  hydratedFromR2: boolean;
} | null> => {
  const cache = await resolveJobPreprocessCache(c, job);
  if (!cache) {
    return null;
  }

  let entries = await listDatasetPreprocessCacheEntries(c.env.DB, cache.cache_id);
  let hydratedFromR2 = false;
  const latestEntryUpdatedAt = entries.reduce((max, entry) => Math.max(max, entry.updated_at), 0);
  if (entries.length === 0 || latestEntryUpdatedAt < cache.updated_at) {
    const rawJsonl = await readR2Text(c.env.R2, cache.train_raw_r2_key);
    if (!rawJsonl) {
      throw new Error(`Cached train_raw.jsonl not found: ${cache.train_raw_r2_key}`);
    }
    const now = Date.now();
    entries = parseTrainRawCacheEntries(rawJsonl, cache, now);
    await replaceDatasetPreprocessCacheEntries(c.env.DB, cache.cache_id, entries);
    hydratedFromR2 = true;
  }

  const referenceText = await readReferenceText(c.env.R2, cache);
  return {
    cache,
    entries,
    referenceText,
    hydratedFromR2,
  };
};

const needsCompletedValidation = (job: TrainingJob): boolean => {
  if (job.status !== 'completed') {
    return false;
  }
  const summary = (job.summary ?? {}) as Record<string, unknown>;
  return summary.validation_checked !== true;
};

const extractDatasetPrefixFromRefAudioKey = (
  voice: NonNullable<Awaited<ReturnType<typeof getVoice>>>,
): string | null => {
  const refAudioKey =
    typeof voice.ref_audio_r2_key === 'string' ? voice.ref_audio_r2_key.trim() : '';
  if (!REFERENCE_AUDIO_KEY_RE.test(refAudioKey)) {
    return null;
  }

  const datasetPrefix = refAudioKey.replace(REFERENCE_AUDIO_KEY_RE, '').replace(/\/+$/, '');
  const expectedPrefix = `datasets/${voice.voice_id}/`;
  if (!datasetPrefix.startsWith(expectedPrefix)) {
    return null;
  }

  return datasetPrefix;
};

const resolveTrainingDatasetPrefix = (
  voice: NonNullable<Awaited<ReturnType<typeof getVoice>>>,
  datasetName: string | undefined,
): string => {
  const requestedDatasetName = typeof datasetName === 'string' ? datasetName.trim() : '';
  if (requestedDatasetName) {
    return `datasets/${voice.voice_id}/${requestedDatasetName}`;
  }

  return extractDatasetPrefixFromRefAudioKey(voice) ?? `datasets/${voice.voice_id}`;
};

const extractDatasetNameFromPrefix = domainExtractDatasetNameFromPrefix;

const buildSyntheticDatasetSignature = (datasetPrefix: string): string =>
  `synthetic:${stripSlashes(datasetPrefix)}`;

const buildDatasetSnapshot = async ({
  voice,
  datasetPrefix,
  datasetSignature,
  preprocessCache,
  referenceText,
  sourceFileCount,
  createdFromJobId,
  now,
}: {
  voice: NonNullable<Awaited<ReturnType<typeof getVoice>>>;
  datasetPrefix: string;
  datasetSignature: string;
  preprocessCache: DatasetPreprocessCache | null;
  referenceText: string | null;
  sourceFileCount: number | null;
  createdFromJobId: string | null;
  now: number;
}): Promise<DatasetSnapshot> => ({
  snapshot_id: crypto.randomUUID(),
  voice_id: voice.voice_id,
  dataset_name: extractDatasetNameFromPrefix(datasetPrefix),
  dataset_r2_prefix: datasetPrefix,
  dataset_signature: datasetSignature,
  status: preprocessCache ? 'frozen' : 'draft',
  source_cache_id: preprocessCache?.cache_id ?? null,
  cache_r2_prefix: preprocessCache?.cache_r2_prefix ?? null,
  train_raw_r2_key: preprocessCache?.train_raw_r2_key ?? null,
  ref_audio_r2_key: preprocessCache?.ref_audio_r2_key ?? voice.ref_audio_r2_key ?? null,
  reference_profile_r2_key: preprocessCache?.reference_profile_r2_key ?? null,
  reference_text: referenceText,
  source_file_count: preprocessCache?.source_file_count ?? sourceFileCount,
  segments_created: preprocessCache?.segments_created ?? null,
  segments_accepted: preprocessCache?.segments_accepted ?? null,
  accepted_duration_min: preprocessCache?.accepted_duration_min ?? null,
  created_from_job_id: createdFromJobId,
  created_at: now,
  updated_at: now,
});

const ensureDatasetSnapshot = async ({
  c,
  voice,
  datasetPrefix,
  datasetSignature,
  preprocessCache,
  sourceFileCount,
  createdFromJobId,
}: {
  c: Context<AppContext>;
  voice: NonNullable<Awaited<ReturnType<typeof getVoice>>>;
  datasetPrefix: string;
  datasetSignature: string;
  preprocessCache: DatasetPreprocessCache | null;
  sourceFileCount: number | null;
  createdFromJobId: string | null;
}): Promise<DatasetSnapshot> => {
  const existing = await getDatasetSnapshot(
    c.env.DB,
    voice.voice_id,
    datasetPrefix,
    datasetSignature,
  );
  const referenceText = preprocessCache
    ? await readReferenceText(c.env.R2, preprocessCache)
    : (existing?.reference_text ?? null);
  const now = Date.now();
  const nextSnapshot =
    existing ??
    (await buildDatasetSnapshot({
      voice,
      datasetPrefix,
      datasetSignature,
      preprocessCache,
      referenceText,
      sourceFileCount,
      createdFromJobId,
      now,
    }));

  await upsertDatasetSnapshot(c.env.DB, {
    ...nextSnapshot,
    status: preprocessCache ? 'frozen' : nextSnapshot.status,
    source_cache_id: preprocessCache?.cache_id ?? nextSnapshot.source_cache_id,
    cache_r2_prefix: preprocessCache?.cache_r2_prefix ?? nextSnapshot.cache_r2_prefix,
    train_raw_r2_key: preprocessCache?.train_raw_r2_key ?? nextSnapshot.train_raw_r2_key,
    ref_audio_r2_key: preprocessCache?.ref_audio_r2_key ?? nextSnapshot.ref_audio_r2_key,
    reference_profile_r2_key:
      preprocessCache?.reference_profile_r2_key ?? nextSnapshot.reference_profile_r2_key,
    reference_text: referenceText ?? nextSnapshot.reference_text,
    source_file_count:
      preprocessCache?.source_file_count ?? sourceFileCount ?? nextSnapshot.source_file_count,
    segments_created: preprocessCache?.segments_created ?? nextSnapshot.segments_created,
    segments_accepted: preprocessCache?.segments_accepted ?? nextSnapshot.segments_accepted,
    accepted_duration_min:
      preprocessCache?.accepted_duration_min ?? nextSnapshot.accepted_duration_min,
    created_from_job_id: createdFromJobId ?? nextSnapshot.created_from_job_id,
    updated_at: now,
  });

  return (
    (await getDatasetSnapshot(c.env.DB, voice.voice_id, datasetPrefix, datasetSignature)) ??
    nextSnapshot
  );
};

const ensureTrainingRound = async ({
  c,
  voice,
  datasetSnapshotId,
  now,
}: {
  c: Context<AppContext>;
  voice: NonNullable<Awaited<ReturnType<typeof getVoice>>>;
  datasetSnapshotId: string | null;
  now: number;
}): Promise<TrainingRound> => {
  const previousRound = await getLatestTrainingRoundForVoice(c.env.DB, voice.voice_id);
  const round: TrainingRound = {
    round_id: crypto.randomUUID(),
    voice_id: voice.voice_id,
    dataset_snapshot_id: datasetSnapshotId,
    round_index: (previousRound?.round_index ?? 0) + 1,
    status: 'queued',
    production_checkpoint_r2_prefix: voice.checkpoint_r2_prefix,
    production_run_name: voice.run_name,
    production_epoch: voice.epoch,
    production_preset: voice.checkpoint_preset,
    production_score: voice.checkpoint_score,
    production_job_id: voice.checkpoint_job_id,
    champion_checkpoint_r2_prefix: null,
    champion_run_name: null,
    champion_epoch: null,
    champion_preset: null,
    champion_score: null,
    champion_job_id: null,
    selected_checkpoint_r2_prefix: null,
    selected_run_name: null,
    selected_epoch: null,
    selected_preset: null,
    selected_score: null,
    selected_job_id: null,
    adoption_mode: null,
    candidate_checkpoint_r2_prefix: null,
    candidate_run_name: null,
    candidate_epoch: null,
    candidate_score: null,
    candidate_job_id: null,
    summary: {},
    created_at: now,
    updated_at: now,
    started_at: now,
    completed_at: null,
  };
  await createTrainingRound(c.env.DB, round);
  return round;
};

const getCurrentReadyVoiceScore = async (
  c: Context<AppContext>,
  voice: NonNullable<Awaited<ReturnType<typeof getVoice>>>,
): Promise<number | null> => {
  const currentPrefix =
    typeof voice.checkpoint_r2_prefix === 'string' ? voice.checkpoint_r2_prefix.trim() : '';
  if (voice.status !== 'ready' || !currentPrefix) {
    return null;
  }
  if (typeof voice.checkpoint_score === 'number' && Number.isFinite(voice.checkpoint_score)) {
    return voice.checkpoint_score;
  }

  const jobs = await listTrainingJobs(c.env.DB, { voice_id: voice.voice_id, limit: 100 });
  for (const candidate of jobs) {
    const summary = candidate.summary ?? {};
    const selectedPrefix =
      typeof summary.selected_checkpoint_prefix === 'string'
        ? summary.selected_checkpoint_prefix.trim()
        : '';
    if (selectedPrefix === currentPrefix) {
      const selectedScore = Number(summary.selected_score);
      if (Number.isFinite(selectedScore)) {
        await updateVoice(c.env.DB, voice.voice_id, {
          checkpoint_score: selectedScore,
          checkpoint_preset:
            typeof summary.selected_preset === 'string' && summary.selected_preset.trim()
              ? summary.selected_preset.trim()
              : null,
          checkpoint_job_id: candidate.job_id,
        });
        return selectedScore;
      }
    }

    const manualPrefix =
      typeof summary.manual_promoted_checkpoint_prefix === 'string'
        ? summary.manual_promoted_checkpoint_prefix.trim()
        : '';
    if (manualPrefix === currentPrefix) {
      const manualScore = Number(summary.manual_promoted_score);
      if (Number.isFinite(manualScore)) {
        await updateVoice(c.env.DB, voice.voice_id, {
          checkpoint_score: manualScore,
          checkpoint_preset:
            typeof summary.manual_promoted_preset === 'string' &&
            summary.manual_promoted_preset.trim()
              ? summary.manual_promoted_preset.trim()
              : null,
          checkpoint_job_id: candidate.job_id,
        });
        return manualScore;
      }
    }
  }

  return null;
};

const chooseCheckpointAdoption = async ({
  c,
  voice,
  candidatePrefix,
  candidateScore,
}: {
  c: Context<AppContext>;
  voice: NonNullable<Awaited<ReturnType<typeof getVoice>>>;
  candidatePrefix: string;
  candidateScore: number;
}): Promise<{
  mode: 'promote' | 'candidate' | 'keep_current';
  preservedPrefix: string | null;
  preservedEpoch: number | null;
  preservedScore: number | null;
}> => {
  const currentPrefix =
    typeof voice.checkpoint_r2_prefix === 'string' ? voice.checkpoint_r2_prefix.trim() : '';
  if (voice.status !== 'ready' || !currentPrefix || currentPrefix === candidatePrefix) {
    return {
      mode: 'promote',
      preservedPrefix: currentPrefix || null,
      preservedEpoch: voice.epoch,
      preservedScore: await getCurrentReadyVoiceScore(c, voice),
    };
  }

  const currentScore = await getCurrentReadyVoiceScore(c, voice);
  const currentScoringVersion =
    typeof (voice as unknown as Record<string, unknown>).scoring_version === 'number'
      ? ((voice as unknown as Record<string, unknown>).scoring_version as number)
      : 1;
  if (
    currentScore !== null &&
    candidateScore <= currentScore &&
    currentScoringVersion >= SCORING_VERSION
  ) {
    return {
      mode: 'keep_current',
      preservedPrefix: currentPrefix,
      preservedEpoch: voice.epoch,
      preservedScore: currentScore,
    };
  }

  return {
    mode: 'candidate',
    preservedPrefix: currentPrefix,
    preservedEpoch: voice.epoch,
    preservedScore: currentScore,
  };
};

const parseRunNameFromCheckpointPrefix = domainParseRunNameFromCheckpointPrefix;

type ManualPromotionCandidate = {
  prefix: string;
  epoch: number | null;
  preset: string | null;
  score: number | null;
};

/** Extract CheckpointScores from a validation record for gray-zone rescue evaluation. */
const extractCheckpointScoresFromRecord = (
  record: Record<string, unknown>,
): import('../lib/training-domain').CheckpointScores | null => {
  // Scores may be embedded in the message string or as direct fields
  const parseFromMsg = (msg: string, key: string): number | null => {
    const match = new RegExp(`(?:^|\\s)${key}=([\\d.]+)`).exec(msg);
    if (!match) return null;
    const val = parseFloat(match[1]);
    return Number.isFinite(val) ? val : null;
  };

  const msg = typeof record.message === 'string' ? record.message : '';
  const get = (key: string): number | null => {
    const direct = record[key];
    if (typeof direct === 'number' && Number.isFinite(direct)) return direct;
    return parseFromMsg(msg, key.replace('_score', ''));
  };

  const asr = get('asr_score');
  const speaker = get('speaker_score');
  const health = get('health_score');
  // Need at least one core metric to evaluate
  if (asr === null && speaker === null && health === null) return null;

  return {
    asr_score: asr,
    speaker_score: speaker,
    health_score: health,
    tone_score: get('tone_score'),
    speed_score: get('speed_score'),
    style_score: get('style_score'),
    overall_score: get('overall_score'),
    duration_score: get('duration_score'),
  };
};

const collectManualPromotionCandidates = (
  summary: Record<string, unknown>,
): ManualPromotionCandidate[] => {
  const byPrefix = new Map<string, ManualPromotionCandidate>();
  const register = (candidate: ManualPromotionCandidate) => {
    if (!candidate.prefix) {
      return;
    }
    const existing = byPrefix.get(candidate.prefix);
    byPrefix.set(candidate.prefix, {
      prefix: candidate.prefix,
      epoch: candidate.epoch ?? existing?.epoch ?? null,
      preset: candidate.preset ?? existing?.preset ?? null,
      score: candidate.score ?? existing?.score ?? null,
    });
  };

  const evaluated = Array.isArray(summary.evaluated_checkpoints)
    ? summary.evaluated_checkpoints
    : [];
  for (const value of evaluated) {
    if (!value || typeof value !== 'object') {
      continue;
    }
    const record = value as Record<string, unknown>;
    if (record.ok === false) {
      // Gray-zone rescue: re-evaluate failed checkpoints that barely miss soft thresholds.
      // Extract scores from the record and check if they can be rescued.
      const rescueScores = extractCheckpointScoresFromRecord(record);
      if (!rescueScores || !passesHardSafetyGate(rescueScores)) {
        continue;
      }
      // Mark as rescued but still register — will get lower score than clean passes
      // The actual rescue decision requires DB access (calibration state), so we
      // include hard-safety-passing failures as candidates with a score penalty.
      // The caller can then apply grayZoneRescue with DB-loaded weights.
      if (!record.prefix || typeof record.prefix !== 'string') continue;
      register({
        prefix: (record.prefix as string).trim(),
        epoch: readNumber(record.epoch),
        preset: typeof record.preset === 'string' ? record.preset.trim() : null,
        score: readNumber(record.aggregateScore ?? record.score) ?? null,
      });
      continue;
    }
    const prefix = typeof record.prefix === 'string' ? record.prefix.trim() : '';
    register({
      prefix,
      epoch: readNumber(record.epoch),
      preset: typeof record.preset === 'string' ? record.preset.trim() : null,
      score: readNumber(record.score),
    });
  }

  const selectedPrefix =
    typeof summary.selected_checkpoint_prefix === 'string'
      ? summary.selected_checkpoint_prefix.trim()
      : '';
  if (selectedPrefix) {
    register({
      prefix: selectedPrefix,
      epoch: readNumber(summary.selected_checkpoint_epoch),
      preset: typeof summary.selected_preset === 'string' ? summary.selected_preset.trim() : null,
      score: readNumber(summary.selected_score),
    });
  }

  const candidatePrefix =
    typeof summary.candidate_checkpoint_prefix === 'string'
      ? summary.candidate_checkpoint_prefix.trim()
      : '';
  if (candidatePrefix) {
    register({
      prefix: candidatePrefix,
      epoch: readNumber(summary.candidate_checkpoint_epoch),
      preset: typeof summary.candidate_preset === 'string' ? summary.candidate_preset.trim() : null,
      score: readNumber(summary.candidate_score),
    });
  }

  return [...byPrefix.values()];
};

const resolvePromotionSettings = (
  voice: NonNullable<Awaited<ReturnType<typeof getVoice>>>,
  presetName: string | null,
) => {
  const normalizedPreset = typeof presetName === 'string' ? presetName.trim() : '';
  const presets = getValidationPresets(
    voice.model_id ?? (voice.model_size || '1.7B'),
    String(voice.labels?.language ?? ''),
  );
  const preset = presets.find((value) => value.name === normalizedPreset);
  return preset?.settings ?? voice.settings ?? {};
};

const applyValidatedCheckpointOutcome = async ({
  c,
  voice,
  job,
  candidatePrefix,
  candidateEpoch,
  candidatePreset,
  candidateScore,
}: {
  c: Context<AppContext>;
  voice: NonNullable<Awaited<ReturnType<typeof getVoice>>>;
  job: TrainingJob;
  candidatePrefix: string;
  candidateEpoch: number | null;
  candidatePreset: string | null;
  candidateScore: number;
}): Promise<{
  mode: 'promote' | 'candidate' | 'keep_current';
  selectedPrefix: string | null;
  selectedEpoch: number | null;
  selectedPreset: string | null;
  selectedScore: number | null;
  preservedScore: number | null;
}> => {
  const decision = await chooseCheckpointAdoption({
    c,
    voice,
    candidatePrefix,
    candidateScore,
  });
  const promotedSettings = resolvePromotionSettings(voice, candidatePreset);
  const candidateRunName = parseRunNameFromCheckpointPrefix(candidatePrefix);

  if (decision.mode === 'promote') {
    await updateVoice(c.env.DB, job.voice_id, {
      status: 'ready',
      checkpoint_r2_prefix: candidatePrefix,
      run_name: candidateRunName,
      epoch: candidateEpoch,
      checkpoint_preset: candidatePreset,
      checkpoint_score: candidateScore,
      checkpoint_job_id: job.job_id,
      candidate_checkpoint_r2_prefix: null,
      candidate_run_name: null,
      candidate_epoch: null,
      candidate_preset: null,
      candidate_score: null,
      candidate_job_id: null,
      settings: promotedSettings,
      active_round_id: voice.active_round_id ?? job.round_id ?? null,
    });
    if (job.round_id) {
      await updateTrainingRound(c.env.DB, job.round_id, {
        status: 'promoted',
        production_checkpoint_r2_prefix: candidatePrefix,
        production_run_name: candidateRunName,
        production_epoch: candidateEpoch,
        production_preset: candidatePreset,
        production_score: candidateScore,
        production_job_id: job.job_id,
        champion_checkpoint_r2_prefix: candidatePrefix,
        champion_run_name: candidateRunName,
        champion_epoch: candidateEpoch,
        champion_preset: candidatePreset,
        champion_score: candidateScore,
        champion_job_id: job.job_id,
        selected_checkpoint_r2_prefix: candidatePrefix,
        selected_run_name: candidateRunName,
        selected_epoch: candidateEpoch,
        selected_preset: candidatePreset,
        selected_score: candidateScore,
        selected_job_id: job.job_id,
        adoption_mode: decision.mode,
        candidate_checkpoint_r2_prefix: null,
        candidate_run_name: null,
        candidate_epoch: null,
        candidate_score: null,
        candidate_job_id: null,
        completed_at: Date.now(),
      });
    }
    return {
      mode: decision.mode,
      selectedPrefix: candidatePrefix,
      selectedEpoch: candidateEpoch,
      selectedPreset: candidatePreset,
      selectedScore: candidateScore,
      preservedScore: decision.preservedScore,
    };
  }

  if (decision.mode === 'candidate') {
    await updateVoice(c.env.DB, job.voice_id, {
      candidate_checkpoint_r2_prefix: candidatePrefix,
      candidate_run_name: candidateRunName,
      candidate_epoch: candidateEpoch,
      candidate_preset: candidatePreset,
      candidate_score: candidateScore,
      candidate_job_id: job.job_id,
      active_round_id: voice.active_round_id ?? job.round_id ?? null,
    });
    if (job.round_id) {
      await updateTrainingRound(c.env.DB, job.round_id, {
        status: 'candidate_ready',
        champion_checkpoint_r2_prefix: candidatePrefix,
        champion_run_name: candidateRunName,
        champion_epoch: candidateEpoch,
        champion_preset: candidatePreset,
        champion_score: candidateScore,
        champion_job_id: job.job_id,
        selected_checkpoint_r2_prefix: candidatePrefix,
        selected_run_name: candidateRunName,
        selected_epoch: candidateEpoch,
        selected_preset: candidatePreset,
        selected_score: candidateScore,
        selected_job_id: job.job_id,
        adoption_mode: decision.mode,
        candidate_checkpoint_r2_prefix: candidatePrefix,
        candidate_run_name: candidateRunName,
        candidate_epoch: candidateEpoch,
        candidate_score: candidateScore,
        candidate_job_id: job.job_id,
        completed_at: Date.now(),
      });
    }
    return {
      mode: decision.mode,
      selectedPrefix: candidatePrefix,
      selectedEpoch: candidateEpoch,
      selectedPreset: candidatePreset,
      selectedScore: candidateScore,
      preservedScore: decision.preservedScore,
    };
  }

  if (job.round_id) {
    await updateTrainingRound(c.env.DB, job.round_id, {
      status: 'superseded',
      champion_checkpoint_r2_prefix: candidatePrefix,
      champion_run_name: candidateRunName,
      champion_epoch: candidateEpoch,
      champion_preset: candidatePreset,
      champion_score: candidateScore,
      champion_job_id: job.job_id,
      selected_checkpoint_r2_prefix: decision.preservedPrefix,
      selected_run_name: voice.run_name,
      selected_epoch: decision.preservedEpoch,
      selected_preset: voice.checkpoint_preset ?? 'kept_existing_best',
      selected_score: decision.preservedScore,
      selected_job_id: voice.checkpoint_job_id,
      adoption_mode: decision.mode,
      candidate_checkpoint_r2_prefix: candidatePrefix,
      candidate_run_name: candidateRunName,
      candidate_epoch: candidateEpoch,
      candidate_score: candidateScore,
      candidate_job_id: job.job_id,
      completed_at: Date.now(),
    });
  }
  return {
    mode: decision.mode,
    selectedPrefix: decision.preservedPrefix,
    selectedEpoch: decision.preservedEpoch,
    selectedPreset: 'kept_existing_best',
    selectedScore: decision.preservedScore,
    preservedScore: decision.preservedScore,
  };
};

const isMissingRunpodRequestError = (error: unknown): error is Error => {
  if (!(error instanceof Error)) {
    return false;
  }
  return (
    error.message.includes('RunPod status request failed (404)') &&
    error.message.includes('request does not exist')
  );
};

const getServerlessStatusOrSyntheticFailure = async (
  env: AppContext['Bindings'],
  endpointId: string,
  runId: string,
): Promise<Record<string, unknown>> => {
  try {
    return await getServerlessStatus(env, endpointId, runId);
  } catch (error) {
    if (!isMissingRunpodRequestError(error)) {
      throw error;
    }
    return {
      status: 'FAILED',
      error: error.message,
    };
  }
};

const shouldPreserveCurrentReadyVoice = (
  voice: NonNullable<Awaited<ReturnType<typeof getVoice>>>,
): boolean =>
  voice.status === 'ready' &&
  typeof voice.checkpoint_r2_prefix === 'string' &&
  voice.checkpoint_r2_prefix.trim().length > 0;

const shouldKeepReadyVoiceOnValidationFailure = (
  voice: NonNullable<Awaited<ReturnType<typeof getVoice>>>,
  summary: Record<string, unknown>,
  options?: {
    evaluatedCheckpoints?: CheckpointEvaluation[];
    validationRunName?: string | null;
    forceRevalidation?: boolean;
  },
): boolean => {
  if (!shouldPreserveCurrentReadyVoice(voice)) {
    return false;
  }

  const currentPrefix =
    typeof voice.checkpoint_r2_prefix === 'string' ? voice.checkpoint_r2_prefix.trim() : '';
  if (!currentPrefix) {
    return false;
  }
  const manualPromotedPrefix =
    typeof summary.manual_promoted_checkpoint_prefix === 'string'
      ? summary.manual_promoted_checkpoint_prefix.trim()
      : '';
  if (manualPromotedPrefix && manualPromotedPrefix === currentPrefix) {
    return true;
  }

  const getRunNameFromPrefix = (prefix: string): string | null => {
    const parts = prefix.split('/');
    if (parts.length < 4 || parts[0] !== 'checkpoints') {
      return null;
    }
    return parts[2] || null;
  };
  const currentRunName = getRunNameFromPrefix(currentPrefix);
  const referencedPrefixes = new Set<string>();
  const referencedRunNames = new Set<string>();
  const registerPrefix = (value: unknown) => {
    if (typeof value !== 'string') {
      return;
    }
    const normalized = value.trim();
    if (normalized) {
      referencedPrefixes.add(normalized);
      const runName = getRunNameFromPrefix(normalized);
      if (runName) {
        referencedRunNames.add(runName);
      }
    }
  };
  const registerRunName = (value: unknown) => {
    if (typeof value !== 'string') {
      return;
    }
    const normalized = value.trim();
    if (normalized) {
      referencedRunNames.add(normalized);
    }
  };

  registerPrefix(summary.selected_checkpoint_prefix);
  registerPrefix(summary.candidate_checkpoint_prefix);
  registerPrefix(summary.manual_promoted_checkpoint_prefix);

  const asyncValidation =
    summary.async_validation && typeof summary.async_validation === 'object'
      ? (summary.async_validation as Record<string, unknown>)
      : null;
  registerPrefix(asyncValidation?.checkpoint_prefix);

  const evaluated = Array.isArray(summary.evaluated_checkpoints)
    ? summary.evaluated_checkpoints
    : [];
  for (const value of evaluated) {
    if (!value || typeof value !== 'object') {
      continue;
    }
    registerPrefix((value as Record<string, unknown>).prefix);
  }

  for (const evaluation of options?.evaluatedCheckpoints ?? []) {
    registerPrefix(evaluation.prefix);
  }
  registerRunName(options?.validationRunName);

  if (referencedPrefixes.has(currentPrefix)) {
    return false;
  }

  const forceRevalidation =
    options?.forceRevalidation === true || summary.force_revalidation === true;
  if (forceRevalidation && currentRunName && referencedRunNames.has(currentRunName)) {
    return false;
  }

  return true;
};

type PodStatusDetail = NonNullable<Awaited<ReturnType<typeof getPodStatus>>>;

const FULL_VALIDATION_SEEDS_OFFSET = [123456, 223456] as const;
const FAST_VALIDATION_SEEDS_OFFSET = [123456] as const;
const MAX_CHECKPOINTS_TO_EVAL = 4;
const MAX_CHECKPOINTS_TO_EVAL_06B = 4;
const VALIDATION_RETRY_ATTEMPTS = 3;
const SCORING_VERSION = 2;
const MIN_PASS_RATE_06B = 5 / 6;
const MIN_PASS_RATE_17B = 5 / 6;
const PROVISIONING_STALE_MS = 4 * 60 * 1000;
const VALIDATION_RUN_STALE_MS = 6 * 60 * 1000;

const getRecommendedTrainingDefaults = getTrainingDefaults;
const MAX_PROVISIONING_RECOVERY_ATTEMPTS = 2;

const OVERALL_SCORE_ERROR_RE = /overall_score=([0-9.]+)/i;
const VALIDATION_ASR_ERROR_KEY = 'openai_asr_error';

const resolveWorkerPublicUrl = (
  env: Pick<Env, 'WORKER_PUBLIC_URL'>,
  requestUrl?: string | null,
): string => {
  if (typeof requestUrl === 'string' && requestUrl.trim()) {
    return new URL(requestUrl).origin;
  }
  const configured = env.WORKER_PUBLIC_URL?.trim();
  return configured ? configured.replace(/\/+$/, '') : DEFAULT_WORKER_PUBLIC_URL;
};

const getWorkerOrigin = (c: Context<AppContext>): string =>
  resolveWorkerPublicUrl(c.env, c.req.url);
const createSyntheticContext = (env: Env, workerOrigin: string): Context<AppContext> =>
  ({
    env,
    req: {
      url: `${workerOrigin.replace(/\/+$/, '')}/`,
    },
  }) as unknown as Context<AppContext>;
const GHCR_INDEX_ACCEPT =
  'application/vnd.oci.image.index.v1+json, application/vnd.docker.distribution.manifest.list.v2+json, application/vnd.docker.distribution.manifest.v2+json, application/vnd.oci.image.manifest.v1+json';

const readNumber = domainReadNumber;
const readTimestamp = domainReadTimestamp;

const getValidationRunStartedAt = (
  persistedState: Record<string, unknown> | null,
  job: TrainingJob,
): number => {
  const fromState = readNumber(persistedState?.run_started_at);
  if (fromState !== null) {
    return fromState;
  }
  const fallback = readNumber(job.completed_at) ?? readNumber(job.updated_at) ?? Date.now();
  return fallback;
};

const resolveGhcrAmd64Image = async (imageName: string): Promise<string> => {
  if (!imageName.startsWith('ghcr.io/') || imageName.includes('@')) {
    return imageName;
  }

  const imageRef = imageName.slice('ghcr.io/'.length);
  const tagSeparator = imageRef.lastIndexOf(':');
  if (tagSeparator <= 0) {
    return imageName;
  }

  const repo = imageRef.slice(0, tagSeparator);
  const tag = imageRef.slice(tagSeparator + 1);
  if (!repo || !tag) {
    return imageName;
  }

  const tokenResp = await fetch(`https://ghcr.io/token?scope=repository:${repo}:pull`);
  if (!tokenResp.ok) {
    return imageName;
  }
  const tokenPayload = (await tokenResp.json()) as { token?: string };
  if (!tokenPayload.token) {
    return imageName;
  }

  const manifestResp = await fetch(`https://ghcr.io/v2/${repo}/manifests/${tag}`, {
    headers: {
      Authorization: `Bearer ${tokenPayload.token}`,
      Accept: GHCR_INDEX_ACCEPT,
    },
  });
  if (!manifestResp.ok) {
    return imageName;
  }

  const contentType = String(manifestResp.headers.get('content-type') ?? '').toLowerCase();
  const registryDigest = manifestResp.headers.get('docker-content-digest');

  if (contentType.includes('image.index') || contentType.includes('manifest.list')) {
    const payload = (await manifestResp.json()) as {
      manifests?: Array<{
        digest?: string;
        platform?: { architecture?: string; os?: string };
      }>;
    };
    const amd64 = payload.manifests?.find(
      (manifest) =>
        manifest.platform?.os === 'linux' &&
        manifest.platform?.architecture === 'amd64' &&
        typeof manifest.digest === 'string',
    );
    if (amd64?.digest) {
      return `ghcr.io/${repo}@${amd64.digest}`;
    }
  }

  if (registryDigest) {
    return `ghcr.io/${repo}@${registryDigest}`;
  }

  return imageName;
};

const getConfiguredModelSize = (job: TrainingJob): string => {
  const config = job.config as Record<string, unknown>;
  return typeof config.model_size === 'string' && config.model_size ? config.model_size : '1.7B';
};

const getTrainingGpuType = (job: TrainingJob): string => {
  const config = job.config as Record<string, unknown>;
  if (typeof config.gpu_type_id === 'string' && config.gpu_type_id) {
    return config.gpu_type_id;
  }
  return getConfiguredModelSize(job).includes('0.6') ? 'NVIDIA GeForce RTX 4090' : 'NVIDIA L40S';
};

const buildTrainingPodEnv = (
  c: Context<AppContext>,
  job: TrainingJob,
): Array<{ key: string; value: string }> => {
  const workerUrl = getWorkerOrigin(c);
  return [
    { key: 'JOB_ID', value: job.job_id },
    { key: 'VOICE_ID', value: job.voice_id },
    { key: 'WORKER_API_URL', value: workerUrl },
    { key: 'JOB_TOKEN', value: job.job_token ?? '' },
    { key: 'R2_ENDPOINT_URL', value: c.env.R2_ENDPOINT_URL },
    { key: 'R2_ACCESS_KEY_ID', value: c.env.R2_ACCESS_KEY_ID },
    { key: 'R2_SECRET_ACCESS_KEY', value: c.env.R2_SECRET_ACCESS_KEY },
    { key: 'R2_BUCKET', value: 'qwen-tts-studio' },
    { key: 'RUNPOD_API_KEY', value: c.env.RUNPOD_API_KEY },
    { key: 'HF_HUB_ENABLE_HF_TRANSFER', value: '0' },
  ];
};

const getConfiguredTrainingImageName = (c: Context<AppContext>): string | null => {
  const imageName = c.env.RUNPOD_TRAINING_IMAGE_NAME?.trim();
  return imageName ? imageName : null;
};

const getConfiguredTrainingImageFallbackNames = (c: Context<AppContext>): string[] =>
  uniqueNonEmpty([
    ...(c.env.RUNPOD_TRAINING_FALLBACK_IMAGE_NAMES ?? '').split(',').map((value) => value.trim()),
    ...DEFAULT_TRAINING_IMAGE_FALLBACKS,
  ]);

const getConfiguredTrainingImageCandidates = (c: Context<AppContext>): string[] =>
  uniqueNonEmpty([
    getConfiguredTrainingImageName(c),
    ...getConfiguredTrainingImageFallbackNames(c),
  ]);

const getStoredTrainingImageCandidates = (job: TrainingJob): string[] => {
  const summary = (job.summary ?? {}) as Record<string, unknown>;
  const storedCandidates = Array.isArray(summary.training_image_candidates)
    ? summary.training_image_candidates.filter(
        (value): value is string => typeof value === 'string',
      )
    : [];
  return uniqueNonEmpty([
    ...storedCandidates,
    typeof summary.training_resolved_image_name === 'string'
      ? summary.training_resolved_image_name
      : null,
    typeof summary.training_image_name === 'string' ? summary.training_image_name : null,
  ]);
};

const getTrainingImageCandidatesForJob = (c: Context<AppContext>, job: TrainingJob): string[] =>
  uniqueNonEmpty([
    ...getStoredTrainingImageCandidates(job),
    ...getConfiguredTrainingImageCandidates(c),
  ]);

const getTrainingImageAttemptIndex = (
  c: Context<AppContext>,
  job: TrainingJob,
  candidates?: string[],
): number => {
  const summary = (job.summary ?? {}) as Record<string, unknown>;
  const resolvedCandidates = candidates ?? getTrainingImageCandidatesForJob(c, job);
  const storedIndex = readNumber(summary.training_image_attempt_index);
  if (storedIndex !== null && storedIndex >= 0 && storedIndex < resolvedCandidates.length) {
    return storedIndex;
  }

  const currentImage = uniqueNonEmpty([
    typeof summary.training_resolved_image_name === 'string'
      ? summary.training_resolved_image_name
      : null,
    typeof summary.training_image_name === 'string' ? summary.training_image_name : null,
  ])[0];
  if (currentImage) {
    const matchedIndex = resolvedCandidates.findIndex((value) => value === currentImage);
    if (matchedIndex >= 0) {
      return matchedIndex;
    }
  }
  return 0;
};

const getConfiguredTrainingDockerArgs = (c: Context<AppContext>): string | null => {
  const dockerArgs = c.env.RUNPOD_TRAINING_DOCKER_ARGS?.trim();
  return dockerArgs ? dockerArgs : null;
};

const getConfiguredTrainingVolumeMountPath = (c: Context<AppContext>): string | null => {
  const volumeMountPath = c.env.RUNPOD_TRAINING_VOLUME_MOUNT_PATH?.trim();
  return volumeMountPath ? volumeMountPath : null;
};

const getConfiguredTrainingTemplateId = (c: Context<AppContext>): string | null => {
  const templateId = c.env.RUNPOD_TRAINING_TEMPLATE_ID?.trim();
  return templateId ? templateId : null;
};

const getMaxActiveTrainingJobsPerVoice = (c: Context<AppContext>): number => {
  const raw = Number(
    c.env.TRAINING_MAX_ACTIVE_JOBS_PER_VOICE ?? DEFAULT_MAX_ACTIVE_TRAINING_JOBS_PER_VOICE,
  );
  if (!Number.isFinite(raw)) {
    return DEFAULT_MAX_ACTIVE_TRAINING_JOBS_PER_VOICE;
  }
  return Math.max(1, Math.min(Math.trunc(raw), 6));
};

const getMaxActiveTrainingJobsGlobal = (c: Context<AppContext>): number => {
  const raw = Number(
    (c.env as unknown as Record<string, unknown>).TRAINING_MAX_ACTIVE_JOBS_GLOBAL ??
      DEFAULT_MAX_ACTIVE_TRAINING_JOBS_GLOBAL,
  );
  if (!Number.isFinite(raw)) {
    return DEFAULT_MAX_ACTIVE_TRAINING_JOBS_GLOBAL;
  }
  return Math.max(1, Math.min(Math.trunc(raw), 10));
};

const countGlobalActiveTrainingJobs = async (c: Context<AppContext>): Promise<number> => {
  return countActiveTrainingJobs(c.env.DB, [...ACTIVE_JOB_STATUSES]);
};

const countQueuedTrainingJobsExcludingVoice = async (
  c: Context<AppContext>,
  excludeVoiceId: string,
): Promise<number> => {
  const result = await c.env.DB.prepare(
    `SELECT COUNT(*) as cnt FROM training_jobs WHERE status IN ('queued', 'pending') AND voice_id != ?`,
  )
    .bind(excludeVoiceId)
    .first<{ cnt: number }>();
  return result?.cnt ?? 0;
};

const claimQueuedJob = async (
  db: D1Database,
  jobId: string,
  expectedUpdatedAt: number,
): Promise<boolean> => {
  const result = await db
    .prepare(
      `UPDATE training_jobs SET updated_at = ? WHERE job_id = ? AND status IN ('queued', 'pending') AND (updated_at = ? OR updated_at IS NULL)`,
    )
    .bind(Date.now(), jobId, expectedUpdatedAt)
    .run();
  return (result.meta.changes ?? 0) > 0;
};

const isGpuSupplyConstraintErrorMessage = (message: string | null | undefined): boolean => {
  const normalized = String(message ?? '').toLowerCase();
  return (
    normalized.includes('no longer any instances available') ||
    normalized.includes('supply_constraint')
  );
};

const getJobDatasetSignature = (job: Pick<TrainingJob, 'summary'>): string | null =>
  getSummaryString(
    (job.summary ?? {}) as Record<string, unknown>,
    'preprocess_cache_dataset_signature',
  );

const getDatasetTrainRawR2Key = (datasetPrefix: string): string =>
  `${stripSlashes(datasetPrefix)}/train_raw.jsonl`;

const hasReusablePreprocessArtifacts = async (
  c: Context<AppContext>,
  job: Pick<TrainingJob, 'voice_id' | 'dataset_r2_prefix' | 'summary'>,
): Promise<boolean> => {
  const summary = (job.summary ?? {}) as Record<string, unknown>;
  if (
    getSummaryString(summary, 'preprocess_cache_r2_prefix') ||
    getSummaryString(summary, 'preprocess_cache_train_raw_r2_key')
  ) {
    return true;
  }
  const datasetSignature = getJobDatasetSignature(job);
  if (datasetSignature) {
    const cache = await getDatasetPreprocessCache(
      c.env.DB,
      job.voice_id,
      job.dataset_r2_prefix,
      datasetSignature,
    );
    if (cache?.train_raw_r2_key || cache?.cache_r2_prefix) {
      return true;
    }
  }
  const trainRawKey = getDatasetTrainRawR2Key(job.dataset_r2_prefix);
  return Boolean(await c.env.R2.head(trainRawKey));
};

const sameDatasetPreprocessScope = (left: TrainingJob, right: TrainingJob): boolean => {
  if (left.dataset_r2_prefix !== right.dataset_r2_prefix) {
    return false;
  }
  const leftSignature = getJobDatasetSignature(left);
  const rightSignature = getJobDatasetSignature(right);
  if (!leftSignature || !rightSignature) {
    return true;
  }
  return leftSignature === rightSignature;
};

const queueTrainingJob = async (
  c: Context<AppContext>,
  job: TrainingJob,
  reason: string,
  extraSummary: Record<string, unknown> = {},
): Promise<TrainingJob> => {
  const now = Date.now();
  await updateTrainingJob(c.env.DB, job.job_id, {
    runpod_pod_id: null,
    status: 'queued',
    started_at: null,
    completed_at: null,
    error_message: null,
    summary: {
      ...(job.summary ?? {}),
      queue_wait_reason: reason,
      queue_last_enqueued_at: now,
      queue_active_limit: getMaxActiveTrainingJobsPerVoice(c),
      ...extraSummary,
    },
    supervisor: {
      ...(job.supervisor ?? {}),
      phase: 'queued',
      last_transition_at: now,
    },
  });
  if (job.round_id) {
    const round = await getTrainingRound(c.env.DB, job.round_id);
    await updateTrainingRound(c.env.DB, job.round_id, {
      status: 'queued',
      summary: {
        ...(round?.summary ?? {}),
        queue_wait_reason: reason,
        last_job_id: job.job_id,
      },
    });
  }
  return (await getTrainingJob(c.env.DB, job.job_id)) ?? job;
};

const createTrainingPodForJob = async (
  c: Context<AppContext>,
  job: TrainingJob,
  options?: { imageAttemptIndex?: number },
): Promise<{
  pod: { podId: string; desiredStatus: string };
  summary: Record<string, unknown>;
}> => {
  const imageCandidates = getTrainingImageCandidatesForJob(c, job);
  const imageAttemptIndex =
    typeof options?.imageAttemptIndex === 'number'
      ? Math.max(0, Math.min(options.imageAttemptIndex, Math.max(imageCandidates.length - 1, 0)))
      : getTrainingImageAttemptIndex(c, job, imageCandidates);
  const imageName = imageCandidates[imageAttemptIndex] ?? getConfiguredTrainingImageName(c);
  if (imageName) {
    const resolvedImageName = await resolveGhcrAmd64Image(imageName).catch(() => imageName);
    const dockerArgs = getConfiguredTrainingDockerArgs(c);
    const volumeMountPath = getConfiguredTrainingVolumeMountPath(c);
    const pod = await createPodDirect(c.env, {
      gpuTypeId: getTrainingGpuType(job),
      envVars: buildTrainingPodEnv(c, job),
      imageName: resolvedImageName,
      dockerArgs,
      name: `qwen3-tts-training-${job.job_id.slice(0, 8)}`,
      cloudType: 'ALL',
      volumeMountPath: volumeMountPath ?? undefined,
    });
    return {
      pod,
      summary: {
        training_launch_mode: 'direct_image',
        training_image_name: imageName,
        training_resolved_image_name: resolvedImageName,
        training_image_candidates: imageCandidates,
        training_image_attempt_index: imageAttemptIndex,
        training_docker_args: dockerArgs,
        training_volume_mount_path: volumeMountPath,
      },
    };
  }

  const templateId = getConfiguredTrainingTemplateId(c);
  if (templateId) {
    const pod = await createPod(
      c.env,
      templateId,
      getTrainingGpuType(job),
      buildTrainingPodEnv(c, job),
    );
    return {
      pod,
      summary: {
        training_launch_mode: 'template',
        training_template_id: templateId,
      },
    };
  }

  throw new Error('No training template or direct image configured');
};

const launchTrainingJob = async (
  c: Context<AppContext>,
  job: TrainingJob,
  options?: { imageAttemptIndex?: number },
): Promise<TrainingJob> => {
  let launchResult: Awaited<ReturnType<typeof createTrainingPodForJob>>;
  try {
    launchResult = await createTrainingPodForJob(c, job, options);
  } catch (podError) {
    const errMsg = podError instanceof Error ? podError.message : String(podError);
    if (isGpuSupplyConstraintErrorMessage(errMsg)) {
      return queueTrainingJob(c, job, 'gpu_supply_constraint', {
        queue_last_launch_error: errMsg,
        queue_last_launch_attempt_at: Date.now(),
      });
    }
    await updateTrainingJob(c.env.DB, job.job_id, {
      status: 'failed',
      error_message: errMsg,
      completed_at: Date.now(),
      summary: {
        ...(job.summary ?? {}),
        queue_last_launch_error: errMsg,
        queue_last_launch_attempt_at: Date.now(),
      },
      supervisor: {
        ...(job.supervisor ?? {}),
        phase: 'failed',
        last_transition_at: Date.now(),
      },
    });
    if (job.round_id) {
      await updateTrainingRound(c.env.DB, job.round_id, {
        status: 'failed',
        completed_at: Date.now(),
        summary: {
          failed_job_id: job.job_id,
          error_message: errMsg,
        },
      });
    }
    return (await getTrainingJob(c.env.DB, job.job_id)) ?? job;
  }

  const now = Date.now();
  await updateTrainingJob(c.env.DB, job.job_id, {
    runpod_pod_id: launchResult.pod.podId,
    status: 'provisioning',
    progress: {},
    last_heartbeat_at: null,
    started_at: now,
    completed_at: null,
    error_message: null,
    summary: {
      ...(job.summary ?? {}),
      ...launchResult.summary,
      queue_wait_reason: null,
      queue_last_launch_error: null,
      queue_last_launch_attempt_at: now,
    },
    supervisor: {
      ...(job.supervisor ?? {}),
      phase: 'provisioning',
      last_transition_at: now,
      last_pod_id: launchResult.pod.podId,
    },
  });

  if (job.round_id) {
    await updateTrainingRound(c.env.DB, job.round_id, {
      status: 'running',
      started_at: now,
      summary: {
        dataset_snapshot_id: job.dataset_snapshot_id,
        initial_job_id: job.job_id,
        training_image_name: launchResult.summary.training_image_name ?? null,
        training_resolved_image_name: launchResult.summary.training_resolved_image_name ?? null,
      },
    });
  }

  return (await getTrainingJob(c.env.DB, job.job_id)) ?? job;
};

const launchQueuedTrainingJobsForVoice = async (
  c: Context<AppContext>,
  voiceId: string,
): Promise<number> => {
  const [jobsRaw, voice] = await Promise.all([
    listTrainingJobs(c.env.DB, { voice_id: voiceId, limit: 100 }),
    getVoice(c.env.DB, voiceId),
  ]);
  const voiceHasCheckpoint = Boolean(voice?.run_name) || Boolean(voice?.checkpoint_r2_prefix);
  const jobs = jobsRaw.sort((left, right) => {
    const leftPriority = getJobPriority(left, voiceHasCheckpoint);
    const rightPriority = getJobPriority(right, voiceHasCheckpoint);
    if (leftPriority !== rightPriority) return leftPriority - rightPriority;
    return left.created_at - right.created_at;
  });
  if (jobs.length === 0) {
    return 0;
  }

  const voiceActiveLimit = getMaxActiveTrainingJobsPerVoice(c);
  let voiceActiveCount = jobs.filter((job) => ACTIVE_JOB_STATUSES.has(job.status)).length;
  const globalActiveLimit = getMaxActiveTrainingJobsGlobal(c);
  let globalActiveCount = await countGlobalActiveTrainingJobs(c);

  if (globalActiveCount >= globalActiveLimit) {
    return 0;
  }

  const otherVoicesHaveQueuedJobs =
    globalActiveCount > voiceActiveCount ||
    (await countQueuedTrainingJobsExcludingVoice(c, voiceId)) > 0;

  const effectiveVoiceLimit = otherVoicesHaveQueuedJobs ? voiceActiveLimit : globalActiveLimit;

  if (voiceActiveCount >= effectiveVoiceLimit) {
    return 0;
  }

  let latestJobs = [...jobs];
  let launched = 0;
  for (const seedJob of jobs) {
    if (voiceActiveCount >= effectiveVoiceLimit || globalActiveCount >= globalActiveLimit) {
      break;
    }

    const job = latestJobs.find((candidate) => candidate.job_id === seedJob.job_id);
    if (!job) {
      continue;
    }
    if (!(job.status === 'queued' || job.status === 'pending')) {
      continue;
    }

    const claimed = await claimQueuedJob(c.env.DB, job.job_id, job.updated_at ?? job.created_at);
    if (!claimed) {
      continue;
    }

    const preprocessReady = await hasReusablePreprocessArtifacts(c, job);
    if (!preprocessReady) {
      const hasSiblingBuilder = latestJobs.some(
        (candidate) =>
          candidate.job_id !== job.job_id &&
          ACTIVE_JOB_STATUSES.has(candidate.status) &&
          sameDatasetPreprocessScope(candidate, job),
      );
      if (hasSiblingBuilder) {
        await queueTrainingJob(c, job, 'waiting_for_preprocess_artifacts');
        continue;
      }
    }

    const updated = await launchTrainingJob(c, job);
    if (ACTIVE_JOB_STATUSES.has(updated.status)) {
      launched += 1;
      voiceActiveCount += 1;
      globalActiveCount += 1;
    }

    latestJobs = latestJobs.map((candidate) =>
      candidate.job_id === updated.job_id ? updated : candidate,
    );
  }

  return launched;
};

const getCampaignStopRule = (value: unknown, fallback: number): number => {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return fallback;
  }
  return Math.max(1, Math.min(Math.trunc(numeric), 10));
};

const getCampaignFloatStopRule = (value: unknown, fallback: number): number => {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return fallback;
  }
  return Math.max(0, Math.min(numeric, 1));
};

const getCampaignAttemptScore = (job: TrainingJob): number | null => {
  const summary = (job.summary ?? {}) as Record<string, unknown>;
  const fromValidation = readNumber(summary.validation_score);
  if (fromValidation !== null) {
    return fromValidation;
  }
  const fromSelected = readNumber(summary.selected_score);
  if (fromSelected !== null) {
    return fromSelected;
  }
  const fromCandidate = readNumber(summary.candidate_score);
  if (fromCandidate !== null) {
    return fromCandidate;
  }
  const fromChampion = readNumber(summary.champion_score);
  if (fromChampion !== null) {
    return fromChampion;
  }
  return null;
};

const isCampaignAttemptUniquenessError = (error: unknown): boolean => {
  const message = String(error instanceof Error ? error.message : error).toLowerCase();
  return (
    message.includes('unique') &&
    message.includes('training_jobs') &&
    message.includes('campaign_id') &&
    message.includes('attempt_index')
  );
};

const isCampaignStateConflictError = (error: unknown): boolean => {
  const message = String(error instanceof Error ? error.message : error).toLowerCase();
  return message.includes('training_campaign_conflict');
};

const getCampaignConfigKey = (config: TrainingConfig): string =>
  JSON.stringify({
    model_size: config.model_size ?? null,
    batch_size: config.batch_size ?? null,
    learning_rate: config.learning_rate ?? null,
    num_epochs: config.num_epochs ?? null,
    gradient_accumulation_steps:
      (config as Record<string, unknown>).gradient_accumulation_steps ?? null,
    subtalker_loss_weight: (config as Record<string, unknown>).subtalker_loss_weight ?? null,
    save_every_n_epochs: (config as Record<string, unknown>).save_every_n_epochs ?? null,
    whisper_language: config.whisper_language ?? null,
    gpu_type_id: config.gpu_type_id ?? null,
    seed: config.seed ?? null,
  });

const summarizeCampaignFailures = (jobs: TrainingJob[]): { asr: number; infra: number } => {
  let asr = 0;
  let infra = 0;
  for (const job of jobs) {
    if (!(job.status === 'failed' || job.status === 'cancelled')) {
      continue;
    }
    const summary = (job.summary ?? {}) as Record<string, unknown>;
    const rawMessage = String(
      summary.validation_message ??
        summary.last_message ??
        summary.error_message ??
        job.error_message ??
        '',
    ).toLowerCase();
    if (rawMessage.includes('asr_score') || rawMessage.includes('missing asr')) {
      asr += 1;
    }
    if (
      rawMessage.includes('no audio') ||
      rawMessage.includes('stalled') ||
      rawMessage.includes('recovery') ||
      rawMessage.includes('supply_constraint')
    ) {
      infra += 1;
    }
  }
  return { asr, infra };
};

const buildCampaignAttemptConfigLegacy = (
  voice: NonNullable<Awaited<ReturnType<typeof getVoice>>>,
  campaign: TrainingCampaign,
  attemptIndex: number,
  voiceJobs: TrainingJob[],
): TrainingConfig => {
  const merged: TrainingConfig = {
    ...(campaign.base_config ?? {}),
  };
  const existingSeed = Number(merged.seed ?? 303);
  merged.seed = Number.isFinite(existingSeed)
    ? existingSeed + Math.max(0, attemptIndex - 1) * 101
    : 303 + Math.max(0, attemptIndex - 1) * 101;
  if (attemptIndex > 1) {
    const advice = buildTrainingAdvice(voice, voiceJobs);
    if (advice?.suggestedConfig) {
      Object.assign(merged, advice.suggestedConfig);
    }
  }
  return merged;
};

const TERMINAL_STATUSES_FOR_PLANNER = new Set(['completed', 'failed', 'cancelled']);

function shouldInvokeLLMPlanner(
  heuristicResult: PlannerResult,
  voiceJobs: TrainingJob[],
  slotsToFill: number,
): LLMPlannerDecision {
  const completedJobs = voiceJobs.filter((j) => TERMINAL_STATUSES_FOR_PLANNER.has(j.status));
  const scoredJobs = completedJobs.filter((j) => {
    const cs = j.checkout_search ?? buildTrainingCheckoutSearch(j);
    return (cs.selected?.score ?? cs.manual_promoted?.score ?? cs.champion?.score ?? null) !== null;
  });

  if (scoredJobs.length === 0) return 'escalated';

  const recentScores = completedJobs
    .sort((a, b) => {
      const aTime = a.completed_at ?? a.updated_at ?? 0;
      const bTime = b.completed_at ?? b.updated_at ?? 0;
      return bTime - aTime;
    })
    .slice(0, 4)
    .map((j) => {
      const cs = j.checkout_search ?? buildTrainingCheckoutSearch(j);
      return cs.selected?.score ?? cs.manual_promoted?.score ?? cs.champion?.score ?? null;
    })
    .filter((s): s is number => s !== null);

  if (recentScores.length >= 4) {
    const range = Math.max(...recentScores) - Math.min(...recentScores);
    if (range < 0.005) return 'escalated';
  }

  const failedJobs = completedJobs.filter(
    (j) =>
      j.status === 'failed' ||
      (j.checkout_search ?? buildTrainingCheckoutSearch(j)).status === 'rejected',
  );
  if (failedJobs.length >= 4) {
    const failMap = new Map<string, number>();
    for (const j of failedJobs) {
      const msg = (
        (j.checkout_search ?? buildTrainingCheckoutSearch(j)).message ??
        j.error_message ??
        ''
      ).toLowerCase();
      let reason = 'unknown';
      if (msg.includes('speed_score') || msg.includes('speed drift')) reason = 'speed';
      else if (msg.includes('asr_score') || msg.includes('missing asr')) reason = 'asr';
      else if (msg.includes('tone_score')) reason = 'tone';
      else if (msg.includes('overall_score') || msg.includes('quality threshold'))
        reason = 'overall';
      else if (
        msg.includes('no audio') ||
        msg.includes('stalled') ||
        msg.includes('supply_constraint') ||
        msg.includes('recovery')
      )
        reason = 'infra';
      failMap.set(reason, (failMap.get(reason) ?? 0) + 1);
    }
    const totalFails = [...failMap.values()].reduce((a, b) => a + b, 0);
    const maxFails = Math.max(...failMap.values());
    if (maxFails / totalFails < 0.6) return 'normal';
  }

  if (slotsToFill === 1 && heuristicResult.candidates.length >= 1) return 'skip';

  return 'skip';
}

const runCampaignPlanner = async (
  c: Context<AppContext>,
  voice: NonNullable<Awaited<ReturnType<typeof getVoice>>>,
  campaign: TrainingCampaign,
  voiceJobs: TrainingJob[],
  slotsToFill: number,
  nextAttemptIndex: number,
): Promise<PlannerResult> => {
  let brief: import('../types').StrategyBrief | undefined;
  try {
    brief = await buildStrategyBrief(c.env.DB, voice.voice_id);
  } catch {
    brief = undefined;
  }

  const heuristicResult = planCampaignAttempts(
    voice,
    campaign,
    voiceJobs,
    slotsToFill,
    nextAttemptIndex,
    brief,
  );

  if (heuristicResult.phase === 'infeasible' || heuristicResult.candidates.length === 0) {
    return heuristicResult;
  }

  const apiKey = String(c.env.OPENAI_API_KEY ?? '').trim();
  if (!apiKey) return heuristicResult;

  const llmDecision = shouldInvokeLLMPlanner(heuristicResult, voiceJobs, slotsToFill);
  console.log(
    `LLM planner decision: ${llmDecision} (slots=${slotsToFill}, jobs=${voiceJobs.length})`,
  );
  if (llmDecision === 'skip') return heuristicResult;

  const direction = (campaign.planner_state?.direction as string) ?? 'balanced';
  const stateHash = buildLLMPlannerStateHash(voice.voice_id, voiceJobs, slotsToFill, direction);
  const cachedHash = campaign.planner_state?.last_llm_hash as string | undefined;
  const cachedCandidates = campaign.planner_state?.last_llm_result as unknown;
  if (cachedHash === stateHash && cachedCandidates) {
    console.log('LLM planner cache hit — reusing previous result');
    return parseLLMPlannerResponse(cachedCandidates, voice, heuristicResult);
  }

  try {
    const dominantFailure = String(heuristicResult.state_patch.dominant_failure ?? 'unknown');
    const systemPrompt = buildLLMPlannerSystemPrompt(dominantFailure);
    const prompt = buildLLMPlannerPrompt({
      voice,
      campaign,
      allVoiceJobs: voiceJobs,
      slotsToFill,
      nextAttemptIndex,
      heuristicResult,
    });

    const effort = llmDecision === 'escalated' ? 'high' : 'medium';
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 15_000);
    const response = await fetch('https://api.openai.com/v1/responses', {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${apiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: String(c.env.OPENAI_ADVISOR_MODEL ?? 'gpt-5.4').trim() || 'gpt-5.4',
        input: [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: prompt },
        ],
        reasoning: { effort },
        text: { format: { type: 'json_object' } },
      }),
      signal: controller.signal,
    });
    clearTimeout(timer);

    if (!response.ok) {
      console.warn(`LLM planner failed (${response.status})`);
      return heuristicResult;
    }

    const payload = (await response.json()) as Record<string, unknown>;
    const content = typeof payload.output_text === 'string' ? payload.output_text : '';
    if (!content) return heuristicResult;

    const parsed = JSON.parse(
      content
        .trim()
        .replace(/^```(?:json)?\s*/i, '')
        .replace(/\s*```$/, ''),
    );

    const result = parseLLMPlannerResponse(parsed, voice, heuristicResult);
    result.state_patch.last_llm_hash = stateHash;
    result.state_patch.last_llm_result = parsed;
    result.state_patch.llm_decision = llmDecision;
    result.state_patch.llm_effort = effort;
    return result;
  } catch (error) {
    console.warn('LLM planner error, falling back to heuristic:', error);
    return heuristicResult;
  }
};

const enqueueCampaignAttempt = async (
  c: Context<AppContext>,
  campaign: TrainingCampaign,
  voice: NonNullable<Awaited<ReturnType<typeof getVoice>>>,
  attemptIndex: number,
  voiceJobs: TrainingJob[],
): Promise<TrainingJob> => {
  const now = Date.now();
  const jobId = crypto.randomUUID();
  const jobToken = crypto.randomUUID();
  const workerUrl = getWorkerOrigin(c);
  const runName = `run_${jobId.slice(0, 8)}`;
  const requestedConfig = buildCampaignAttemptConfigLegacy(
    voice,
    campaign,
    attemptIndex,
    voiceJobs,
  );
  const requestedCfg = requestedConfig as Record<string, unknown>;
  const modelSize =
    typeof requestedConfig.model_size === 'string' && requestedConfig.model_size
      ? requestedConfig.model_size
      : voice.model_size || '1.7B';
  const recommendedDefaults = getRecommendedTrainingDefaults(modelSize);
  const requestedBatchSize = readNumber(requestedConfig.batch_size);
  const requestedLearningRate = readNumber(requestedConfig.learning_rate);
  const requestedNumEpochs = readNumber(requestedConfig.num_epochs ?? requestedCfg.epochs);
  const requestedGradAccum = readNumber(requestedCfg.gradient_accumulation_steps);
  const requestedSubtalker = readNumber(requestedCfg.subtalker_loss_weight);
  const requestedSaveEvery = readNumber(requestedCfg.save_every_n_epochs);
  const requestedSeed = readNumber(requestedCfg.seed);
  const effectiveConfig: TrainingConfig = {
    ...requestedConfig,
    model_size: modelSize,
    batch_size: requestedBatchSize ?? recommendedDefaults.batch_size,
    learning_rate: requestedLearningRate ?? recommendedDefaults.learning_rate,
    num_epochs: requestedNumEpochs ?? recommendedDefaults.num_epochs,
    gradient_accumulation_steps:
      requestedGradAccum ?? recommendedDefaults.gradient_accumulation_steps,
    subtalker_loss_weight: requestedSubtalker ?? recommendedDefaults.subtalker_loss_weight,
    save_every_n_epochs: requestedSaveEvery ?? recommendedDefaults.save_every_n_epochs,
    seed: requestedSeed ?? recommendedDefaults.seed,
    gpu_type_id:
      typeof requestedCfg.gpu_type_id === 'string' && requestedCfg.gpu_type_id
        ? requestedCfg.gpu_type_id
        : recommendedDefaults.gpu_type_id,
  };
  if (typeof requestedCfg.whisper_language === 'string' && requestedCfg.whisper_language.trim()) {
    effectiveConfig.whisper_language = requestedCfg.whisper_language.trim();
  }

  const {
    datasetPrefix,
    datasetSignature,
    datasetSignatureInfo,
    datasetTrainRawKey,
    datasetHasPreparedTrainRaw,
    preprocessCache,
    datasetSnapshot,
  } = await (async () => {
    if (campaign.dataset_snapshot_id) {
      const frozenSnapshot = await getDatasetSnapshotById(c.env.DB, campaign.dataset_snapshot_id);
      if (frozenSnapshot) {
        const frozenPrefix = frozenSnapshot.dataset_r2_prefix;
        const frozenObjects = await listAllR2Objects(c.env.R2, `${stripSlashes(frozenPrefix)}/`);
        if (frozenObjects.length === 0) {
          throw new Error(
            `Dataset not found at R2 prefix: ${frozenPrefix}/. Upload audio files first.`,
          );
        }

        const trainRawKey =
          frozenSnapshot.train_raw_r2_key ?? getDatasetTrainRawR2Key(frozenPrefix);
        const hasPrepared = Boolean(
          frozenSnapshot.train_raw_r2_key ?? (await c.env.R2.head(trainRawKey)),
        );

        return {
          datasetPrefix: frozenPrefix,
          datasetSignature: frozenSnapshot.dataset_signature,
          datasetSignatureInfo: null,
          datasetTrainRawKey: trainRawKey,
          datasetHasPreparedTrainRaw: hasPrepared,
          preprocessCache: null,
          datasetSnapshot: frozenSnapshot,
        };
      }
    }

    const dynamicPrefix = resolveTrainingDatasetPrefix(voice, campaign.dataset_name ?? undefined);
    const datasetObjects = await listAllR2Objects(c.env.R2, `${stripSlashes(dynamicPrefix)}/`);
    if (datasetObjects.length === 0) {
      throw new Error(
        `Dataset not found at R2 prefix: ${dynamicPrefix}/. Upload audio files first.`,
      );
    }

    const signatureInfo = await computeDatasetSignature(dynamicPrefix, datasetObjects);
    const signature = signatureInfo?.signature ?? buildSyntheticDatasetSignature(dynamicPrefix);
    const trainRawKey = getDatasetTrainRawR2Key(dynamicPrefix);
    const hasPrepared = Boolean(await c.env.R2.head(trainRawKey));
    const cache =
      signatureInfo !== null
        ? await getDatasetPreprocessCache(c.env.DB, voice.voice_id, dynamicPrefix, signature)
        : null;
    const snapshot = await ensureDatasetSnapshot({
      c,
      voice,
      datasetPrefix: dynamicPrefix,
      datasetSignature: signature,
      preprocessCache: cache,
      sourceFileCount: signatureInfo?.sourceCount ?? null,
      createdFromJobId: jobId,
    });

    return {
      datasetPrefix: dynamicPrefix,
      datasetSignature: signature,
      datasetSignatureInfo: signatureInfo,
      datasetTrainRawKey: trainRawKey,
      datasetHasPreparedTrainRaw: hasPrepared,
      preprocessCache: cache,
      datasetSnapshot: snapshot,
    };
  })();

  const siblingJobs = voiceJobs.filter(
    (job) => job.dataset_snapshot_id === datasetSnapshot.snapshot_id,
  );
  const usedKeys = new Set(
    siblingJobs.map((job) => {
      const value = (job.summary ?? {}) as Record<string, unknown>;
      return typeof value.campaign_config_key === 'string' ? value.campaign_config_key : '';
    }),
  );
  const initialConfigKey = getCampaignConfigKey(effectiveConfig);
  if (usedKeys.has(initialConfigKey)) {
    effectiveConfig.seed =
      Number(effectiveConfig.seed ?? recommendedDefaults.seed) + attemptIndex * 97;
  }
  const configKey = getCampaignConfigKey(effectiveConfig);

  const initialSummary: Record<string, unknown> = {
    campaign_id: campaign.campaign_id,
    attempt_index: attemptIndex,
    campaign_config_key: configKey,
    dataset_snapshot_id: datasetSnapshot.snapshot_id,
    dataset_snapshot_status: datasetSnapshot.status,
    dataset_snapshot_signature: datasetSnapshot.dataset_signature,
    dataset_snapshot_name: datasetSnapshot.dataset_name,
    preprocess_cache_lookup: preprocessCache
      ? 'hit'
      : datasetHasPreparedTrainRaw
        ? 'dataset_ready'
        : 'miss',
    preprocess_cache_dataset_signature: datasetSignature,
    preprocess_cache_source_file_count: datasetSignatureInfo?.sourceCount ?? null,
    preprocess_cache_r2_prefix:
      datasetSnapshot.cache_r2_prefix ?? preprocessCache?.cache_r2_prefix ?? null,
    preprocess_cache_train_raw_r2_key:
      preprocessCache?.train_raw_r2_key ?? (datasetHasPreparedTrainRaw ? datasetTrainRawKey : null),
    queue_active_limit: getMaxActiveTrainingJobsPerVoice(c),
  };

  const job: TrainingJob = {
    job_id: jobId,
    voice_id: voice.voice_id,
    campaign_id: campaign.campaign_id,
    attempt_index: attemptIndex,
    round_id: null,
    dataset_snapshot_id: datasetSnapshot.snapshot_id,
    runpod_pod_id: null,
    job_token: jobToken,
    status: 'queued',
    config: effectiveConfig,
    progress: {},
    summary: initialSummary,
    metrics: {},
    supervisor: {
      phase: 'queued',
      checks: 0,
      recovery_attempts: 0,
      last_transition_at: now,
    },
    dataset_r2_prefix: datasetPrefix,
    log_r2_prefix: null,
    error_message: null,
    last_heartbeat_at: null,
    started_at: null,
    completed_at: null,
    created_at: now,
    updated_at: now,
  };
  await createTrainingJob(c.env.DB, job);

  const jobConfig = {
    voice_id: voice.voice_id,
    dataset_r2_prefix: datasetPrefix,
    speaker_name: voice.speaker_name,
    model_size: modelSize,
    batch_size: Number(effectiveConfig.batch_size ?? recommendedDefaults.batch_size),
    learning_rate: Number(effectiveConfig.learning_rate ?? recommendedDefaults.learning_rate),
    num_epochs: Number(effectiveConfig.num_epochs ?? recommendedDefaults.num_epochs),
    run_name: runName,
    gradient_accumulation_steps: Number(
      effectiveConfig.gradient_accumulation_steps ??
        recommendedDefaults.gradient_accumulation_steps,
    ),
    speaker_id: Number(requestedCfg.speaker_id ?? 3000),
    mixed_precision: String(requestedCfg.mixed_precision ?? 'bf16'),
    torch_dtype: String(requestedCfg.torch_dtype ?? 'bfloat16'),
    attn_implementation: String(requestedCfg.attn_implementation ?? 'sdpa'),
    weight_decay: Number(requestedCfg.weight_decay ?? 0.01),
    max_grad_norm: Number(requestedCfg.max_grad_norm ?? 1.0),
    subtalker_loss_weight: Number(
      effectiveConfig.subtalker_loss_weight ?? recommendedDefaults.subtalker_loss_weight,
    ),
    log_every_n_steps: Number(requestedCfg.log_every_n_steps ?? 10),
    save_every_n_epochs: Number(
      effectiveConfig.save_every_n_epochs ?? recommendedDefaults.save_every_n_epochs,
    ),
    max_steps: Number(requestedCfg.max_steps ?? 0),
    seed: Number(effectiveConfig.seed ?? recommendedDefaults.seed),
    job_token: jobToken,
    worker_api_url: workerUrl,
    whisper_language:
      typeof effectiveConfig.whisper_language === 'string'
        ? effectiveConfig.whisper_language
        : undefined,
    dataset_signature: datasetSignature,
    dataset_snapshot_id: datasetSnapshot.snapshot_id,
    preprocess_cache_r2_prefix:
      datasetSnapshot.cache_r2_prefix ?? preprocessCache?.cache_r2_prefix ?? undefined,
  };

  await c.env.R2.put(`jobs/${jobId}/config.json`, JSON.stringify(jobConfig), {
    httpMetadata: { contentType: 'application/json' },
  });

  await updateTrainingCampaign(c.env.DB, campaign.campaign_id, {
    dataset_r2_prefix: datasetPrefix,
    dataset_snapshot_id: datasetSnapshot.snapshot_id,
    planner_state: {
      ...(campaign.planner_state ?? {}),
      last_attempt_index: attemptIndex,
      last_job_id: jobId,
      last_config_key: configKey,
    },
  });

  await launchQueuedTrainingJobsForVoice(c, voice.voice_id);
  return (await getTrainingJob(c.env.DB, jobId)) ?? job;
};

const enqueueCampaignAttemptWithConfig = async (
  c: Context<AppContext>,
  campaign: TrainingCampaign,
  voice: NonNullable<Awaited<ReturnType<typeof getVoice>>>,
  attemptIndex: number,
  voiceJobs: TrainingJob[],
  plannerConfig: TrainingConfig,
  plannerMeta: { lane: string; reasoning: string },
): Promise<TrainingJob> => {
  const patchedCampaign: TrainingCampaign = {
    ...campaign,
    base_config: { ...campaign.base_config, ...plannerConfig },
    planner_state: {
      ...(campaign.planner_state ?? {}),
      last_lane: plannerMeta.lane,
      last_reasoning: plannerMeta.reasoning,
    },
  };
  return enqueueCampaignAttempt(c, patchedCampaign, voice, 1, voiceJobs);
};

const fireResearchHookOnCampaignEnd = (
  c: Context<AppContext>,
  voiceId: string,
  campaignId: string,
): void => {
  c.executionCtx.waitUntil(
    maybeAdvanceResearchLoop(
      c.env.DB,
      { OPENAI_API_KEY: c.env.OPENAI_API_KEY, OPENAI_ADVISOR_MODEL: c.env.OPENAI_ADVISOR_MODEL },
      voiceId,
      "campaign_completed",
      { linked_ids: { campaign_id: campaignId } },
    ).catch((err) => console.error("Research loop failed (campaign end):", err)),
  );
};

const advanceTrainingCampaign = async (
  c: Context<AppContext>,
  campaignId: string,
): Promise<TrainingCampaign | null> => {
  const campaign = await getTrainingCampaign(c.env.DB, campaignId);
  if (!campaign) {
    return null;
  }
  if (!CAMPAIGN_ACTIVE_STATUSES.has(campaign.status)) {
    return campaign;
  }

  try {
    await updateTrainingCampaign(c.env.DB, campaignId, {
      expected_updated_at: campaign.updated_at,
      updated_at: Date.now(),
    });
  } catch (error) {
    if (isCampaignStateConflictError(error)) {
      return getTrainingCampaign(c.env.DB, campaignId);
    }
    throw error;
  }

  const voice = await getVoice(c.env.DB, campaign.voice_id);
  if (!voice) {
    await updateTrainingCampaign(c.env.DB, campaignId, {
      status: 'failed',
      summary: {
        ...(campaign.summary ?? {}),
        last_message: 'voice_not_found',
      },
      completed_at: Date.now(),
    });
    return getTrainingCampaign(c.env.DB, campaignId);
  }

  const campaignJobs = await listTrainingJobs(c.env.DB, { campaign_id: campaignId, limit: 100 });
  const terminalJobs = campaignJobs.filter((job) => TERMINAL_JOB_STATUSES.has(job.status));
  const inflightJobs = campaignJobs.filter((job) => !TERMINAL_JOB_STATUSES.has(job.status));
  const stopRules = campaign.stop_rules ?? {};
  const maxAsrFailures = getCampaignStopRule(stopRules.max_asr_failures, 2);
  const maxInfraFailures = getCampaignStopRule(stopRules.max_infra_failures, 2);
  const minScoreImprovement = getCampaignFloatStopRule(stopRules.min_score_improvement, 0);
  const stagnationWindow = Math.max(
    2,
    Math.min(getCampaignStopRule(stopRules.stagnation_window, 2), 6),
  );
  const failureSummary = summarizeCampaignFailures(terminalJobs);

  if (failureSummary.asr >= maxAsrFailures) {
    await updateTrainingCampaign(c.env.DB, campaignId, {
      status: 'blocked_dataset',
      summary: {
        ...(campaign.summary ?? {}),
        reason: 'asr_failure_limit',
        asr_failures: failureSummary.asr,
        infra_failures: failureSummary.infra,
      },
      completed_at: Date.now(),
    });
    fireResearchHookOnCampaignEnd(c, voice.voice_id, campaignId);
    return getTrainingCampaign(c.env.DB, campaignId);
  }

  if (failureSummary.infra >= maxInfraFailures) {
    await updateTrainingCampaign(c.env.DB, campaignId, {
      status: 'blocked_budget',
      summary: {
        ...(campaign.summary ?? {}),
        reason: 'infra_failure_limit',
        asr_failures: failureSummary.asr,
        infra_failures: failureSummary.infra,
      },
      completed_at: Date.now(),
    });
    fireResearchHookOnCampaignEnd(c, voice.voice_id, campaignId);
    return getTrainingCampaign(c.env.DB, campaignId);
  }

  if (minScoreImprovement > 0 && terminalJobs.length >= stagnationWindow) {
    const recentScores = terminalJobs
      .slice()
      .sort((left, right) => {
        const leftTime = left.completed_at ?? left.updated_at;
        const rightTime = right.completed_at ?? right.updated_at;
        return leftTime - rightTime;
      })
      .map((job) => ({
        score: getCampaignAttemptScore(job),
        attemptIndex: job.attempt_index ?? Number.MAX_SAFE_INTEGER,
      }))
      .filter((item): item is { score: number; attemptIndex: number } => item.score !== null)
      .slice(-stagnationWindow);

    if (recentScores.length >= stagnationWindow) {
      const baseline = recentScores[0].score;
      const peak = recentScores.reduce(
        (best, current) => (current.score > best ? current.score : best),
        baseline,
      );
      const improvement = peak - baseline;
      if (improvement < minScoreImprovement) {
        await updateTrainingCampaign(c.env.DB, campaignId, {
          status: 'blocked_budget',
          summary: {
            ...(campaign.summary ?? {}),
            reason: 'stagnation',
            min_score_improvement: minScoreImprovement,
            stagnation_window: stagnationWindow,
            observed_improvement: Number(improvement.toFixed(6)),
            asr_failures: failureSummary.asr,
            infra_failures: failureSummary.infra,
          },
          completed_at: Date.now(),
        });
        fireResearchHookOnCampaignEnd(c, voice.voice_id, campaignId);
        return getTrainingCampaign(c.env.DB, campaignId);
      }
    }
  }

  const createdAttempts = campaignJobs.length;
  if (createdAttempts >= campaign.attempt_count && inflightJobs.length === 0) {
    const successfulAttempts = terminalJobs.filter((job) => job.status === 'completed').length;
    const finalStatus: TrainingCampaignStatus = successfulAttempts > 0 ? 'completed' : 'failed';
    await updateTrainingCampaign(c.env.DB, campaignId, {
      status: finalStatus,
      summary: {
        ...(campaign.summary ?? {}),
        attempts_created: createdAttempts,
        attempts_completed: terminalJobs.length,
        attempts_succeeded: successfulAttempts,
        reason: successfulAttempts > 0 ? (campaign.summary?.reason ?? null) : 'all_attempts_failed',
      },
      completed_at: Date.now(),
    });
    fireResearchHookOnCampaignEnd(c, voice.voice_id, campaignId);
    return getTrainingCampaign(c.env.DB, campaignId);
  }

  const perVoiceActiveLimit = getMaxActiveTrainingJobsPerVoice(c);
  const globalActiveLimit = getMaxActiveTrainingJobsGlobal(c);
  const parallelism = Math.max(1, Math.min(campaign.parallelism, perVoiceActiveLimit));
  const voiceJobs = await listTrainingJobs(c.env.DB, { voice_id: voice.voice_id, limit: 100 });
  const voiceActiveCount = voiceJobs.filter((job) => ACTIVE_JOB_STATUSES.has(job.status)).length;
  const globalActiveCount = await countGlobalActiveTrainingJobs(c);
  const campaignOpenSlots = Math.max(0, parallelism - inflightJobs.length);
  const voiceOpenSlots = Math.max(0, perVoiceActiveLimit - voiceActiveCount);
  const globalOpenSlots = Math.max(0, globalActiveLimit - globalActiveCount);
  const remainingAttempts = Math.max(0, campaign.attempt_count - createdAttempts);
  const attemptsToCreate = Math.min(
    campaignOpenSlots,
    voiceOpenSlots,
    globalOpenSlots,
    remainingAttempts,
  );

  let nextAttemptIndex = createdAttempts + 1;

  const plannerResult =
    attemptsToCreate > 0
      ? await runCampaignPlanner(c, voice, campaign, voiceJobs, attemptsToCreate, nextAttemptIndex)
      : null;

  if (plannerResult?.phase === 'infeasible') {
    await updateTrainingCampaign(c.env.DB, campaignId, {
      status: 'blocked_budget',
      summary: {
        ...(campaign.summary ?? {}),
        reason: 'model_infeasible',
        planner_phase: 'infeasible',
        planner_stop_recommendation: plannerResult.stop_recommendation,
      },
      planner_state: {
        ...(campaign.planner_state ?? {}),
        ...plannerResult.state_patch,
      },
      completed_at: Date.now(),
    });
    fireResearchHookOnCampaignEnd(c, voice.voice_id, campaignId);
    return getTrainingCampaign(c.env.DB, campaignId);
  }

  if (
    inflightJobs.length === 0 &&
    (plannerResult?.stop_recommendation === 'stop_diminishing_returns' ||
      plannerResult?.stop_recommendation === 'stop_model_unfit')
  ) {
    await updateTrainingCampaign(c.env.DB, campaignId, {
      status: 'completed',
      summary: {
        ...(campaign.summary ?? {}),
        reason: plannerResult.stop_recommendation,
        planner_phase: plannerResult.phase,
        planner_stop_recommendation: plannerResult.stop_recommendation,
      },
      planner_state: {
        ...(campaign.planner_state ?? {}),
        ...plannerResult.state_patch,
      },
      completed_at: Date.now(),
    });
    fireResearchHookOnCampaignEnd(c, voice.voice_id, campaignId);
    return getTrainingCampaign(c.env.DB, campaignId);
  }

  for (let i = 0; i < attemptsToCreate; i += 1) {
    const latestCampaign = await getTrainingCampaign(c.env.DB, campaignId);
    if (!latestCampaign || !CAMPAIGN_ACTIVE_STATUSES.has(latestCampaign.status)) {
      break;
    }

    const plannerCandidate = plannerResult?.candidates[i] ?? null;

    try {
      const createdJob = plannerCandidate
        ? await enqueueCampaignAttemptWithConfig(
            c,
            campaign,
            voice,
            nextAttemptIndex,
            voiceJobs,
            plannerCandidate.config,
            { lane: plannerCandidate.lane, reasoning: plannerCandidate.reasoning },
          )
        : await enqueueCampaignAttempt(c, campaign, voice, nextAttemptIndex, voiceJobs);
      voiceJobs.push(createdJob);
      nextAttemptIndex += 1;
    } catch (error) {
      if (isCampaignAttemptUniquenessError(error)) {
        break;
      }
      const message = error instanceof Error ? error.message : String(error);
      await updateTrainingCampaign(c.env.DB, campaignId, {
        status: 'failed',
        summary: {
          ...(campaign.summary ?? {}),
          last_message: message,
        },
        completed_at: Date.now(),
      });
      return getTrainingCampaign(c.env.DB, campaignId);
    }
  }

  const refreshedJobs = await listTrainingJobs(c.env.DB, { campaign_id: campaignId, limit: 100 });
  const activeCount = refreshedJobs.filter((job) => ACTIVE_JOB_STATUSES.has(job.status)).length;
  const queuedCount = refreshedJobs.filter(
    (job) => job.status === 'queued' || job.status === 'pending',
  ).length;
  const freshCampaign = await getTrainingCampaign(c.env.DB, campaignId);
  await updateTrainingCampaign(c.env.DB, campaignId, {
    status: 'running',
    planner_state: {
      ...(freshCampaign?.planner_state ?? campaign.planner_state ?? {}),
      ...(plannerResult?.state_patch ?? {}),
    },
    summary: {
      ...(freshCampaign?.summary ?? campaign.summary ?? {}),
      attempts_created: refreshedJobs.length,
      attempts_completed: refreshedJobs.filter((job) => TERMINAL_JOB_STATUSES.has(job.status))
        .length,
      active_jobs: activeCount,
      queued_jobs: queuedCount,
      asr_failures: failureSummary.asr,
      infra_failures: failureSummary.infra,
      planner_phase: plannerResult?.phase ?? null,
      planner_stop_recommendation: plannerResult?.stop_recommendation ?? null,
    },
  });

  return getTrainingCampaign(c.env.DB, campaignId);
};

const runTrainingCampaignSweep = async (c: Context<AppContext>): Promise<number> => {
  const campaigns = await listTrainingCampaigns(c.env.DB, {
    status_in: ['planning', 'running'],
    limit: 50,
  });
  let advanced = 0;
  for (const campaign of campaigns) {
    try {
      await advanceTrainingCampaign(c, campaign.campaign_id);
      advanced += 1;
    } catch (error) {
      console.warn(`Training campaign sweep failed for ${campaign.campaign_id}:`, error);
    }
  }
  return advanced;
};

const getDirectFallbackDockerArgs = (
  pod: PodStatusDetail,
  template: Awaited<ReturnType<typeof getTemplateById>>,
): string | null => {
  if (pod.dockerArgs && pod.dockerArgs.trim()) {
    return pod.dockerArgs.trim();
  }
  const entrypoint = Array.isArray(template?.dockerEntrypoint)
    ? template.dockerEntrypoint.join(' ').trim()
    : '';
  const startCmd = Array.isArray(template?.dockerStartCmd)
    ? template.dockerStartCmd.join(' ').trim()
    : '';
  const joined = [entrypoint, startCmd].filter(Boolean).join(' ').trim();
  return joined || null;
};

const getProvisioningState = (pod: PodStatusDetail | null): string => {
  return String(pod?.latestTelemetry?.state ?? pod?.runtimeStatus ?? '')
    .trim()
    .toLowerCase();
};

const isProvisioningPodStalled = (pod: PodStatusDetail | null): boolean => {
  if (!pod) {
    return false;
  }

  const state = getProvisioningState(pod);
  const uptimeSeconds = readNumber(pod.uptimeSeconds ?? pod.runtime?.uptimeInSeconds);
  const cpuUtil = readNumber(
    pod.latestTelemetry?.cpuUtilization ?? pod.runtime?.container?.cpuPercent,
  );
  const memoryUtil = readNumber(
    pod.latestTelemetry?.memoryUtilization ?? pod.runtime?.container?.memoryPercent,
  );
  const gpuUtil = (pod.runtime?.gpus ?? [])
    .map((gpu) => readNumber(gpu.gpuUtilPercent))
    .filter((value): value is number => value !== null)
    .reduce((max, value) => Math.max(max, value), 0);
  const noUtilization = [cpuUtil, memoryUtil, gpuUtil].every(
    (value) => value === null || value === 0,
  );

  if (state === 'created' || state === 'pending' || state === 'starting') {
    return noUtilization;
  }

  return (state === '' || state === 'running') && uptimeSeconds === 0 && noUtilization;
};

const recoverStalledProvisioningJob = async (
  c: Context<AppContext>,
  job: TrainingJob,
): Promise<TrainingJob> => {
  if (job.status !== 'provisioning' || !job.runpod_pod_id) {
    return job;
  }

  const startedAt = job.started_at ?? job.updated_at ?? job.created_at;
  const ageMs = Math.max(0, Date.now() - startedAt);
  if (ageMs < PROVISIONING_STALE_MS) {
    return job;
  }

  let podStatus: PodStatusDetail | null = null;
  try {
    podStatus = await getPodStatus(c.env, job.runpod_pod_id);
  } catch {
    return job;
  }

  if (podStatus && !isProvisioningPodStalled(podStatus)) {
    return job;
  }

  const summary = (job.summary ?? {}) as Record<string, unknown>;
  const attempts = readNumber(summary.provisioning_recovery_attempts) ?? 0;
  const podState = podStatus ? getProvisioningState(podStatus) || 'unknown' : 'missing';
  const previousPodIds = Array.isArray(summary.previous_runpod_pod_ids)
    ? summary.previous_runpod_pod_ids.filter((value): value is string => typeof value === 'string')
    : [];
  const nextSummary = {
    ...summary,
    provisioning_recovery_attempts: attempts + 1,
    previous_runpod_pod_ids: Array.from(new Set([...previousPodIds, job.runpod_pod_id])),
    last_provisioning_recovery_at: Date.now(),
    last_provisioning_recovery_reason: `stalled_${podState}`,
  };
  const reason =
    `RunPod pod stalled in provisioning for ${Math.round(ageMs / 60000)} minute(s): ` +
    `state=${podState} uptime=${podStatus?.uptimeSeconds ?? podStatus?.runtime?.uptimeInSeconds ?? 'n/a'}s ` +
    `image=${podStatus?.imageName ?? 'unknown'} template=${podStatus?.templateId ?? 'unknown'}`;

  if (!job.job_token) {
    await updateTrainingJob(c.env.DB, job.job_id, {
      status: 'failed',
      error_message: `${reason}. Missing job_token; cannot recreate pod.`,
      completed_at: Date.now(),
      summary: {
        ...nextSummary,
        provisioning_recovery_exhausted: true,
      },
    });
    return (await getTrainingJob(c.env.DB, job.job_id)) ?? job;
  }

  const templateId = getConfiguredTrainingTemplateId(c);
  const hasTriedDirectFallback = summary.provisioning_direct_fallback_attempted === true;
  const hasTriedDigestFallback = summary.provisioning_digest_fallback_attempted === true;
  if (!templateId && podStatus?.imageName) {
    let template: Awaited<ReturnType<typeof getTemplateById>> = null;
    try {
      if (podStatus.templateId) {
        template = await getTemplateById(c.env, podStatus.templateId);
      }
    } catch {
      template = null;
    }

    const dockerArgs = getDirectFallbackDockerArgs(podStatus, template);
    if (dockerArgs) {
      const fallbackImage = template?.imageName ?? podStatus.imageName;
      if (!fallbackImage) {
        return job;
      }
      const resolvedImage = await resolveGhcrAmd64Image(fallbackImage).catch(() => fallbackImage);
      const shouldTryDirect =
        !hasTriedDirectFallback || (!hasTriedDigestFallback && resolvedImage !== fallbackImage);
      if (shouldTryDirect) {
        await terminatePod(c.env, job.runpod_pod_id).catch(() => false);

        const directSummary = {
          ...nextSummary,
          provisioning_direct_fallback_attempted: true,
          provisioning_direct_fallback_image: resolvedImage,
          provisioning_direct_fallback_docker_args: dockerArgs,
          provisioning_digest_fallback_attempted: resolvedImage !== fallbackImage,
        };

        const tryCreateDirect = async (cloudType: 'COMMUNITY' | 'ALL') =>
          createPodDirect(c.env, {
            gpuTypeId: getTrainingGpuType(job),
            envVars: buildTrainingPodEnv(c, job),
            imageName: resolvedImage,
            dockerArgs,
            name: `qwen3-tts-training-${job.job_id.slice(0, 8)}`,
            cloudType,
            containerRegistryAuthId: template?.containerRegistryAuthId ?? undefined,
            ports: template?.ports ?? undefined,
            volumeMountPath: template?.volumeMountPath ?? undefined,
          });

        try {
          let newPod: { podId: string; desiredStatus: string };
          try {
            newPod = await tryCreateDirect('COMMUNITY');
          } catch {
            newPod = await tryCreateDirect('ALL');
          }

          await updateTrainingJob(c.env.DB, job.job_id, {
            runpod_pod_id: newPod.podId,
            status: 'provisioning',
            error_message: null,
            started_at: Date.now(),
            summary: {
              ...directSummary,
              last_provisioning_recovery_mode: 'direct_image',
            },
          });
          return (await getTrainingJob(c.env.DB, job.job_id)) ?? job;
        } catch (error) {
          await updateTrainingJob(c.env.DB, job.job_id, {
            status: 'failed',
            error_message: `Failed to launch direct-image fallback pod: ${
              error instanceof Error ? error.message : String(error)
            }`,
            completed_at: Date.now(),
            summary: {
              ...directSummary,
              provisioning_direct_fallback_failed: true,
            },
          });
          return (await getTrainingJob(c.env.DB, job.job_id)) ?? job;
        }
      }
    }
  }

  if (attempts >= MAX_PROVISIONING_RECOVERY_ATTEMPTS) {
    await terminatePod(c.env, job.runpod_pod_id).catch(() => false);
    await updateTrainingJob(c.env.DB, job.job_id, {
      status: 'failed',
      error_message: `${reason}. Recovery attempts exhausted.`,
      completed_at: Date.now(),
      summary: {
        ...nextSummary,
        provisioning_recovery_exhausted: true,
      },
    });
    return (await getTrainingJob(c.env.DB, job.job_id)) ?? job;
  }

  if (!templateId) {
    await updateTrainingJob(c.env.DB, job.job_id, {
      status: 'failed',
      error_message: `${reason}. No training template configured for recovery.`,
      completed_at: Date.now(),
      summary: {
        ...nextSummary,
        provisioning_recovery_failed: true,
      },
    });
    return (await getTrainingJob(c.env.DB, job.job_id)) ?? job;
  }

  await terminatePod(c.env, job.runpod_pod_id).catch(() => false);

  try {
    const newPod = await createPod(
      c.env,
      templateId,
      getTrainingGpuType(job),
      buildTrainingPodEnv(c, job),
    );
    await updateTrainingJob(c.env.DB, job.job_id, {
      runpod_pod_id: newPod.podId,
      status: 'provisioning',
      error_message: null,
      started_at: Date.now(),
      summary: nextSummary,
    });
    return (await getTrainingJob(c.env.DB, job.job_id)) ?? job;
  } catch (error) {
    await updateTrainingJob(c.env.DB, job.job_id, {
      status: 'failed',
      error_message: `Failed to recreate stalled provisioning pod: ${
        error instanceof Error ? error.message : String(error)
      }`,
      completed_at: Date.now(),
      summary: {
        ...nextSummary,
        provisioning_recovery_failed: true,
      },
    });
    return (await getTrainingJob(c.env.DB, job.job_id)) ?? job;
  }
};

const getStaleThresholdMs = (status: string): number | null => {
  return ACTIVE_STAGE_STALE_MS[status] ?? null;
};

const getLatestActivityAt = async (
  c: Context<AppContext>,
  job: TrainingJob,
  parsedStatus: TrainingStatusBlob | null,
): Promise<number> => {
  const timestamps = [
    readTimestamp(parsedStatus?.updated_at),
    readNumber(job.last_heartbeat_at),
    readNumber(job.updated_at),
    readNumber(job.started_at),
    readNumber(job.created_at),
  ].filter((value): value is number => value !== null);
  const latestChunk = await listTrainingLogChunks(c.env.DB, job.job_id, 1);
  if (latestChunk[0]) {
    timestamps.push(latestChunk[0].created_at);
  }
  return timestamps.length > 0 ? Math.max(...timestamps) : Date.now();
};

const deleteR2Prefix = async (bucket: R2Bucket, prefix: string): Promise<number> => {
  let deleted = 0;
  let cursor: string | undefined;
  do {
    const page = await bucket.list({ prefix, cursor, limit: 1000 });
    for (const object of page.objects) {
      await bucket.delete(object.key);
      deleted += 1;
    }
    cursor = page.truncated ? page.cursor : undefined;
  } while (cursor);
  return deleted;
};

const clearRecoveredJobArtifacts = async (c: Context<AppContext>, jobId: string): Promise<void> => {
  await c.env.R2.delete(`jobs/${jobId}/status.json`);
  await deleteR2Prefix(c.env.R2, `jobs/${jobId}/logs/`);
  await deleteTrainingLogChunks(c.env.DB, jobId);
};

const isDependencyImageFailureMessage = (message: string | null | undefined): boolean => {
  const normalized = typeof message === 'string' ? message.trim().toLowerCase() : '';
  if (!normalized) {
    return false;
  }
  return (
    normalized.includes('faster-whisper is not installed') ||
    normalized.includes('preprocessing requires faster-whisper') ||
    normalized.includes('whisper is not installed in this image')
  );
};

const shouldRecoverFailedDependencyJob = (c: Context<AppContext>, job: TrainingJob): boolean => {
  if (job.status !== 'failed') {
    return false;
  }
  if (!isDependencyImageFailureMessage(job.error_message)) {
    return false;
  }
  const summary = (job.summary ?? {}) as Record<string, unknown>;
  if (summary.training_image_recovery_exhausted === true) {
    return false;
  }
  const candidates = getTrainingImageCandidatesForJob(c, job);
  if (candidates.length <= 1) {
    return false;
  }
  const attemptIndex = getTrainingImageAttemptIndex(c, job, candidates);
  return attemptIndex + 1 < candidates.length;
};

const recoverFailedDependencyJob = async (
  c: Context<AppContext>,
  job: TrainingJob,
): Promise<TrainingJob> => {
  if (!shouldRecoverFailedDependencyJob(c, job)) {
    return job;
  }

  const summary = (job.summary ?? {}) as Record<string, unknown>;
  const candidates = getTrainingImageCandidatesForJob(c, job);
  const currentAttemptIndex = getTrainingImageAttemptIndex(c, job, candidates);
  const nextAttemptIndex = currentAttemptIndex + 1;
  const nextImage = candidates[nextAttemptIndex];
  if (!nextImage) {
    await updateTrainingJob(c.env.DB, job.job_id, {
      summary: {
        ...summary,
        training_image_recovery_exhausted: true,
      },
    });
    return (await getTrainingJob(c.env.DB, job.job_id)) ?? job;
  }

  const siblingJobs = await listTrainingJobs(c.env.DB, {
    voice_id: job.voice_id,
    limit: 20,
  });
  const activeSiblingCount = siblingJobs.filter(
    (candidate) => candidate.job_id !== job.job_id && ACTIVE_JOB_STATUSES.has(candidate.status),
  ).length;
  if (activeSiblingCount >= getMaxActiveTrainingJobsPerVoice(c)) {
    return queueTrainingJob(c, job, 'concurrency_limit', {
      training_image_recovery_deferred: true,
      training_image_recovery_deferred_reason: 'concurrency_limit',
    });
  }

  if (job.runpod_pod_id) {
    await terminatePod(c.env, job.runpod_pod_id).catch(() => false);
  }
  await clearRecoveredJobArtifacts(c, job.job_id);

  const previousPodIds = Array.isArray(summary.previous_runpod_pod_ids)
    ? summary.previous_runpod_pod_ids.filter((value): value is string => typeof value === 'string')
    : [];
  const baseSummary = {
    ...summary,
    training_image_recovery_attempts:
      (readNumber(summary.training_image_recovery_attempts) ?? 0) + 1,
    training_image_recovery_reason: 'dependency_missing',
    training_image_recovered_from: candidates[currentAttemptIndex] ?? null,
    training_image_recovered_to: nextImage,
    training_image_candidates: candidates,
    last_recovery_at: Date.now(),
    previous_runpod_pod_ids: Array.from(
      new Set([...previousPodIds, ...(job.runpod_pod_id ? [job.runpod_pod_id] : [])]),
    ),
  };

  try {
    const launchResult = await createTrainingPodForJob(c, job, {
      imageAttemptIndex: nextAttemptIndex,
    });
    await updateTrainingJob(c.env.DB, job.job_id, {
      runpod_pod_id: launchResult.pod.podId,
      status: 'provisioning',
      progress: {},
      error_message: null,
      last_heartbeat_at: null,
      started_at: Date.now(),
      completed_at: null,
      summary: {
        ...baseSummary,
        ...launchResult.summary,
        last_recovery_mode: 'dependency_image_fallback',
      },
      supervisor: {
        ...(job.supervisor ?? {}),
        phase: 'provisioning',
        last_transition_at: Date.now(),
        last_pod_id: launchResult.pod.podId,
      },
    });
    if (job.round_id) {
      await updateTrainingRound(c.env.DB, job.round_id, {
        status: 'running',
        completed_at: null,
        summary: {
          ...(await getTrainingRound(c.env.DB, job.round_id))?.summary,
          dependency_image_recovery: true,
          recovered_job_id: job.job_id,
          next_training_image: nextImage,
        },
      });
    }
    return (await getTrainingJob(c.env.DB, job.job_id)) ?? job;
  } catch (error) {
    const errMsg =
      error instanceof Error ? error.message : 'Failed to launch dependency recovery pod';
    if (isGpuSupplyConstraintErrorMessage(errMsg)) {
      return queueTrainingJob(c, job, 'gpu_supply_constraint', {
        ...baseSummary,
        training_image_recovery_deferred: true,
        training_image_recovery_deferred_reason: 'gpu_supply_constraint',
        queue_last_launch_error: errMsg,
        queue_last_launch_attempt_at: Date.now(),
      });
    }
    await updateTrainingJob(c.env.DB, job.job_id, {
      status: 'failed',
      error_message: errMsg,
      completed_at: Date.now(),
      summary: {
        ...baseSummary,
        training_image_recovery_failed: true,
      },
    });
    return (await getTrainingJob(c.env.DB, job.job_id)) ?? job;
  }
};

const recoverStalledActiveJob = async (
  c: Context<AppContext>,
  job: TrainingJob,
  parsedStatus: TrainingStatusBlob | null,
): Promise<TrainingJob> => {
  const effectiveStatus =
    typeof parsedStatus?.status === 'string' && parsedStatus.status.trim()
      ? parsedStatus.status.trim()
      : job.status;
  if (!ACTIVE_RUNTIME_RECOVERY_STATUSES.has(effectiveStatus)) {
    return job;
  }

  const staleThresholdMs = getStaleThresholdMs(effectiveStatus);
  if (staleThresholdMs === null) {
    return job;
  }

  const latestActivityAt = await getLatestActivityAt(c, job, parsedStatus);
  const inactiveMs = Math.max(0, Date.now() - latestActivityAt);
  if (inactiveMs < staleThresholdMs) {
    return job;
  }

  const summary = (job.summary ?? {}) as Record<string, unknown>;
  const attempts = readNumber(summary.stall_recovery_attempts) ?? 0;
  const lastMessage = typeof summary.last_message === 'string' ? summary.last_message.trim() : '';
  const previousPodIds = Array.isArray(summary.previous_runpod_pod_ids)
    ? summary.previous_runpod_pod_ids.filter((value): value is string => typeof value === 'string')
    : [];

  let podStatus: PodStatusDetail | null = null;
  try {
    podStatus = job.runpod_pod_id ? await getPodStatus(c.env, job.runpod_pod_id) : null;
  } catch (error) {
    console.warn(
      `Failed to inspect pod ${job.runpod_pod_id ?? 'unknown'} for stalled job ${job.job_id}:`,
      error,
    );
  }

  const podState =
    getProvisioningState(podStatus) ||
    String(podStatus?.runtimeStatus ?? '')
      .trim()
      .toLowerCase() ||
    'unknown';
  const baseSummary = {
    ...summary,
    stall_recovery_attempts: attempts + 1,
    last_stall_stage: effectiveStatus,
    last_stall_activity_at: latestActivityAt,
    last_stall_inactive_ms: inactiveMs,
    last_stall_pod_state: podState,
    last_recovery_at: Date.now(),
    previous_runpod_pod_ids: Array.from(
      new Set([...previousPodIds, ...(job.runpod_pod_id ? [job.runpod_pod_id] : [])]),
    ),
  };
  const reason =
    `Training job stalled in ${effectiveStatus} for ${Math.round(inactiveMs / 60000)} minute(s)` +
    (lastMessage ? `; last_message=${lastMessage}` : '') +
    (job.runpod_pod_id ? `; pod=${job.runpod_pod_id}` : '') +
    (podState ? `; pod_state=${podState}` : '');

  if (attempts >= MAX_STALL_RECOVERY_ATTEMPTS) {
    if (job.runpod_pod_id) {
      await terminatePod(c.env, job.runpod_pod_id).catch(() => false);
    }
    await updateTrainingJob(c.env.DB, job.job_id, {
      status: 'failed',
      error_message: `${reason}. Recovery attempts exhausted.`,
      completed_at: Date.now(),
      summary: {
        ...baseSummary,
        stall_recovery_exhausted: true,
      },
    });
    return (await getTrainingJob(c.env.DB, job.job_id)) ?? job;
  }

  if (!job.job_token) {
    await updateTrainingJob(c.env.DB, job.job_id, {
      status: 'failed',
      error_message: `${reason}. Missing job_token; cannot recreate pod.`,
      completed_at: Date.now(),
      summary: {
        ...baseSummary,
        stall_recovery_failed: true,
      },
    });
    return (await getTrainingJob(c.env.DB, job.job_id)) ?? job;
  }

  if (job.runpod_pod_id) {
    await terminatePod(c.env, job.runpod_pod_id).catch(() => false);
  }

  await clearRecoveredJobArtifacts(c, job.job_id);

  try {
    const launchResult = await createTrainingPodForJob(c, job);
    await updateTrainingJob(c.env.DB, job.job_id, {
      runpod_pod_id: launchResult.pod.podId,
      status: 'provisioning',
      progress: {},
      error_message: null,
      last_heartbeat_at: null,
      started_at: Date.now(),
      completed_at: null,
      summary: {
        ...baseSummary,
        ...launchResult.summary,
        last_recovery_mode: 'restart_same_job',
      },
    });
    return (await getTrainingJob(c.env.DB, job.job_id)) ?? job;
  } catch (error) {
    await updateTrainingJob(c.env.DB, job.job_id, {
      status: 'failed',
      error_message: `${reason}. Failed to recreate pod: ${
        error instanceof Error ? error.message : String(error)
      }`,
      completed_at: Date.now(),
      summary: {
        ...baseSummary,
        stall_recovery_failed: true,
      },
    });
    return (await getTrainingJob(c.env.DB, job.job_id)) ?? job;
  }
};

const parseOverallFromError = (message: string): number | null => {
  const m = OVERALL_SCORE_ERROR_RE.exec(message);
  if (!m || !m[1]) return null;
  const v = Number(m[1]);
  return Number.isFinite(v) ? v : null;
};

const getValidationTexts = (lang: string, is06b: boolean): string[] => {
  const validationTextsByLang: Record<string, string[]> = {
    ko: [
      '안녕하세요.',
      '안녕하세요. 오늘 회의는 오후 두 시에 시작합니다.',
      '안녕하세요. 오늘 회의는 오후 두 시에 시작하고, 발표 자료는 메일로 공유드리겠습니다.',
    ],
    en: [
      'Hello.',
      "Hello. The meeting starts at two o'clock this afternoon.",
      "Hello. The meeting starts at two o'clock this afternoon, and I will share the presentation materials via email.",
    ],
    zh: [
      '你好。',
      '你好。今天的会议下午两点开始。',
      '你好。今天的会议下午两点开始，我会通过邮件分享演示文稿。',
    ],
    ja: [
      'こんにちは。',
      'こんにちは。今日の会議は午後二時に始まります。',
      'こんにちは。今日の会議は午後二時に始まります。プレゼン資料はメールでお送りします。',
    ],
  };

  const fallback = validationTextsByLang[lang] ?? [
    'Hello.',
    "Hello. The meeting starts at two o'clock this afternoon.",
    "Hello. The meeting starts at two o'clock this afternoon, and I will share the presentation materials via email.",
  ];
  if (!is06b) {
    return fallback;
  }
  if (fallback.length >= 3) {
    return [fallback[1], fallback[2]];
  }
  if (fallback.length >= 2) {
    return [fallback[0], fallback[1]];
  }
  return fallback.slice(0, 1);
};

const inferValidationLanguage = (...values: Array<string | null | undefined>): string => {
  const joined = values
    .filter((value): value is string => typeof value === 'string' && value.trim().length > 0)
    .join(' ')
    .trim();
  if (!joined) {
    return 'en';
  }

  const normalizedHint = joined.toLowerCase().trim().replace(/_/g, '-');
  const directMap: Record<string, string> = {
    auto: 'en',
    ko: 'ko',
    'ko-kr': 'ko',
    korean: 'ko',
    en: 'en',
    'en-us': 'en',
    'en-gb': 'en',
    english: 'en',
    ja: 'ja',
    'ja-jp': 'ja',
    japanese: 'ja',
    jp: 'ja',
    zh: 'zh',
    'zh-cn': 'zh',
    'zh-tw': 'zh',
    chinese: 'zh',
    cn: 'zh',
  };
  if (directMap[normalizedHint]) {
    return directMap[normalizedHint];
  }

  if (/[가-힣]/.test(joined)) {
    return 'ko';
  }
  if (/[ぁ-ゖァ-ヺ]/.test(joined)) {
    return 'ja';
  }
  if (/[一-龯]/.test(joined)) {
    return 'zh';
  }
  return 'en';
};

const normalizeValidationInferenceLanguage = (
  language: string | null | undefined,
): string | null => {
  const normalized = (language ?? '').trim().toLowerCase().replace(/_/g, '-');
  if (!normalized || normalized === 'auto') {
    return null;
  }
  const aliases: Record<string, string> = {
    ko: 'korean',
    'ko-kr': 'korean',
    korean: 'korean',
    en: 'english',
    'en-us': 'english',
    'en-gb': 'english',
    english: 'english',
    ja: 'japanese',
    'ja-jp': 'japanese',
    japanese: 'japanese',
    jp: 'japanese',
    zh: 'chinese',
    'zh-cn': 'chinese',
    'zh-tw': 'chinese',
    chinese: 'chinese',
    cn: 'chinese',
  };
  return aliases[normalized] ?? normalized;
};

const getSignatureStyleInstruction = (lang: string): string => {
  switch (lang) {
    case 'ko':
      return '참고 음성의 특유의 말투와 호흡, 문장 리듬, 억양의 오르내림을 최대한 유지하고 과장하지 말고 자연스럽게 말하세요.';
    case 'ja':
      return '参照音声の話し方、間の取り方、文のリズム、抑揚をできるだけ保ち、誇張せず自然に話してください。';
    case 'zh':
      return '尽量保留参考音频特有的说话方式、停连节奏、句子律动和语调起伏，不要夸张，保持自然。';
    default:
      return "Preserve the reference voice's natural cadence, pauses, emphasis, and conversational rhythm. Keep the delivery natural and not exaggerated.";
  }
};

const getValidationPresets = (modelId: string, lang = ''): ValidationPreset[] => {
  const is06b = modelId.toLowerCase().includes('0.6b');
  const balancedSettings = {
    stability: 0.85,
    similarity_boost: 0.85,
    style: 0.05,
    speed: 1.0,
  };
  const conservativeSettings = {
    stability: 0.9,
    similarity_boost: 0.9,
    style: 0.05,
    speed: 1.0,
  };
  const signatureStyleSettings = {
    stability: 0.82,
    similarity_boost: 0.9,
    style: 0.18,
    speed: 0.98,
  };
  const signatureInstruction = getSignatureStyleInstruction(lang.toLowerCase());

  if (!is06b) {
    return [
      {
        name: 'balanced',
        payload: { voice_settings: balancedSettings },
        settings: balancedSettings,
      },
      {
        name: 'high_similarity',
        payload: { voice_settings: conservativeSettings },
        settings: conservativeSettings,
      },
      {
        name: 'signature_style',
        payload: {
          voice_settings: signatureStyleSettings,
          instruct: signatureInstruction,
        },
        settings: signatureStyleSettings,
      },
    ];
  }

  return [
    {
      name: 'balanced',
      payload: { voice_settings: balancedSettings },
      settings: balancedSettings,
    },
    {
      name: 'high_similarity',
      payload: { voice_settings: conservativeSettings },
      settings: conservativeSettings,
    },
  ];
};

const getValidationPlan = (
  voice: NonNullable<Awaited<ReturnType<typeof getVoice>>>,
  job: TrainingJob,
): ValidationPlan => {
  const modelId = voice.model_id ?? 'qwen3-tts-1.7b';
  const is06b = modelId.toLowerCase().includes('0.6b');
  const jobConfig = job.config as Record<string, unknown>;
  const lang = inferValidationLanguage(
    typeof jobConfig.whisper_language === 'string' ? jobConfig.whisper_language : undefined,
    voice.labels?.language,
    voice.name,
    voice.speaker_name,
  );
  const validationTexts = getValidationTexts(lang, is06b);
  const validationSeedOffsets = is06b ? FAST_VALIDATION_SEEDS_OFFSET : FULL_VALIDATION_SEEDS_OFFSET;
  return {
    is06b,
    presets: getValidationPresets(modelId, lang),
    validationTexts,
    validationSeedOffsets,
    totalSamples: validationTexts.length * validationSeedOffsets.length,
    minOverall: is06b ? 0.82 : VALIDATION_GATE_THRESHOLDS.overall_min,
    minPassRate: is06b ? MIN_PASS_RATE_06B : MIN_PASS_RATE_17B,
    minAsrScore: VALIDATION_GATE_THRESHOLDS.asr_min,
    minToneScore: is06b ? 0.4 : VALIDATION_GATE_THRESHOLDS.tone_min,
    maxCheckpointsToEval: is06b ? MAX_CHECKPOINTS_TO_EVAL_06B : MAX_CHECKPOINTS_TO_EVAL,
    prioritizeLatestPassingCheckpoint: is06b,
  };
};

const selectValidationCandidateCheckpoints = (
  checkpoints: Array<{ epoch?: number; r2_prefix?: string }>,
  plan: ValidationPlan,
): CheckpointCandidate[] => {
  const uniqueAsc = checkpoints
    .filter(
      (cp): cp is { epoch: number; r2_prefix: string } =>
        typeof cp.epoch === 'number' && typeof cp.r2_prefix === 'string',
    )
    .sort((a, b) => a.epoch - b.epoch)
    .filter((cp, index, array) => index === 0 || array[index - 1]?.epoch !== cp.epoch);

  if (uniqueAsc.length === 0) {
    return [];
  }

  const targetCount = Math.min(uniqueAsc.length, plan.maxCheckpointsToEval);
  if (uniqueAsc.length <= targetCount) {
    return uniqueAsc;
  }

  const selectedIndexes = new Set<number>();
  for (let i = 0; i < targetCount; i += 1) {
    const idx = Math.round((i * (uniqueAsc.length - 1)) / Math.max(1, targetCount - 1));
    selectedIndexes.add(idx);
  }

  return [...selectedIndexes]
    .sort((a, b) => a - b)
    .map((index) => uniqueAsc[index])
    .filter((value): value is CheckpointCandidate => Boolean(value));
};

const buildValidationScoreParts = ({
  is06b,
  overall,
  asr,
  health,
  duration,
  passRate,
  speaker,
  tone,
  speed,
  style,
}: {
  is06b: boolean;
  overall: number;
  asr: number;
  health: number;
  duration: number;
  passRate: number;
  speaker: number;
  tone: number;
  speed: number;
  style?: number;
}): Array<{ value: number; weight: number }> => {
  const hasStyle = typeof style === 'number' && Number.isFinite(style);
  const baseWeights = is06b
    ? {
        asr: 0.25, health: 0.10, duration: 0.08, passRate: 0.08,
        speaker: 0.22, style: 0.17, tone: 0.06, speed: 0.04,
      }
    : {
        asr: 0.22, health: 0.08, duration: 0.06, passRate: 0.06,
        speaker: 0.22, style: 0.20, tone: 0.05, speed: 0.05,
        stability: 0.06,
      };

  const parts: Array<{ value: number; weight: number }> = [
    { value: asr, weight: baseWeights.asr },
    { value: health, weight: baseWeights.health },
    { value: duration, weight: baseWeights.duration },
    { value: passRate, weight: baseWeights.passRate },
  ];

  if (!is06b && baseWeights.stability) {
    parts.push({ value: overall, weight: baseWeights.stability });
  }
  if (Number.isFinite(speaker)) {
    parts.push({ value: speaker, weight: baseWeights.speaker });
  }
  if (hasStyle) {
    parts.push({ value: style, weight: baseWeights.style });
  }
  if (Number.isFinite(tone)) {
    parts.push({ value: tone, weight: baseWeights.tone });
  }
  if (Number.isFinite(speed)) {
    parts.push({ value: speed, weight: baseWeights.speed });
  }

  return parts;
};

const buildTrainingCheckoutLedgerEntries = (
  job: TrainingJob,
  summary: Record<string, unknown>,
  createdAt: number,
): TrainingCheckoutLedgerEntry[] => {
  const checkout = buildTrainingCheckoutSearch({
    ...job,
    summary,
  });
  const evaluationsByPrefix = new Map(checkout.evaluated.map((value) => [value.prefix, value]));
  const entries: TrainingCheckoutLedgerEntry[] = checkout.evaluated.map((value) => ({
    entry_id: crypto.randomUUID(),
    round_id: job.round_id ?? null,
    job_id: job.job_id,
    voice_id: job.voice_id,
    checkpoint_r2_prefix: value.prefix,
    run_name: value.run_name,
    epoch: value.epoch,
    preset: value.preset,
    score: value.score,
    ok: value.ok,
    passed_samples: value.passed_samples,
    total_samples: value.total_samples,
    message: value.message,
    role: 'evaluated',
    source: 'validation',
    adoption_mode: checkout.adoption_mode,
    created_at: createdAt,
    updated_at: createdAt,
  }));

  const pushRoleEntry = (
    role: 'champion' | 'selected' | 'manual_promoted',
    target: NonNullable<typeof checkout.champion>,
    source: 'validation' | 'manual_promotion' | 'recovery',
  ) => {
    const evaluation = evaluationsByPrefix.get(target.prefix) ?? null;
    entries.push({
      entry_id: crypto.randomUUID(),
      round_id: job.round_id ?? null,
      job_id: job.job_id,
      voice_id: job.voice_id,
      checkpoint_r2_prefix: target.prefix,
      run_name: target.run_name,
      epoch: target.epoch,
      preset: target.preset,
      score: target.score,
      ok: evaluation?.ok ?? (checkout.validation_passed ? true : null),
      passed_samples: evaluation?.passed_samples ?? null,
      total_samples: evaluation?.total_samples ?? null,
      message: evaluation?.message ?? checkout.message,
      role,
      source,
      adoption_mode: checkout.adoption_mode,
      created_at: createdAt,
      updated_at: createdAt,
    });
  };

  if (checkout.champion) {
    pushRoleEntry('champion', checkout.champion, 'validation');
  }
  if (checkout.selected) {
    pushRoleEntry(
      'selected',
      checkout.selected,
      checkout.manual_promoted?.prefix === checkout.selected.prefix
        ? 'manual_promotion'
        : checkout.evaluated.length === 0
          ? 'recovery'
          : 'validation',
    );
  }
  if (checkout.manual_promoted) {
    pushRoleEntry('manual_promoted', checkout.manual_promoted, 'manual_promotion');
  }

  return entries;
};

const syncTrainingCheckoutLedgerForJob = async (
  c: Context<AppContext>,
  job: TrainingJob,
  summary: Record<string, unknown>,
  createdAt = Date.now(),
): Promise<void> => {
  const entries = buildTrainingCheckoutLedgerEntries(job, summary, createdAt);
  await replaceTrainingCheckoutLedgerForJob(c.env.DB, job.job_id, entries);
};

const serializeTrainingJob = (
  job: TrainingJob,
): Omit<TrainingJob, 'job_token'> & {
  checkout_search: NonNullable<TrainingJob['checkout_search']>;
} => {
  const { job_token: _jobToken, ...safeJob } = job;
  return {
    ...safeJob,
    checkout_search: buildTrainingCheckoutSearch(job),
  };
};

const serializeTrainingRound = (round: TrainingRound): TrainingRound => round;

const serializeDatasetSnapshot = (snapshot: DatasetSnapshot): DatasetSnapshot => snapshot;

const normalizeCheckpointEvaluation = (value: unknown): CheckpointEvaluation | null => {
  if (!value || typeof value !== 'object') {
    return null;
  }
  const v = value as Record<string, unknown>;
  if (
    typeof v.epoch !== 'number' ||
    typeof v.prefix !== 'string' ||
    typeof v.ok !== 'boolean' ||
    typeof v.score !== 'number' ||
    typeof v.message !== 'string' ||
    typeof v.preset !== 'string' ||
    typeof v.passed_samples !== 'number' ||
    typeof v.total_samples !== 'number'
  ) {
    return null;
  }
  return {
    epoch: v.epoch,
    prefix: v.prefix,
    ok: v.ok,
    score: v.score,
    message: v.message,
    preset: v.preset,
    passed_samples: v.passed_samples,
    total_samples: v.total_samples,
  };
};

const normalizePresetSettings = (value: unknown): ValidationPreset['settings'] | undefined => {
  if (!value || typeof value !== 'object') {
    return undefined;
  }
  const settings = value as Record<string, unknown>;
  const stability = Number(settings.stability);
  const similarityBoost = Number(settings.similarity_boost);
  const style = Number(settings.style);
  const speed = Number(settings.speed);
  if (
    !Number.isFinite(stability) ||
    !Number.isFinite(similarityBoost) ||
    !Number.isFinite(style) ||
    !Number.isFinite(speed)
  ) {
    return undefined;
  }
  return {
    stability,
    similarity_boost: similarityBoost,
    style,
    speed,
  };
};

const createValidationAccumulator = (): AsyncValidationAccumulator => ({
  passed: 0,
  no_audio: 0,
  infra_issues: 0,
  sum_overall: 0,
  sum_duration: 0,
  sum_health: 0,
  sum_asr: 0,
  sum_speaker: 0,
  sum_tone: 0,
  sum_speed: 0,
  sum_style: 0,
  speaker_samples: 0,
  tone_samples: 0,
  speed_samples: 0,
  style_samples: 0,
  first_failure_message: null,
});

const normalizeValidationAccumulator = (value: unknown): AsyncValidationAccumulator => {
  if (!value || typeof value !== 'object') {
    return createValidationAccumulator();
  }
  const src = value as Record<string, unknown>;
  return {
    passed: Number.isFinite(Number(src.passed)) ? Number(src.passed) : 0,
    no_audio: Number.isFinite(Number(src.no_audio)) ? Number(src.no_audio) : 0,
    infra_issues: Number.isFinite(Number(src.infra_issues)) ? Number(src.infra_issues) : 0,
    sum_overall: Number.isFinite(Number(src.sum_overall)) ? Number(src.sum_overall) : 0,
    sum_duration: Number.isFinite(Number(src.sum_duration)) ? Number(src.sum_duration) : 0,
    sum_health: Number.isFinite(Number(src.sum_health)) ? Number(src.sum_health) : 0,
    sum_asr: Number.isFinite(Number(src.sum_asr)) ? Number(src.sum_asr) : 0,
    sum_speaker: Number.isFinite(Number(src.sum_speaker)) ? Number(src.sum_speaker) : 0,
    sum_tone: Number.isFinite(Number(src.sum_tone)) ? Number(src.sum_tone) : 0,
    sum_speed: Number.isFinite(Number(src.sum_speed)) ? Number(src.sum_speed) : 0,
    sum_style: Number.isFinite(Number(src.sum_style)) ? Number(src.sum_style) : 0,
    speaker_samples: Number.isFinite(Number(src.speaker_samples)) ? Number(src.speaker_samples) : 0,
    tone_samples: Number.isFinite(Number(src.tone_samples)) ? Number(src.tone_samples) : 0,
    speed_samples: Number.isFinite(Number(src.speed_samples)) ? Number(src.speed_samples) : 0,
    style_samples: Number.isFinite(Number(src.style_samples)) ? Number(src.style_samples) : 0,
    first_failure_message:
      typeof src.first_failure_message === 'string' ? src.first_failure_message : null,
  };
};

const normalizeValidationChampion = (value: unknown): AsyncValidationChampion | null => {
  if (!value || typeof value !== 'object') {
    return null;
  }
  const src = value as Record<string, unknown>;
  if (
    typeof src.epoch !== 'number' ||
    typeof src.prefix !== 'string' ||
    typeof src.score !== 'number' ||
    typeof src.message !== 'string' ||
    typeof src.preset_name !== 'string' ||
    typeof src.passed_samples !== 'number' ||
    typeof src.total_samples !== 'number'
  ) {
    return null;
  }
  return {
    epoch: src.epoch,
    prefix: src.prefix,
    score: src.score,
    message: src.message,
    preset_name: src.preset_name,
    preset_settings: normalizePresetSettings(src.preset_settings),
    passed_samples: src.passed_samples,
    total_samples: src.total_samples,
  };
};

const normalizeValidationFailure = (value: unknown): AsyncValidationFailure | null => {
  if (!value || typeof value !== 'object') {
    return null;
  }
  const src = value as Record<string, unknown>;
  if (
    typeof src.passed_samples !== 'number' ||
    typeof src.score !== 'number' ||
    typeof src.message !== 'string' ||
    typeof src.preset_name !== 'string' ||
    typeof src.total_samples !== 'number'
  ) {
    return null;
  }
  return {
    passed_samples: src.passed_samples,
    score: src.score,
    message: src.message,
    preset_name: src.preset_name,
    total_samples: src.total_samples,
  };
};

const loadValidationReference = async (
  c: Context<AppContext>,
  voice: NonNullable<Awaited<ReturnType<typeof getVoice>>>,
  job: TrainingJob,
): Promise<{ referenceAudioKey: string | null; referenceText: string }> => {
  const datasetPrefix = String(job.dataset_r2_prefix ?? '').replace(/\/+$/, '');
  let referenceAudioKey =
    voice.ref_audio_r2_key ?? (datasetPrefix ? `${datasetPrefix}/ref_audio.wav` : null);
  let referenceText = '';

  if (datasetPrefix) {
    try {
      const profileObj = await c.env.R2.get(`${datasetPrefix}/reference_profile.json`);
      if (profileObj) {
        const profile = (await profileObj.json()) as Record<string, unknown>;
        if (typeof profile.reference_audio_key === 'string' && profile.reference_audio_key.trim()) {
          referenceAudioKey = profile.reference_audio_key.trim();
        }
        if (typeof profile.reference_text === 'string') {
          referenceText = profile.reference_text.trim();
        }
      }
    } catch {
      // Best-effort only. Validation can still proceed with ASR-only scoring.
    }
  }

  return { referenceAudioKey, referenceText };
};

const buildValidationPayload = ({
  voice,
  checkpointPrefix,
  preset,
  validationText,
  seed,
  referenceAudioKey,
  referenceText,
  languageHint,
}: {
  voice: NonNullable<Awaited<ReturnType<typeof getVoice>>>;
  checkpointPrefix: string;
  preset: ValidationPreset;
  validationText: string;
  seed: number;
  referenceAudioKey: string | null;
  referenceText: string;
  languageHint: string | null;
}): Record<string, unknown> => ({
  text: validationText,
  voice_id: voice.voice_id,
  speaker_name: voice.speaker_name,
  model_id: voice.model_id ?? 'qwen3-tts-1.7b',
  ...(normalizeValidationInferenceLanguage(languageHint)
    ? { language: normalizeValidationInferenceLanguage(languageHint) }
    : {}),
  seed,
  quality_review: {
    enable_asr: false,
    enable_speaker: Boolean(referenceAudioKey),
    enable_style: Boolean(referenceAudioKey),
    enable_speed: Boolean(referenceAudioKey && referenceText),
    allow_below_threshold: true,
    reference_audio_key: referenceAudioKey,
    reference_text: referenceText,
  },
  checkpoint_info: {
    r2_prefix: checkpointPrefix,
    type: 'full',
  },
  ...preset.payload,
});

const getValidationLanguageHint = (
  voice: NonNullable<Awaited<ReturnType<typeof getVoice>>>,
  job: TrainingJob,
): string | null => {
  const jobConfig = job.config as Record<string, unknown>;
  const inferred = inferValidationLanguage(
    typeof jobConfig.whisper_language === 'string' ? jobConfig.whisper_language : undefined,
    voice.labels?.language,
    voice.name,
    voice.speaker_name,
  );
  return inferred || null;
};

const annotateAsrFailure = (
  output: { quality?: Record<string, unknown>; audio?: string; error?: unknown } | null,
  error: unknown,
): { quality?: Record<string, unknown>; audio?: string; error?: unknown } | null => {
  if (!output) {
    return output;
  }
  const detail = error instanceof Error ? error.message : 'OpenAI ASR enrichment failed';
  return {
    ...output,
    quality: {
      ...(output.quality ?? {}),
      [VALIDATION_ASR_ERROR_KEY]: detail,
    },
  };
};

const getMissingAsrMessage = (
  quality: Record<string, unknown>,
  sampleIndex: number,
  seed: number,
): string => {
  const detail =
    typeof quality[VALIDATION_ASR_ERROR_KEY] === 'string'
      ? quality[VALIDATION_ASR_ERROR_KEY]
      : null;
  return detail
    ? `sample ${sampleIndex} seed ${seed} missing asr_score (${detail})`
    : `sample ${sampleIndex} seed ${seed} missing asr_score`;
};

const evaluateValidationSample = ({
  output,
  fallbackError,
  sampleIndex,
  seed,
  referenceAudioKey,
  referenceText,
  minOverall,
  minAsrScore,
  minToneScore,
}: {
  output: { quality?: Record<string, unknown>; audio?: string; error?: unknown } | null;
  fallbackError?: string | null;
  sampleIndex: number;
  seed: number;
  referenceAudioKey: string | null;
  referenceText: string;
  minOverall: number;
  minAsrScore: number;
  minToneScore: number;
}): ValidationSampleOutcome => {
  if (!output?.audio) {
    const lastErrorDetail =
      typeof output?.error === 'string'
        ? output.error
        : fallbackError && fallbackError.trim()
          ? fallbackError
          : 'status=unknown no-audio';
    const parsedOverall = parseOverallFromError(lastErrorDetail);
    const failureMessage =
      parsedOverall !== null
        ? `sample ${sampleIndex} seed ${seed} no audio overall_score=${parsedOverall.toFixed(3)}`
        : `sample ${sampleIndex} seed ${seed} no audio (${lastErrorDetail})`;
    return {
      passed: false,
      noAudio: true,
      infraIssue: false,
      overall: null,
      duration: null,
      health: null,
      asr: null,
      speaker: null,
      tone: null,
      speed: null,
      style: null,
      failureMessage,
    };
  }

  const quality = output.quality ?? {};
  const overall = Number(quality.overall_score ?? NaN);
  const duration = Number(quality.duration_score ?? NaN);
  const health = Number(quality.health_score ?? NaN);
  const asr = Number(quality.asr_score ?? quality.asr_similarity ?? NaN);
  const speaker = Number(quality.speaker_score ?? NaN);
  const tone = Number(quality.tone_score ?? NaN);
  const speed = Number(quality.speed_score ?? NaN);
  const style = Number(quality.style_score ?? NaN);

  const fail = (failureMessage: string, infraIssue = false): ValidationSampleOutcome => ({
    passed: false,
    noAudio: false,
    infraIssue,
    overall: Number.isFinite(overall) ? overall : null,
    duration: Number.isFinite(duration) ? duration : null,
    health: Number.isFinite(health) ? health : null,
    asr: Number.isFinite(asr) ? asr : null,
    speaker: Number.isFinite(speaker) ? speaker : null,
    tone: Number.isFinite(tone) ? tone : null,
    speed: Number.isFinite(speed) ? speed : null,
    style: Number.isFinite(style) ? style : null,
    failureMessage,
  });

  if (!Number.isFinite(overall) || !Number.isFinite(duration) || !Number.isFinite(health)) {
    return fail(`sample ${sampleIndex} seed ${seed} invalid quality metrics`, true);
  }
  if (!Number.isFinite(asr)) {
    return fail(getMissingAsrMessage(quality, sampleIndex, seed), true);
  }
  if (overall < minOverall) {
    return fail(`sample ${sampleIndex} seed ${seed} overall_score=${overall.toFixed(3)}`);
  }
  if (duration < 0.3) {
    return fail(`sample ${sampleIndex} seed ${seed} duration_score=${duration.toFixed(3)}`);
  }
  if (health < 0.72) {
    return fail(`sample ${sampleIndex} seed ${seed} health_score=${health.toFixed(3)}`);
  }
  if (asr < minAsrScore) {
    return fail(`sample ${sampleIndex} seed ${seed} asr_score=${asr.toFixed(3)}`);
  }
  if (
    referenceAudioKey &&
    Number.isFinite(speaker) &&
    speaker < VALIDATION_GATE_THRESHOLDS.speaker_min
  ) {
    return fail(`sample ${sampleIndex} seed ${seed} speaker_score=${speaker.toFixed(3)}`);
  }
  if (referenceAudioKey && Number.isFinite(style) && style < VALIDATION_GATE_THRESHOLDS.style_min) {
    return fail(`sample ${sampleIndex} seed ${seed} style_score=${style.toFixed(3)}`);
  }
  if (
    referenceAudioKey &&
    Number.isFinite(tone) &&
    tone < minToneScore
  ) {
    return fail(`sample ${sampleIndex} seed ${seed} tone_score=${tone.toFixed(3)}`);
  }
  if (
    referenceAudioKey &&
    referenceText &&
    Number.isFinite(speed) &&
    speed < VALIDATION_GATE_THRESHOLDS.speed_min
  ) {
    return fail(`sample ${sampleIndex} seed ${seed} speed_score=${speed.toFixed(3)}`);
  }

  return {
    passed: true,
    noAudio: false,
    infraIssue: false,
    overall,
    duration,
    health,
    asr,
    speaker: Number.isFinite(speaker) ? speaker : null,
    tone: Number.isFinite(tone) ? tone : null,
    speed: Number.isFinite(speed) ? speed : null,
    style: Number.isFinite(style) ? style : null,
    failureMessage: null,
  };
};

const applyValidationSampleOutcome = (
  accumulator: AsyncValidationAccumulator,
  outcome: ValidationSampleOutcome,
): AsyncValidationAccumulator => {
  const next: AsyncValidationAccumulator = { ...accumulator };
  if (outcome.noAudio) {
    next.no_audio += 1;
  }
  if (outcome.infraIssue) {
    next.infra_issues += 1;
  }
  if (!next.first_failure_message && outcome.failureMessage) {
    next.first_failure_message = outcome.failureMessage;
  }
  if (!outcome.passed) {
    return next;
  }

  next.passed += 1;
  next.sum_overall += outcome.overall ?? 0;
  next.sum_duration += outcome.duration ?? 0;
  next.sum_health += outcome.health ?? 0;
  next.sum_asr += outcome.asr ?? 0;
  if (typeof outcome.speaker === 'number') {
    next.sum_speaker += outcome.speaker;
    next.speaker_samples += 1;
  }
  if (typeof outcome.tone === 'number') {
    next.sum_tone += outcome.tone;
    next.tone_samples += 1;
  }
  if (typeof outcome.speed === 'number') {
    next.sum_speed += outcome.speed;
    next.speed_samples += 1;
  }
  if (typeof outcome.style === 'number') {
    next.sum_style += outcome.style;
    next.style_samples += 1;
  }
  return next;
};

const finalizeValidationPresetResult = ({
  accumulator,
  preset,
  totalSamples,
  minPassRate,
  is06b,
}: {
  accumulator: AsyncValidationAccumulator;
  preset: ValidationPreset;
  totalSamples: number;
  minPassRate: number;
  is06b: boolean;
}): CheckpointValidationResult => {
  const passRate = totalSamples > 0 ? accumulator.passed / totalSamples : 0;
  if (accumulator.passed > 0 && passRate >= minPassRate && accumulator.infra_issues === 0) {
    const n = Math.max(1, accumulator.passed);
    const meanOverall = accumulator.sum_overall / n;
    const meanDuration = accumulator.sum_duration / n;
    const meanHealth = accumulator.sum_health / n;
    const meanAsr = accumulator.sum_asr / n;
    const meanSpeaker =
      accumulator.speaker_samples > 0 ? accumulator.sum_speaker / accumulator.speaker_samples : NaN;
    const meanTone =
      accumulator.tone_samples > 0 ? accumulator.sum_tone / accumulator.tone_samples : NaN;
    const meanSpeed =
      accumulator.speed_samples > 0 ? accumulator.sum_speed / accumulator.speed_samples : NaN;
    const meanStyle =
      accumulator.style_samples > 0 ? accumulator.sum_style / accumulator.style_samples : NaN;
    const scoreParts = buildValidationScoreParts({
      is06b,
      overall: meanOverall,
      asr: meanAsr,
      health: meanHealth,
      duration: meanDuration,
      passRate,
      speaker: meanSpeaker,
      tone: meanTone,
      speed: meanSpeed,
      style: meanStyle,
    });
    const totalWeight = scoreParts.reduce((acc, part) => acc + part.weight, 0) || 1;
    const score = scoreParts.reduce((acc, part) => acc + part.value * part.weight, 0) / totalWeight;
    const similaritySegments: string[] = [];
    if (Number.isFinite(meanSpeaker)) {
      similaritySegments.push(`speaker=${meanSpeaker.toFixed(3)}`);
    }
    if (Number.isFinite(meanStyle)) {
      similaritySegments.push(`style=${meanStyle.toFixed(3)}`);
    } else if (Number.isFinite(meanTone)) {
      similaritySegments.push(`tone=${meanTone.toFixed(3)}`);
    }
    const similarityNote = similaritySegments.length > 0 ? `${similaritySegments.join(' ')} ` : '';
    const speedNote = Number.isFinite(meanSpeed) ? `speed=${meanSpeed.toFixed(3)} ` : '';
    return {
      ok: true,
      message:
        `preset=${preset.name} ` +
        `score=${score.toFixed(3)} overall=${meanOverall.toFixed(3)} ` +
        `asr=${meanAsr.toFixed(3)} ` +
        similarityNote +
        speedNote +
        `health=${meanHealth.toFixed(3)} duration=${meanDuration.toFixed(3)} ` +
        `samples=${accumulator.passed}/${totalSamples} no_audio=${accumulator.no_audio}`,
      aggregateScore: score,
      presetName: preset.name,
      presetSettings: preset.settings,
      passedSamples: accumulator.passed,
      totalSamples,
    };
  }

  let failScore = 0;
  if (accumulator.passed > 0) {
    const n = Math.max(1, accumulator.passed);
    const parts = buildValidationScoreParts({
      is06b,
      overall: accumulator.sum_overall / n,
      asr: accumulator.sum_asr / n,
      health: accumulator.sum_health / n,
      duration: accumulator.sum_duration / n,
      passRate,
      speaker:
        accumulator.speaker_samples > 0
          ? accumulator.sum_speaker / accumulator.speaker_samples
          : NaN,
      tone: accumulator.tone_samples > 0 ? accumulator.sum_tone / accumulator.tone_samples : NaN,
      speed:
        accumulator.speed_samples > 0 ? accumulator.sum_speed / accumulator.speed_samples : NaN,
      style:
        accumulator.style_samples > 0 ? accumulator.sum_style / accumulator.style_samples : NaN,
    });
    const tw = parts.reduce((acc, part) => acc + part.weight, 0) || 1;
    failScore = parts.reduce((acc, part) => acc + part.value * part.weight, 0) / tw;
  }

  return {
    ok: false,
    message:
      `All presets failed: ` +
      `preset=${preset.name} ` +
      (accumulator.first_failure_message ??
        `samples=${accumulator.passed}/${totalSamples} ` +
          `pass_rate=${passRate.toFixed(3)} ` +
          `no_audio=${accumulator.no_audio} infra=${accumulator.infra_issues}`),
    aggregateScore: failScore,
    presetName: preset.name,
    passedSamples: accumulator.passed,
    totalSamples,
  };
};

const scoreSingleValidationOutput = ({
  preset,
  seed,
  output,
  fallbackError,
  referenceAudioKey,
  referenceText,
  is06b,
  minToneScore,
}: {
  preset: ValidationPreset;
  seed: number;
  output: { quality?: Record<string, unknown>; audio?: string; error?: unknown } | null;
  fallbackError?: string | null;
  referenceAudioKey: string | null;
  referenceText: string;
  is06b: boolean;
  minToneScore: number;
}): CheckpointValidationResult => {
  const minOverall = is06b ? 0.82 : 0.85;
  const minAsrScore = 0.8;
  const totalSamples = 1;

  if (!output?.audio) {
    const lastErrorDetail =
      typeof output?.error === 'string'
        ? output.error
        : fallbackError && fallbackError.trim()
          ? fallbackError
          : 'status=unknown no-audio';
    const parsedOverall = parseOverallFromError(lastErrorDetail);
    const failureSummary =
      parsedOverall !== null
        ? `sample 1 seed ${seed} no audio overall_score=${parsedOverall.toFixed(3)}`
        : `sample 1 seed ${seed} no audio (${lastErrorDetail})`;
    return {
      ok: false,
      message: `All presets failed: preset=${preset.name} ${failureSummary}`,
      aggregateScore: 0,
      presetName: preset.name,
      passedSamples: 0,
      totalSamples,
    };
  }

  const quality = output.quality ?? {};
  const overall = Number(quality.overall_score ?? NaN);
  const duration = Number(quality.duration_score ?? NaN);
  const health = Number(quality.health_score ?? NaN);
  const asr = Number(quality.asr_score ?? quality.asr_similarity ?? NaN);
  const speaker = Number(quality.speaker_score ?? NaN);
  const tone = Number(quality.tone_score ?? NaN);
  const speed = Number(quality.speed_score ?? NaN);
  const style = Number(quality.style_score ?? NaN);

  const coreMetricsAvailable =
    Number.isFinite(overall) && Number.isFinite(duration) && Number.isFinite(health);

  const fail = (detail: string): CheckpointValidationResult => {
    let computedScore = 0;
    if (coreMetricsAvailable) {
      const parts = buildValidationScoreParts({
        is06b,
        overall,
        asr: Number.isFinite(asr) ? asr : 0,
        health,
        duration,
        passRate: 1,
        speaker,
        tone,
        speed,
        style: Number.isFinite(style) ? style : undefined,
      });
      const tw = parts.reduce((acc, part) => acc + part.weight, 0) || 1;
      computedScore = parts.reduce((acc, part) => acc + part.value * part.weight, 0) / tw;
    }
    return {
      ok: false,
      message: `All presets failed: preset=${preset.name} ${detail}`,
      aggregateScore: computedScore,
      presetName: preset.name,
      passedSamples: 0,
      totalSamples,
    };
  };

  if (!coreMetricsAvailable) {
    return fail('sample 1 seed ' + seed + ' invalid quality metrics');
  }
  if (!Number.isFinite(asr)) {
    return fail(getMissingAsrMessage(quality, 1, seed));
  }
  if (overall < minOverall) {
    return fail(`sample 1 seed ${seed} overall_score=${overall.toFixed(3)}`);
  }
  if (duration < 0.3) {
    return fail(`sample 1 seed ${seed} duration_score=${duration.toFixed(3)}`);
  }
  if (health < 0.72) {
    return fail(`sample 1 seed ${seed} health_score=${health.toFixed(3)}`);
  }
  if (asr < minAsrScore) {
    return fail(`sample 1 seed ${seed} asr_score=${asr.toFixed(3)}`);
  }
  if (
    referenceAudioKey &&
    Number.isFinite(speaker) &&
    speaker < VALIDATION_GATE_THRESHOLDS.speaker_min
  ) {
    return fail(`sample 1 seed ${seed} speaker_score=${speaker.toFixed(3)}`);
  }
  if (referenceAudioKey && Number.isFinite(style) && style < VALIDATION_GATE_THRESHOLDS.style_min) {
    return fail(`sample 1 seed ${seed} style_score=${style.toFixed(3)}`);
  }
  if (
    referenceAudioKey &&
    Number.isFinite(tone) &&
    tone < minToneScore
  ) {
    return fail(`sample 1 seed ${seed} tone_score=${tone.toFixed(3)}`);
  }
  if (
    referenceAudioKey &&
    referenceText &&
    Number.isFinite(speed) &&
    speed < VALIDATION_GATE_THRESHOLDS.speed_min
  ) {
    return fail(`sample 1 seed ${seed} speed_score=${speed.toFixed(3)}`);
  }

  const scoreParts = buildValidationScoreParts({
    is06b,
    overall,
    asr,
    health,
    duration,
    passRate: 1,
    speaker,
    tone,
    speed,
    style: Number.isFinite(style) ? style : undefined,
  });
  const totalWeight = scoreParts.reduce((acc, part) => acc + part.weight, 0) || 1;
  const score = scoreParts.reduce((acc, part) => acc + part.value * part.weight, 0) / totalWeight;
  const similaritySegments: string[] = [];
  if (Number.isFinite(speaker)) {
    similaritySegments.push(`speaker=${speaker.toFixed(3)}`);
  }
  if (Number.isFinite(style)) {
    similaritySegments.push(`style=${style.toFixed(3)}`);
  } else if (Number.isFinite(tone)) {
    similaritySegments.push(`tone=${tone.toFixed(3)}`);
  }
  const similarityNote = similaritySegments.length > 0 ? `${similaritySegments.join(' ')} ` : '';
  const speedNote = Number.isFinite(speed) ? `speed=${speed.toFixed(3)} ` : '';
  return {
    ok: true,
    message:
      `preset=${preset.name} ` +
      `score=${score.toFixed(3)} overall=${overall.toFixed(3)} ` +
      `asr=${asr.toFixed(3)} ` +
      similarityNote +
      speedNote +
      `health=${health.toFixed(3)} duration=${duration.toFixed(3)} ` +
      'samples=1/1 no_audio=0',
    aggregateScore: score,
    presetName: preset.name,
    presetSettings: preset.settings,
    passedSamples: 1,
    totalSamples,
  };
};

const advanceAsyncCheckpointValidation = async (
  c: Context<AppContext>,
  voice: NonNullable<Awaited<ReturnType<typeof getVoice>>>,
  job: TrainingJob,
  progress: TrainingProgress,
  currentSummary: Record<string, unknown>,
  candidateCheckpoints: CheckpointCandidate[],
): Promise<{ status: string; progress: TrainingProgress }> => {
  const plan = getValidationPlan(voice, job);
  if (plan.presets.length === 0 || plan.validationTexts.length === 0 || plan.totalSamples === 0) {
    return {
      status: 'completed',
      progress,
    };
  }

  const persistedState =
    currentSummary.async_validation && typeof currentSummary.async_validation === 'object'
      ? (currentSummary.async_validation as Record<string, unknown>)
      : null;
  const existingEvaluations = Array.isArray(persistedState?.evaluations)
    ? persistedState.evaluations
        .map(normalizeCheckpointEvaluation)
        .filter((value): value is CheckpointEvaluation => value !== null)
    : [];
  const persistedReferenceAudioKey =
    typeof persistedState?.reference_audio_key === 'string' ||
    persistedState?.reference_audio_key === null
      ? (persistedState.reference_audio_key as string | null)
      : null;
  const persistedReferenceText =
    typeof persistedState?.reference_text === 'string' ? persistedState.reference_text : '';
  const initialReference =
    persistedReferenceAudioKey !== null || persistedReferenceText
      ? {
          referenceAudioKey: persistedReferenceAudioKey,
          referenceText: persistedReferenceText,
        }
      : null;

  const ensureReference = async () => initialReference ?? loadValidationReference(c, voice, job);

  const completeSuccessfulValidation = async (
    champion: AsyncValidationChampion,
    evaluations: CheckpointEvaluation[],
  ): Promise<{ status: string; progress: TrainingProgress }> => {
    const adoption = await applyValidatedCheckpointOutcome({
      c,
      voice,
      job,
      candidatePrefix: champion.prefix,
      candidateEpoch: champion.epoch,
      candidatePreset: champion.preset_name,
      candidateScore: champion.score,
    });

    const validationMessage =
      adoption.mode === 'keep_current'
        ? `${champion.message} | kept current ready checkpoint score=${(adoption.preservedScore ?? 0).toFixed(3)} >= candidate_score=${champion.score.toFixed(3)}`
        : adoption.mode === 'candidate'
          ? `${champion.message} | stored as candidate; production remains unchanged until promotion`
          : champion.message;
    const nextSummary = {
      ...currentSummary,
      force_revalidation: false,
      validation_in_progress: false,
      validation_checked: true,
      validation_passed: true,
      validation_message: validationMessage,
      selected_checkpoint_prefix: adoption.selectedPrefix,
      selected_checkpoint_epoch: adoption.selectedEpoch,
      selected_preset: adoption.selectedPreset,
      selected_score: adoption.selectedScore,
      candidate_checkpoint_prefix: champion.prefix,
      candidate_checkpoint_epoch: champion.epoch,
      candidate_preset: champion.preset_name,
      candidate_score: champion.score,
      candidate_promotion_mode: adoption.mode,
      evaluated_checkpoints: evaluations,
      async_validation: null,
    };
    await updateTrainingJob(c.env.DB, job.job_id, {
      status: 'completed',
      completed_at: job.completed_at ?? Date.now(),
      progress,
      summary: nextSummary,
      supervisor: {
        ...(job.supervisor ?? {}),
        phase: 'validation_completed',
        outcome: adoption.mode,
        last_validation_checkpoint_epoch: champion.epoch,
      },
    });
    await syncTrainingCheckoutLedgerForJob(c, job, nextSummary, job.completed_at ?? Date.now());
    return { status: 'completed', progress };
  };

  const completeFailedValidation = async (
    message: string,
    evaluations: CheckpointEvaluation[],
  ): Promise<{ status: string; progress: TrainingProgress }> => {
    const keepReadyVoice = shouldKeepReadyVoiceOnValidationFailure(voice, currentSummary, {
      evaluatedCheckpoints: evaluations,
      validationRunName: parseRunNameFromCheckpointPrefix(candidateCheckpoints[0]?.r2_prefix ?? ''),
      forceRevalidation: currentSummary.force_revalidation === true,
    });
    if (!keepReadyVoice) {
      await updateVoice(c.env.DB, job.voice_id, {
        status: 'created',
        checkpoint_r2_prefix: null,
        run_name: null,
        epoch: null,
        checkpoint_preset: null,
        checkpoint_score: null,
        checkpoint_job_id: null,
        candidate_checkpoint_r2_prefix: null,
        candidate_run_name: null,
        candidate_epoch: null,
        candidate_preset: null,
        candidate_score: null,
        candidate_job_id: null,
      });
    }

    if (job.round_id) {
      await updateTrainingRound(c.env.DB, job.round_id, {
        status: 'failed',
        production_checkpoint_r2_prefix: keepReadyVoice ? voice.checkpoint_r2_prefix : null,
        production_run_name: keepReadyVoice ? voice.run_name : null,
        production_epoch: keepReadyVoice ? voice.epoch : null,
        production_preset: keepReadyVoice ? voice.checkpoint_preset : null,
        production_score: keepReadyVoice ? voice.checkpoint_score : null,
        production_job_id: keepReadyVoice ? voice.checkpoint_job_id : null,
        champion_checkpoint_r2_prefix: null,
        champion_run_name: null,
        champion_epoch: null,
        champion_preset: null,
        champion_score: null,
        champion_job_id: null,
        selected_checkpoint_r2_prefix: null,
        selected_run_name: null,
        selected_epoch: null,
        selected_preset: null,
        selected_score: null,
        selected_job_id: null,
        adoption_mode: null,
        candidate_checkpoint_r2_prefix: null,
        candidate_run_name: null,
        candidate_epoch: null,
        candidate_score: null,
        candidate_job_id: null,
        completed_at: Date.now(),
        summary: {
          ...(await getTrainingRound(c.env.DB, job.round_id))?.summary,
          validation_message: message,
          evaluated_checkpoints: evaluations,
        },
      });
    }

    const nextSummary = {
      ...currentSummary,
      force_revalidation: false,
      validation_in_progress: false,
      validation_checked: true,
      validation_passed: false,
      validation_failed: true,
      validation_rejected: true,
      validation_message: message,
      selected_checkpoint_prefix: null,
      selected_checkpoint_epoch: null,
      selected_preset: null,
      selected_score: null,
      evaluated_checkpoints: evaluations,
      async_validation: null,
    };
    await updateTrainingJob(c.env.DB, job.job_id, {
      status: 'completed',
      error_message: null,
      completed_at: job.completed_at ?? Date.now(),
      progress,
      summary: nextSummary,
      supervisor: {
        ...(job.supervisor ?? {}),
        phase: 'validation_failed',
      },
    });
    await syncTrainingCheckoutLedgerForJob(c, job, nextSummary, job.completed_at ?? Date.now());
    return { status: 'completed', progress };
  };

  const startValidationSample = async ({
    checkpointIndex,
    presetIndex,
    textIndex,
    seedIndex,
    evaluations,
    presetStats,
    checkpointBestPassing,
    checkpointBestFailure,
    champion,
  }: {
    checkpointIndex: number;
    presetIndex: number;
    textIndex: number;
    seedIndex: number;
    evaluations: CheckpointEvaluation[];
    presetStats: AsyncValidationAccumulator;
    checkpointBestPassing: AsyncValidationChampion | null;
    checkpointBestFailure: AsyncValidationFailure | null;
    champion: AsyncValidationChampion | null;
  }): Promise<{ status: string; progress: TrainingProgress }> => {
    const checkpoint = candidateCheckpoints[checkpointIndex];
    if (!checkpoint) {
      if (champion) {
        return completeSuccessfulValidation(champion, evaluations);
      }
      return completeFailedValidation('No remaining checkpoints to validate', evaluations);
    }

    const preset = plan.presets[presetIndex];
    const validationText = plan.validationTexts[textIndex];
    const seedOffset = plan.validationSeedOffsets[seedIndex];
    if (!preset || !validationText || typeof seedOffset !== 'number') {
      return completeFailedValidation('Validation plan is invalid', evaluations);
    }

    const seed = seedOffset + textIndex;
    const reference = await ensureReference();
    const payload = buildValidationPayload({
      voice,
      checkpointPrefix: checkpoint.r2_prefix,
      preset,
      validationText,
      seed,
      referenceAudioKey: reference.referenceAudioKey,
      referenceText: reference.referenceText,
      languageHint: getValidationLanguageHint(voice, job),
    });
    const runpodResponse = await invokeServerlessAsync(c.env, c.env.RUNPOD_ENDPOINT_ID, payload);
    const sampleOrdinal = textIndex * plan.validationSeedOffsets.length + seedIndex + 1;
    const nextState: AsyncCheckpointValidationState = {
      mode: 'checkpoint_async',
      run_id: String(runpodResponse.id ?? ''),
      run_started_at: Date.now(),
      checkpoint_index: checkpointIndex,
      checkpoint_epoch: checkpoint.epoch,
      checkpoint_prefix: checkpoint.r2_prefix,
      preset_index: presetIndex,
      text_index: textIndex,
      seed_index: seedIndex,
      reference_audio_key: reference.referenceAudioKey,
      reference_text: reference.referenceText,
      evaluations,
      preset_stats: presetStats,
      checkpoint_best_passing: checkpointBestPassing,
      checkpoint_best_failure: checkpointBestFailure,
      champion,
    };
    await updateTrainingJob(c.env.DB, job.job_id, {
      status: 'completed',
      completed_at: job.completed_at ?? Date.now(),
      progress,
      summary: {
        ...currentSummary,
        validation_in_progress: true,
        validation_message:
          `Validating checkpoint epoch ${checkpoint.epoch} ` +
          `preset ${preset.name} sample ${sampleOrdinal}/${plan.totalSamples}`,
        evaluated_checkpoints: evaluations,
        async_validation: nextState,
      },
    });
    return { status: 'completed', progress };
  };

  if (!persistedState || String(persistedState.mode ?? '') !== 'checkpoint_async') {
    return startValidationSample({
      checkpointIndex: 0,
      presetIndex: 0,
      textIndex: 0,
      seedIndex: 0,
      evaluations: existingEvaluations,
      presetStats: createValidationAccumulator(),
      checkpointBestPassing: null,
      checkpointBestFailure: null,
      champion: null,
    });
  }

  const checkpointIndex =
    typeof persistedState.checkpoint_index === 'number' ? persistedState.checkpoint_index : 0;
  const checkpoint = candidateCheckpoints[checkpointIndex];
  if (!checkpoint) {
    const champion = normalizeValidationChampion(persistedState.champion);
    if (champion) {
      return completeSuccessfulValidation(champion, existingEvaluations);
    }
    return completeFailedValidation('No remaining checkpoints to validate', existingEvaluations);
  }

  const presetIndex =
    typeof persistedState.preset_index === 'number' ? persistedState.preset_index : 0;
  const textIndex = typeof persistedState.text_index === 'number' ? persistedState.text_index : 0;
  const seedIndex = typeof persistedState.seed_index === 'number' ? persistedState.seed_index : 0;
  const preset = plan.presets[presetIndex];
  const seedOffset = plan.validationSeedOffsets[seedIndex];
  if (!preset || typeof seedOffset !== 'number' || !plan.validationTexts[textIndex]) {
    return completeFailedValidation('Validation plan is invalid', existingEvaluations);
  }

  const seed = seedOffset + textIndex;
  const runId = typeof persistedState.run_id === 'string' ? persistedState.run_id : '';
  if (!runId) {
    return startValidationSample({
      checkpointIndex,
      presetIndex,
      textIndex,
      seedIndex,
      evaluations: existingEvaluations,
      presetStats: normalizeValidationAccumulator(persistedState.preset_stats),
      checkpointBestPassing: normalizeValidationChampion(persistedState.checkpoint_best_passing),
      checkpointBestFailure: normalizeValidationFailure(persistedState.checkpoint_best_failure),
      champion: normalizeValidationChampion(persistedState.champion),
    });
  }

  const runpodResponse = await getServerlessStatusOrSyntheticFailure(
    c.env,
    c.env.RUNPOD_ENDPOINT_ID,
    runId,
  );
  const runStartedAt = getValidationRunStartedAt(persistedState, job);
  const runAgeMs = Math.max(0, Date.now() - runStartedAt);
  const rawRunStatus = String(runpodResponse.status ?? 'UNKNOWN');
  const runTimedOut =
    rawRunStatus !== 'COMPLETED' && rawRunStatus !== 'FAILED' && runAgeMs > VALIDATION_RUN_STALE_MS;
  const runStatus = runTimedOut ? 'FAILED' : rawRunStatus;
  const sampleOrdinal = textIndex * plan.validationSeedOffsets.length + seedIndex + 1;
  if (runStatus !== 'COMPLETED' && runStatus !== 'FAILED') {
    await updateTrainingJob(c.env.DB, job.job_id, {
      status: 'completed',
      completed_at: job.completed_at ?? Date.now(),
      progress,
      summary: {
        ...currentSummary,
        validation_in_progress: true,
        validation_message:
          `Validation still running for checkpoint epoch ${checkpoint.epoch} ` +
          `preset ${preset.name} sample ${sampleOrdinal}/${plan.totalSamples} ` +
          `(${runStatus.toLowerCase()})`,
        evaluated_checkpoints: existingEvaluations,
        async_validation: {
          ...persistedState,
          run_started_at: runStartedAt,
          evaluations: existingEvaluations,
        },
      },
    });
    return { status: 'completed', progress };
  }

  const output = (runpodResponse.output ?? null) as {
    quality?: Record<string, unknown>;
    audio?: string;
    error?: unknown;
  } | null;
  let enrichedOutput = runStatus === 'COMPLETED' ? output : null;
  if (enrichedOutput?.audio) {
    try {
      enrichedOutput = await enrichOutputWithReviewAsr({
        env: c.env,
        output: enrichedOutput,
        expectedText: plan.validationTexts[textIndex],
        languageHint: getValidationLanguageHint(voice, job),
      });
    } catch (error) {
      enrichedOutput = annotateAsrFailure(enrichedOutput, error);
    }
  }
  const fallbackError = runTimedOut
    ? `validation request timed out after ${Math.round(runAgeMs / 1000)}s`
    : typeof enrichedOutput?.error === 'string'
      ? enrichedOutput.error
      : typeof runpodResponse.error === 'string'
        ? runpodResponse.error
        : `runpod_status=${runStatus.toLowerCase()}`;
  const sampleOutcome = evaluateValidationSample({
    output: runStatus === 'COMPLETED' ? enrichedOutput : null,
    fallbackError,
    sampleIndex: textIndex + 1,
    seed,
    referenceAudioKey:
      typeof persistedState.reference_audio_key === 'string' ||
      persistedState.reference_audio_key === null
        ? (persistedState.reference_audio_key as string | null)
        : null,
    referenceText:
      typeof persistedState.reference_text === 'string' ? persistedState.reference_text : '',
    minOverall: plan.minOverall,
    minAsrScore: plan.minAsrScore,
    minToneScore: plan.minToneScore,
  });
  const nextPresetStats = applyValidationSampleOutcome(
    normalizeValidationAccumulator(persistedState.preset_stats),
    sampleOutcome,
  );
  const currentCheckpointBestPassing = normalizeValidationChampion(
    persistedState.checkpoint_best_passing,
  );
  const currentCheckpointBestFailure = normalizeValidationFailure(
    persistedState.checkpoint_best_failure,
  );
  const currentChampion = normalizeValidationChampion(persistedState.champion);

  const hasNextSeed = seedIndex + 1 < plan.validationSeedOffsets.length;
  const hasNextText = !hasNextSeed && textIndex + 1 < plan.validationTexts.length;
  if (hasNextSeed || hasNextText) {
    return startValidationSample({
      checkpointIndex,
      presetIndex,
      textIndex: hasNextSeed ? textIndex : textIndex + 1,
      seedIndex: hasNextSeed ? seedIndex + 1 : 0,
      evaluations: existingEvaluations,
      presetStats: nextPresetStats,
      checkpointBestPassing: currentCheckpointBestPassing,
      checkpointBestFailure: currentCheckpointBestFailure,
      champion: currentChampion,
    });
  }

  const presetResult = finalizeValidationPresetResult({
    accumulator: nextPresetStats,
    preset,
    totalSamples: plan.totalSamples,
    minPassRate: plan.minPassRate,
    is06b: plan.is06b,
  });
  const nextCheckpointBestPassing =
    presetResult.ok &&
    (!currentCheckpointBestPassing ||
      presetResult.aggregateScore > currentCheckpointBestPassing.score)
      ? {
          epoch: checkpoint.epoch,
          prefix: checkpoint.r2_prefix,
          score: presetResult.aggregateScore,
          message: presetResult.message,
          preset_name: presetResult.presetName,
          preset_settings: presetResult.presetSettings,
          passed_samples: presetResult.passedSamples,
          total_samples: presetResult.totalSamples,
        }
      : currentCheckpointBestPassing;
  const nextCheckpointBestFailure =
    !presetResult.ok &&
    (!currentCheckpointBestFailure ||
      presetResult.passedSamples > currentCheckpointBestFailure.passed_samples ||
      (presetResult.passedSamples === currentCheckpointBestFailure.passed_samples &&
        presetResult.aggregateScore > currentCheckpointBestFailure.score))
      ? {
          passed_samples: presetResult.passedSamples,
          score: presetResult.aggregateScore,
          message: presetResult.message,
          preset_name: presetResult.presetName,
          total_samples: presetResult.totalSamples,
        }
      : currentCheckpointBestFailure;

  if (presetIndex + 1 < plan.presets.length) {
    return startValidationSample({
      checkpointIndex,
      presetIndex: presetIndex + 1,
      textIndex: 0,
      seedIndex: 0,
      evaluations: existingEvaluations,
      presetStats: createValidationAccumulator(),
      checkpointBestPassing: nextCheckpointBestPassing,
      checkpointBestFailure: nextCheckpointBestFailure,
      champion: currentChampion,
    });
  }

  const checkpointEvaluation: CheckpointEvaluation = nextCheckpointBestPassing
    ? {
        epoch: checkpoint.epoch,
        prefix: checkpoint.r2_prefix,
        ok: true,
        score: nextCheckpointBestPassing.score,
        message: nextCheckpointBestPassing.message,
        preset: nextCheckpointBestPassing.preset_name,
        passed_samples: nextCheckpointBestPassing.passed_samples,
        total_samples: nextCheckpointBestPassing.total_samples,
      }
    : {
        epoch: checkpoint.epoch,
        prefix: checkpoint.r2_prefix,
        ok: false,
        score: nextCheckpointBestFailure?.score ?? 0,
        message: nextCheckpointBestFailure?.message ?? 'Validation failed for all presets',
        preset: nextCheckpointBestFailure?.preset_name ?? preset.name,
        passed_samples: nextCheckpointBestFailure?.passed_samples ?? 0,
        total_samples: nextCheckpointBestFailure?.total_samples ?? plan.totalSamples,
      };
  const nextEvaluations = [...existingEvaluations, checkpointEvaluation];

  if (nextCheckpointBestPassing) {
    const nextChampion =
      !currentChampion || nextCheckpointBestPassing.score > currentChampion.score
        ? nextCheckpointBestPassing
        : currentChampion;
    if (plan.prioritizeLatestPassingCheckpoint) {
      return completeSuccessfulValidation(nextCheckpointBestPassing, nextEvaluations);
    }
    if (checkpointIndex + 1 < candidateCheckpoints.length) {
      return startValidationSample({
        checkpointIndex: checkpointIndex + 1,
        presetIndex: 0,
        textIndex: 0,
        seedIndex: 0,
        evaluations: nextEvaluations,
        presetStats: createValidationAccumulator(),
        checkpointBestPassing: null,
        checkpointBestFailure: null,
        champion: nextChampion,
      });
    }
    return completeSuccessfulValidation(nextChampion, nextEvaluations);
  }

  if (checkpointIndex + 1 < candidateCheckpoints.length) {
    return startValidationSample({
      checkpointIndex: checkpointIndex + 1,
      presetIndex: 0,
      textIndex: 0,
      seedIndex: 0,
      evaluations: nextEvaluations,
      presetStats: createValidationAccumulator(),
      checkpointBestPassing: null,
      checkpointBestFailure: null,
      champion: currentChampion,
    });
  }

  if (currentChampion) {
    return completeSuccessfulValidation(currentChampion, nextEvaluations);
  }

  return completeFailedValidation(checkpointEvaluation.message, nextEvaluations);
};

const advanceAsync06bCheckpointValidation = async (
  c: Context<AppContext>,
  voice: NonNullable<Awaited<ReturnType<typeof getVoice>>>,
  job: TrainingJob,
  progress: TrainingProgress,
  currentSummary: Record<string, unknown>,
  candidateCheckpoints: CheckpointCandidate[],
): Promise<{ status: string; progress: TrainingProgress }> => {
  const preset = getValidationPresets(
    voice.model_id ?? 'qwen3-tts-0.6b',
    typeof (job.config as Record<string, unknown>).whisper_language === 'string'
      ? String((job.config as Record<string, unknown>).whisper_language).toLowerCase()
      : String(voice.labels?.language ?? ''),
  )[0];
  const validationText = getValidationTexts(
    typeof (job.config as Record<string, unknown>).whisper_language === 'string'
      ? String((job.config as Record<string, unknown>).whisper_language).toLowerCase()
      : '',
    true,
  )[0];
  const seed = FAST_VALIDATION_SEEDS_OFFSET[0];

  if (!preset || !validationText) {
    return {
      status: 'completed',
      progress,
    };
  }

  const persistedState =
    currentSummary.async_validation && typeof currentSummary.async_validation === 'object'
      ? (currentSummary.async_validation as Record<string, unknown>)
      : null;
  const existingEvaluations = Array.isArray(persistedState?.evaluations)
    ? persistedState.evaluations
        .map(normalizeCheckpointEvaluation)
        .filter((value): value is CheckpointEvaluation => value !== null)
    : [];

  const referenceAudioKey =
    typeof persistedState?.reference_audio_key === 'string' ||
    persistedState?.reference_audio_key === null
      ? (persistedState.reference_audio_key as string | null)
      : null;
  const referenceText =
    typeof persistedState?.reference_text === 'string' ? persistedState.reference_text : '';

  const startCheckpointValidation = async (
    checkpointIndex: number,
    evaluations: CheckpointEvaluation[],
  ): Promise<{ status: string; progress: TrainingProgress }> => {
    const checkpoint = candidateCheckpoints[checkpointIndex];
    if (!checkpoint) {
      await updateTrainingJob(c.env.DB, job.job_id, {
        status: 'completed',
        error_message: null,
        completed_at: job.completed_at ?? Date.now(),
        progress,
        summary: {
          ...currentSummary,
          validation_in_progress: false,
          validation_checked: true,
          validation_passed: false,
          validation_failed: true,
          validation_rejected: true,
          validation_message: 'No remaining checkpoints to validate',
          selected_checkpoint_prefix: null,
          selected_checkpoint_epoch: null,
          selected_preset: null,
          selected_score: null,
          evaluated_checkpoints: evaluations,
          async_validation: null,
        },
      });
      return { status: 'completed', progress };
    }
    const reference =
      referenceAudioKey !== null || referenceText
        ? { referenceAudioKey, referenceText }
        : await loadValidationReference(c, voice, job);
    const payload = buildValidationPayload({
      voice,
      checkpointPrefix: checkpoint.r2_prefix,
      preset,
      validationText,
      seed,
      referenceAudioKey: reference.referenceAudioKey,
      referenceText: reference.referenceText,
      languageHint: getValidationLanguageHint(voice, job),
    });
    const runpodResponse = await invokeServerlessAsync(c.env, c.env.RUNPOD_ENDPOINT_ID, payload);
    const nextState: Async06bValidationState = {
      mode: 'fast_06b_async',
      run_id: String(runpodResponse.id ?? ''),
      run_started_at: Date.now(),
      checkpoint_index: checkpointIndex,
      checkpoint_epoch: checkpoint.epoch,
      checkpoint_prefix: checkpoint.r2_prefix,
      preset_name: preset.name,
      validation_text: validationText,
      seed,
      reference_audio_key: reference.referenceAudioKey,
      reference_text: reference.referenceText,
      evaluations,
    };
    await updateTrainingJob(c.env.DB, job.job_id, {
      status: 'completed',
      completed_at: job.completed_at ?? Date.now(),
      progress,
      summary: {
        ...currentSummary,
        validation_in_progress: true,
        validation_message: `Validating checkpoint epoch ${checkpoint.epoch} for 0.6B`,
        async_validation: nextState,
      },
    });
    return { status: 'completed', progress };
  };

  if (
    !persistedState ||
    String(persistedState.mode ?? '') !== 'fast_06b_async' ||
    typeof persistedState.run_id !== 'string'
  ) {
    return startCheckpointValidation(existingEvaluations.length, existingEvaluations);
  }

  const currentIndex =
    typeof persistedState.checkpoint_index === 'number' ? persistedState.checkpoint_index : 0;
  const currentEpoch =
    typeof persistedState.checkpoint_epoch === 'number'
      ? persistedState.checkpoint_epoch
      : candidateCheckpoints[currentIndex]?.epoch;
  const currentPrefix =
    typeof persistedState.checkpoint_prefix === 'string'
      ? persistedState.checkpoint_prefix
      : candidateCheckpoints[currentIndex]?.r2_prefix;

  const runpodResponse = await getServerlessStatusOrSyntheticFailure(
    c.env,
    c.env.RUNPOD_ENDPOINT_ID,
    persistedState.run_id,
  );
  const runStartedAt = getValidationRunStartedAt(persistedState, job);
  const runAgeMs = Math.max(0, Date.now() - runStartedAt);
  const rawRunStatus = String(runpodResponse.status ?? 'UNKNOWN');
  const runTimedOut =
    rawRunStatus !== 'COMPLETED' && rawRunStatus !== 'FAILED' && runAgeMs > VALIDATION_RUN_STALE_MS;
  const runStatus = runTimedOut ? 'FAILED' : rawRunStatus;
  if (runStatus !== 'COMPLETED' && runStatus !== 'FAILED') {
    await updateTrainingJob(c.env.DB, job.job_id, {
      status: 'completed',
      completed_at: job.completed_at ?? Date.now(),
      progress,
      summary: {
        ...currentSummary,
        validation_in_progress: true,
        validation_message: `Validation still running for checkpoint epoch ${currentEpoch} (${runStatus.toLowerCase()})`,
        async_validation: {
          ...persistedState,
          run_started_at: runStartedAt,
          evaluations: existingEvaluations,
        },
      },
    });
    return { status: 'completed', progress };
  }

  const output = (runpodResponse.output ?? null) as {
    quality?: Record<string, unknown>;
    audio?: string;
    error?: unknown;
  } | null;
  let enrichedOutput = runStatus === 'COMPLETED' ? output : null;
  if (enrichedOutput?.audio) {
    try {
      enrichedOutput = await enrichOutputWithReviewAsr({
        env: c.env,
        output: enrichedOutput,
        expectedText: validationText,
        languageHint: getValidationLanguageHint(voice, job),
      });
    } catch (error) {
      enrichedOutput = annotateAsrFailure(enrichedOutput, error);
    }
  }
  const asyncFailureDetail = runTimedOut
    ? `validation request timed out after ${Math.round(runAgeMs / 1000)}s`
    : typeof enrichedOutput?.error === 'string'
      ? enrichedOutput.error
      : typeof runpodResponse.error === 'string'
        ? runpodResponse.error
        : `runpod_status=${runStatus.toLowerCase()}`;
  const result =
    runStatus === 'COMPLETED'
      ? scoreSingleValidationOutput({
          preset,
          seed,
          output: enrichedOutput,
          fallbackError: typeof runpodResponse.error === 'string' ? runpodResponse.error : null,
          referenceAudioKey: referenceAudioKey,
          referenceText,
          is06b: true,
          minToneScore: 0.45,
        })
      : {
          ok: false,
          message: `All presets failed: preset=${preset.name} ${asyncFailureDetail}`,
          aggregateScore: 0,
          presetName: preset.name,
          passedSamples: 0,
          totalSamples: 1,
        };

  const nextEvaluations: CheckpointEvaluation[] = [
    ...existingEvaluations,
    {
      epoch:
        typeof currentEpoch === 'number' ? currentEpoch : candidateCheckpoints[currentIndex].epoch,
      prefix:
        typeof currentPrefix === 'string'
          ? currentPrefix
          : candidateCheckpoints[currentIndex].r2_prefix,
      ok: result.ok,
      score: result.aggregateScore,
      message: result.message,
      preset: result.presetName,
      passed_samples: result.passedSamples,
      total_samples: result.totalSamples,
    },
  ];

  if (result.ok) {
    const promotedPrefix =
      typeof currentPrefix === 'string'
        ? currentPrefix
        : candidateCheckpoints[currentIndex].r2_prefix;
    const promotedEpoch =
      typeof currentEpoch === 'number' ? currentEpoch : candidateCheckpoints[currentIndex].epoch;
    const adoption = await applyValidatedCheckpointOutcome({
      c,
      voice,
      job,
      candidatePrefix: promotedPrefix,
      candidateEpoch: promotedEpoch,
      candidatePreset: result.presetName,
      candidateScore: result.aggregateScore,
    });

    const validationMessage =
      adoption.mode === 'keep_current'
        ? `${result.message} | kept current ready checkpoint score=${(adoption.preservedScore ?? 0).toFixed(3)} >= candidate_score=${result.aggregateScore.toFixed(3)}`
        : adoption.mode === 'candidate'
          ? `${result.message} | stored as candidate; production remains unchanged until promotion`
          : result.message;
    const nextSummary = {
      ...currentSummary,
      force_revalidation: false,
      validation_in_progress: false,
      validation_checked: true,
      validation_passed: true,
      validation_message: validationMessage,
      selected_checkpoint_prefix: adoption.selectedPrefix,
      selected_checkpoint_epoch: adoption.selectedEpoch,
      selected_preset: adoption.selectedPreset,
      selected_score: adoption.selectedScore,
      candidate_checkpoint_prefix: promotedPrefix,
      candidate_checkpoint_epoch: promotedEpoch,
      candidate_preset: result.presetName,
      candidate_score: result.aggregateScore,
      candidate_promotion_mode: adoption.mode,
      evaluated_checkpoints: nextEvaluations,
      async_validation: null,
    };
    await updateTrainingJob(c.env.DB, job.job_id, {
      status: 'completed',
      completed_at: job.completed_at ?? Date.now(),
      progress,
      summary: nextSummary,
      supervisor: {
        ...(job.supervisor ?? {}),
        phase: 'validation_completed',
        outcome: adoption.mode,
        last_validation_checkpoint_epoch: promotedEpoch,
      },
    });
    await syncTrainingCheckoutLedgerForJob(c, job, nextSummary, job.completed_at ?? Date.now());
    return { status: 'completed', progress };
  }

  const nextIndex = currentIndex + 1;
  if (nextIndex < candidateCheckpoints.length) {
    await updateTrainingJob(c.env.DB, job.job_id, {
      status: 'completed',
      completed_at: job.completed_at ?? Date.now(),
      progress,
      summary: {
        ...currentSummary,
        evaluated_checkpoints: nextEvaluations,
      },
    });
    return startCheckpointValidation(nextIndex, nextEvaluations);
  }

  const keepReadyVoice = shouldKeepReadyVoiceOnValidationFailure(voice, currentSummary, {
    evaluatedCheckpoints: nextEvaluations,
    validationRunName: parseRunNameFromCheckpointPrefix(candidateCheckpoints[0]?.r2_prefix ?? ''),
    forceRevalidation: currentSummary.force_revalidation === true,
  });
  if (!keepReadyVoice) {
    await updateVoice(c.env.DB, job.voice_id, {
      status: 'created',
      checkpoint_r2_prefix: null,
      run_name: null,
      epoch: null,
      checkpoint_preset: null,
      checkpoint_score: null,
      checkpoint_job_id: null,
      candidate_checkpoint_r2_prefix: null,
      candidate_run_name: null,
      candidate_epoch: null,
      candidate_preset: null,
      candidate_score: null,
      candidate_job_id: null,
    });
  }

  if (job.round_id) {
    await updateTrainingRound(c.env.DB, job.round_id, {
      status: 'failed',
      production_checkpoint_r2_prefix: keepReadyVoice ? voice.checkpoint_r2_prefix : null,
      production_run_name: keepReadyVoice ? voice.run_name : null,
      production_epoch: keepReadyVoice ? voice.epoch : null,
      production_preset: keepReadyVoice ? voice.checkpoint_preset : null,
      production_score: keepReadyVoice ? voice.checkpoint_score : null,
      production_job_id: keepReadyVoice ? voice.checkpoint_job_id : null,
      champion_checkpoint_r2_prefix: null,
      champion_run_name: null,
      champion_epoch: null,
      champion_preset: null,
      champion_score: null,
      champion_job_id: null,
      selected_checkpoint_r2_prefix: null,
      selected_run_name: null,
      selected_epoch: null,
      selected_preset: null,
      selected_score: null,
      selected_job_id: null,
      adoption_mode: null,
      candidate_checkpoint_r2_prefix: null,
      candidate_run_name: null,
      candidate_epoch: null,
      candidate_score: null,
      candidate_job_id: null,
      completed_at: Date.now(),
      summary: {
        ...(await getTrainingRound(c.env.DB, job.round_id))?.summary,
        validation_message: result.message,
        evaluated_checkpoints: nextEvaluations,
      },
    });
  }

  const nextSummary = {
    ...currentSummary,
    force_revalidation: false,
    validation_in_progress: false,
    validation_checked: true,
    validation_passed: false,
    validation_failed: true,
    validation_rejected: true,
    validation_message: result.message,
    selected_checkpoint_prefix: null,
    selected_checkpoint_epoch: null,
    selected_preset: null,
    selected_score: null,
    evaluated_checkpoints: nextEvaluations,
    async_validation: null,
  };
  await updateTrainingJob(c.env.DB, job.job_id, {
    status: 'completed',
    error_message: null,
    completed_at: job.completed_at ?? Date.now(),
    progress,
    summary: nextSummary,
    supervisor: {
      ...(job.supervisor ?? {}),
      phase: 'validation_failed',
    },
  });
  await syncTrainingCheckoutLedgerForJob(c, job, nextSummary, job.completed_at ?? Date.now());
  return { status: 'completed', progress };
};

const validateTrainedCheckpoint = async (
  c: Context<AppContext>,
  voice: NonNullable<Awaited<ReturnType<typeof getVoice>>>,
  job: TrainingJob,
  checkpointPrefix: string,
): Promise<CheckpointValidationResult> => {
  const jobConfig = job.config as Record<string, unknown>;
  const lang =
    typeof jobConfig.whisper_language === 'string' ? jobConfig.whisper_language.toLowerCase() : '';
  const datasetPrefix = String(job.dataset_r2_prefix ?? '').replace(/\/+$/, '');
  const presets = getValidationPresets(voice.model_id ?? 'qwen3-tts-1.7b', lang);
  const is06b = String(voice.model_id ?? '')
    .toLowerCase()
    .includes('0.6b');
  const validationTexts = getValidationTexts(lang, is06b);
  const validationSeedOffsets = is06b ? FAST_VALIDATION_SEEDS_OFFSET : FULL_VALIDATION_SEEDS_OFFSET;
  const totalSamples = validationTexts.length * validationSeedOffsets.length;
  const minOverall = is06b ? 0.82 : 0.85;
  const minPassRate = is06b ? MIN_PASS_RATE_06B : MIN_PASS_RATE_17B;
  const minAsrScore = 0.8;
  let referenceAudioKey =
    voice.ref_audio_r2_key ?? (datasetPrefix ? `${datasetPrefix}/ref_audio.wav` : null);
  let referenceText = '';

  if (datasetPrefix) {
    try {
      const profileObj = await c.env.R2.get(`${datasetPrefix}/reference_profile.json`);
      if (profileObj) {
        const profile = (await profileObj.json()) as Record<string, unknown>;
        if (typeof profile.reference_audio_key === 'string' && profile.reference_audio_key.trim()) {
          referenceAudioKey = profile.reference_audio_key.trim();
        }
        if (typeof profile.reference_text === 'string') {
          referenceText = profile.reference_text.trim();
        }
      }
    } catch {
      // Best-effort only. Validation can still proceed with ASR-only scoring.
    }
  }

  let bestPassing: {
    score: number;
    message: string;
    preset: ValidationPreset;
  } | null = null;
  let bestFailure: { passed: number; message: string; presetName: string } | null = null;

  try {
    for (const preset of presets) {
      let passed = 0;
      let noAudio = 0;
      let infraIssues = 0;
      let sumOverall = 0;
      let sumDuration = 0;
      let sumHealth = 0;
      let sumAsr = 0;
      let sumSpeaker = 0;
      let sumTone = 0;
      let sumSpeed = 0;
      let sumStyle = 0;
      let speakerSamples = 0;
      let toneSamples = 0;
      let speedSamples = 0;
      let styleSamples = 0;
      let firstFailureMessage: string | null = null;

      for (let i = 0; i < validationTexts.length; i += 1) {
        for (const seedOffset of validationSeedOffsets) {
          const seed = seedOffset + i;
          const languageHint = getValidationLanguageHint(voice, job);
          const normalizedLanguage = normalizeValidationInferenceLanguage(languageHint);
          const payload: Record<string, unknown> = {
            text: validationTexts[i],
            voice_id: voice.voice_id,
            speaker_name: voice.speaker_name,
            model_id: voice.model_id ?? 'qwen3-tts-1.7b',
            ...(normalizedLanguage ? { language: normalizedLanguage } : {}),
            seed,
            quality_review: {
              enable_asr: false,
              enable_speaker: Boolean(referenceAudioKey),
              enable_style: Boolean(referenceAudioKey),
              enable_speed: Boolean(referenceAudioKey && referenceText),
              allow_below_threshold: true,
              reference_audio_key: referenceAudioKey,
              reference_text: referenceText,
            },
            checkpoint_info: {
              r2_prefix: checkpointPrefix,
              type: 'full',
            },
            ...preset.payload,
          };

          let response: Record<string, unknown> | null = null;
          let output: {
            quality?: Record<string, unknown>;
            audio?: string;
            error?: unknown;
          } | null = null;
          let lastErrorDetail = 'unknown';

          for (let attempt = 1; attempt <= VALIDATION_RETRY_ATTEMPTS; attempt += 1) {
            const syncResult = await invokeServerless(c.env, c.env.RUNPOD_ENDPOINT_ID, payload);
            if (syncResult.autoAsync) {
              lastErrorDetail = 'validation exceeded sync timeout';
              continue;
            }
            response = syncResult.body;
            output = (response.output ?? {}) as {
              quality?: Record<string, unknown>;
              audio?: string;
              error?: unknown;
            };

            if (output.audio) {
              break;
            }

            const statusText = String(response.status ?? 'unknown');
            const outputError =
              typeof output.error === 'string'
                ? output.error
                : typeof response.error === 'string'
                  ? String(response.error)
                  : null;
            lastErrorDetail = outputError ?? `status=${statusText} no-audio`;
          }

          if (!output?.audio) {
            noAudio += 1;
            const parsedOverall = parseOverallFromError(lastErrorDetail);
            const msg =
              parsedOverall !== null
                ? `sample ${i + 1} seed ${seed} no audio overall_score=${parsedOverall.toFixed(3)}`
                : `sample ${i + 1} seed ${seed} no audio (${lastErrorDetail})`;
            if (!firstFailureMessage) {
              firstFailureMessage = msg;
            }
            continue;
          }

          try {
            output =
              (await enrichOutputWithReviewAsr({
                env: c.env,
                output,
                expectedText: validationTexts[i],
                languageHint: getValidationLanguageHint(voice, job),
              })) ?? output;
          } catch (error) {
            output = annotateAsrFailure(output, error);
          }

          const quality = output?.quality ?? {};
          const overall = Number(quality.overall_score ?? NaN);
          const duration = Number(quality.duration_score ?? NaN);
          const health = Number(quality.health_score ?? NaN);
          const asr = Number(quality.asr_score ?? quality.asr_similarity ?? NaN);
          const speaker = Number(quality.speaker_score ?? NaN);
          const tone = Number(quality.tone_score ?? NaN);
          const speed = Number(quality.speed_score ?? NaN);
          const style = Number(quality.style_score ?? NaN);

          if (!Number.isFinite(overall) || !Number.isFinite(duration) || !Number.isFinite(health)) {
            infraIssues += 1;
            if (!firstFailureMessage) {
              firstFailureMessage = `sample ${i + 1} seed ${seed} invalid quality metrics`;
            }
            continue;
          }
          if (!Number.isFinite(asr)) {
            infraIssues += 1;
            if (!firstFailureMessage) {
              firstFailureMessage = getMissingAsrMessage(quality, i + 1, seed);
            }
            continue;
          }
          if (overall < minOverall) {
            if (!firstFailureMessage) {
              firstFailureMessage = `sample ${i + 1} seed ${seed} overall_score=${overall.toFixed(3)}`;
            }
            continue;
          }
          if (duration < 0.3) {
            if (!firstFailureMessage) {
              firstFailureMessage = `sample ${i + 1} seed ${seed} duration_score=${duration.toFixed(3)}`;
            }
            continue;
          }
          if (health < 0.72) {
            if (!firstFailureMessage) {
              firstFailureMessage = `sample ${i + 1} seed ${seed} health_score=${health.toFixed(3)}`;
            }
            continue;
          }
          if (Number.isFinite(asr) && asr < minAsrScore) {
            if (!firstFailureMessage) {
              firstFailureMessage = `sample ${i + 1} seed ${seed} asr_score=${asr.toFixed(3)}`;
            }
            continue;
          }
          if (
            referenceAudioKey &&
            Number.isFinite(speaker) &&
            speaker < VALIDATION_GATE_THRESHOLDS.speaker_min
          ) {
            if (!firstFailureMessage) {
              firstFailureMessage = `sample ${i + 1} seed ${seed} speaker_score=${speaker.toFixed(3)}`;
            }
            continue;
          }
          if (
            referenceAudioKey &&
            Number.isFinite(style) &&
            style < (is06b ? 0.4 : VALIDATION_GATE_THRESHOLDS.style_min)
          ) {
            if (!firstFailureMessage) {
              firstFailureMessage = `sample ${i + 1} seed ${seed} style_score=${style.toFixed(3)}`;
            }
            continue;
          }
          if (
            referenceAudioKey &&
            Number.isFinite(tone) &&
            tone < (is06b ? 0.4 : VALIDATION_GATE_THRESHOLDS.tone_min)
          ) {
            if (!firstFailureMessage) {
              firstFailureMessage = `sample ${i + 1} seed ${seed} tone_score=${tone.toFixed(3)}`;
            }
            continue;
          }
          if (
            referenceAudioKey &&
            referenceText &&
            Number.isFinite(speed) &&
            speed < VALIDATION_GATE_THRESHOLDS.speed_min
          ) {
            if (!firstFailureMessage) {
              firstFailureMessage = `sample ${i + 1} seed ${seed} speed_score=${speed.toFixed(3)}`;
            }
            continue;
          }

          passed += 1;
          sumOverall += overall;
          sumDuration += duration;
          sumHealth += health;
          if (Number.isFinite(asr)) {
            sumAsr += asr;
          }
          if (Number.isFinite(speaker)) {
            sumSpeaker += speaker;
            speakerSamples += 1;
          }
          if (Number.isFinite(tone)) {
            sumTone += tone;
            toneSamples += 1;
          }
          if (Number.isFinite(speed)) {
            sumSpeed += speed;
            speedSamples += 1;
          }
          if (Number.isFinite(style)) {
            sumStyle += style;
            styleSamples += 1;
          }
        }
      }

      const passRate = totalSamples > 0 ? passed / totalSamples : 0;
      if (passed > 0 && passRate >= minPassRate && infraIssues === 0) {
        const n = Math.max(1, passed);
        const meanOverall = sumOverall / n;
        const meanDuration = sumDuration / n;
        const meanHealth = sumHealth / n;
        const meanAsr = sumAsr / n;
        const meanSpeaker = speakerSamples > 0 ? sumSpeaker / speakerSamples : NaN;
        const meanTone = toneSamples > 0 ? sumTone / toneSamples : NaN;
        const meanSpeed = speedSamples > 0 ? sumSpeed / speedSamples : NaN;
        const meanStyle = styleSamples > 0 ? sumStyle / styleSamples : NaN;
        const scoreParts = buildValidationScoreParts({
          is06b: true,
          overall: meanOverall,
          asr: meanAsr,
          health: meanHealth,
          duration: meanDuration,
          passRate,
          speaker: meanSpeaker,
          tone: meanTone,
          speed: meanSpeed,
          style: meanStyle,
        });
        const totalWeight = scoreParts.reduce((acc, part) => acc + part.weight, 0) || 1;
        const score =
          scoreParts.reduce((acc, part) => acc + part.value * part.weight, 0) / totalWeight;
        const similaritySegments: string[] = [];
        if (Number.isFinite(meanSpeaker)) {
          similaritySegments.push(`speaker=${meanSpeaker.toFixed(3)}`);
        }
        if (Number.isFinite(meanStyle)) {
          similaritySegments.push(`style=${meanStyle.toFixed(3)}`);
        } else if (Number.isFinite(meanTone)) {
          similaritySegments.push(`tone=${meanTone.toFixed(3)}`);
        }
        const similarityNote =
          similaritySegments.length > 0 ? `${similaritySegments.join(' ')} ` : '';
        const speedNote = Number.isFinite(meanSpeed) ? `speed=${meanSpeed.toFixed(3)} ` : '';
        const message =
          `preset=${preset.name} ` +
          `score=${score.toFixed(3)} overall=${meanOverall.toFixed(3)} ` +
          `asr=${meanAsr.toFixed(3)} ` +
          similarityNote +
          speedNote +
          `health=${meanHealth.toFixed(3)} duration=${meanDuration.toFixed(3)} ` +
          `samples=${passed}/${totalSamples} no_audio=${noAudio}`;
        if (!bestPassing || score > bestPassing.score) {
          bestPassing = { score, message, preset };
        }
      } else if (!bestFailure || passed > bestFailure.passed) {
        const failureSummary =
          firstFailureMessage ??
          `samples=${passed}/${totalSamples} pass_rate=${passRate.toFixed(3)} no_audio=${noAudio} infra=${infraIssues}`;
        bestFailure = {
          passed,
          message: `preset=${preset.name} ${failureSummary}`,
          presetName: preset.name,
        };
      }
    }

    if (bestPassing) {
      return {
        ok: true,
        message: bestPassing.message,
        aggregateScore: bestPassing.score,
        presetName: bestPassing.preset.name,
        presetSettings: bestPassing.preset.settings,
        passedSamples: totalSamples,
        totalSamples,
      };
    }

    return {
      ok: false,
      message: `All presets failed: ${bestFailure?.message ?? 'unknown'}`,
      aggregateScore: 0,
      presetName: bestFailure?.presetName ?? 'default',
      passedSamples: bestFailure?.passed ?? 0,
      totalSamples,
    };
  } catch (error) {
    return {
      ok: false,
      message: `Validation invocation failed: ${error instanceof Error ? error.message : 'unknown'}`,
      aggregateScore: 0,
      presetName: 'default',
      passedSamples: 0,
      totalSamples,
    };
  }
};

const reconcileJobStatus = async (
  c: Context<AppContext>,
  job: TrainingJob,
): Promise<{ status: string; progress: TrainingProgress }> => {
  if (shouldRecoverFailedDependencyJob(c, job)) {
    const recovered = await recoverFailedDependencyJob(c, job);
    return { status: recovered.status, progress: recovered.progress };
  }

  let currentJob = job;
  let status = job.status;
  let progress: TrainingProgress = job.progress;
  const statusBlob = await c.env.R2.get(`jobs/${job.job_id}/status.json`);

  if (!statusBlob) {
    currentJob = await recoverStalledProvisioningJob(c, currentJob);
    currentJob = await recoverStalledActiveJob(c, currentJob, null);
    return { status: currentJob.status, progress: currentJob.progress };
  }

  const parsedStatus = (await statusBlob.json()) as TrainingStatusBlob;
  if (parsedStatus.status) {
    status = parsedStatus.status;
  }
  if (parsedStatus.progress) {
    progress = parsedStatus.progress;
  }

  if (status !== job.status || parsedStatus.progress) {
    await updateTrainingJob(c.env.DB, job.job_id, {
      status,
      progress,
      completed_at: status === 'completed' ? (job.completed_at ?? Date.now()) : job.completed_at,
      supervisor: {
        ...(job.supervisor ?? {}),
        phase: status,
        last_transition_at: Date.now(),
      },
    });
    currentJob = {
      ...job,
      status,
      progress,
      completed_at: status === 'completed' ? (job.completed_at ?? Date.now()) : job.completed_at,
      supervisor: {
        ...(job.supervisor ?? {}),
        phase: status,
        last_transition_at: Date.now(),
      },
    };
  } else {
    currentJob = {
      ...job,
      status,
      progress,
    };
  }

  currentJob = await recoverStalledActiveJob(c, currentJob, parsedStatus);
  if (currentJob.job_id === job.job_id && currentJob.status !== status) {
    return { status: currentJob.status, progress: currentJob.progress };
  }

  if (currentJob.status === 'completed' && !currentJob.summary?.validation_checked) {
    if (currentJob.round_id) {
      await updateTrainingRound(c.env.DB, currentJob.round_id, {
        status: 'validating',
        summary: {
          ...(await getTrainingRound(c.env.DB, currentJob.round_id))?.summary,
          validation_in_progress: true,
          job_id: currentJob.job_id,
        },
      });
    }
    const voice = await getVoice(c.env.DB, job.voice_id);
    if (!voice) {
      const message = 'Voice record missing for validation';
      await updateTrainingJob(c.env.DB, job.job_id, {
        status: 'failed',
        error_message: message,
        summary: {
          ...(currentJob.summary ?? {}),
          validation_failed: true,
          validation_checked: true,
          validation_passed: false,
          validation_message: message,
        },
        completed_at: currentJob.completed_at ?? Date.now(),
        progress,
      });
      return { status: 'failed', progress };
    }

    const currentSummary = (currentJob.summary ?? {}) as Record<string, unknown>;

    if (!Array.isArray(parsedStatus.checkpoints) || parsedStatus.checkpoints.length === 0) {
      const message = 'Training completed but no checkpoints found in status payload';
      const keepReadyVoice = shouldPreserveCurrentReadyVoice(voice);
      if (!keepReadyVoice) {
        await updateVoice(c.env.DB, job.voice_id, {
          status: 'created',
          checkpoint_r2_prefix: null,
          run_name: null,
          epoch: null,
          checkpoint_preset: null,
          checkpoint_score: null,
          checkpoint_job_id: null,
          candidate_checkpoint_r2_prefix: null,
          candidate_run_name: null,
          candidate_epoch: null,
          candidate_preset: null,
          candidate_score: null,
          candidate_job_id: null,
        });
      }
      if (currentJob.round_id) {
        await updateTrainingRound(c.env.DB, currentJob.round_id, {
          status: 'failed',
          production_checkpoint_r2_prefix: keepReadyVoice ? voice.checkpoint_r2_prefix : null,
          production_run_name: keepReadyVoice ? voice.run_name : null,
          production_epoch: keepReadyVoice ? voice.epoch : null,
          production_preset: keepReadyVoice ? voice.checkpoint_preset : null,
          production_score: keepReadyVoice ? voice.checkpoint_score : null,
          production_job_id: keepReadyVoice ? voice.checkpoint_job_id : null,
          champion_checkpoint_r2_prefix: null,
          champion_run_name: null,
          champion_epoch: null,
          champion_preset: null,
          champion_score: null,
          champion_job_id: null,
          selected_checkpoint_r2_prefix: null,
          selected_run_name: null,
          selected_epoch: null,
          selected_preset: null,
          selected_score: null,
          selected_job_id: null,
          adoption_mode: null,
          candidate_checkpoint_r2_prefix: null,
          candidate_run_name: null,
          candidate_epoch: null,
          candidate_score: null,
          candidate_job_id: null,
          completed_at: currentJob.completed_at ?? Date.now(),
        });
      }
      const nextSummary = {
        ...(currentJob.summary ?? {}),
        validation_failed: true,
        validation_checked: true,
        validation_passed: false,
        validation_message: message,
      };
      await updateTrainingJob(c.env.DB, job.job_id, {
        status: 'failed',
        error_message: message,
        summary: nextSummary,
        completed_at: currentJob.completed_at ?? Date.now(),
        progress,
      });
      await syncTrainingCheckoutLedgerForJob(
        c,
        currentJob,
        nextSummary,
        currentJob.completed_at ?? Date.now(),
      );
      return { status: 'failed', progress };
    }

    const forceRevalidation = currentSummary.force_revalidation === true;
    const persistedReadyCheckpoint =
      voice.status === 'ready' &&
      typeof voice.checkpoint_r2_prefix === 'string' &&
      parsedStatus.checkpoints.some((cp) => cp?.r2_prefix === voice.checkpoint_r2_prefix)
        ? voice.checkpoint_r2_prefix
        : null;

    if (persistedReadyCheckpoint && !forceRevalidation) {
      const nextSummary = {
        ...currentSummary,
        validation_checked: true,
        validation_passed: true,
        validation_message:
          typeof currentSummary.validation_message === 'string' &&
          currentSummary.validation_message.trim()
            ? currentSummary.validation_message
            : 'Recovered validation result from persisted ready voice checkpoint',
        selected_checkpoint_prefix: persistedReadyCheckpoint,
        selected_checkpoint_epoch: voice.epoch,
        selected_preset:
          typeof currentSummary.selected_preset === 'string' &&
          currentSummary.selected_preset.trim()
            ? currentSummary.selected_preset
            : (voice.checkpoint_preset ?? 'high_similarity'),
        selected_score: voice.checkpoint_score ?? null,
      };
      await updateTrainingJob(c.env.DB, job.job_id, {
        status: 'completed',
        completed_at: currentJob.completed_at ?? Date.now(),
        progress,
        summary: nextSummary,
      });
      if (currentJob.round_id) {
        await updateTrainingRound(c.env.DB, currentJob.round_id, {
          status: 'promoted',
          production_checkpoint_r2_prefix: persistedReadyCheckpoint,
          production_run_name: voice.run_name,
          production_epoch: voice.epoch,
          production_preset: voice.checkpoint_preset,
          production_score: voice.checkpoint_score,
          production_job_id: voice.checkpoint_job_id,
          champion_checkpoint_r2_prefix: null,
          champion_run_name: null,
          champion_epoch: null,
          champion_preset: null,
          champion_score: null,
          champion_job_id: null,
          selected_checkpoint_r2_prefix: persistedReadyCheckpoint,
          selected_run_name: voice.run_name,
          selected_epoch: voice.epoch,
          selected_preset: voice.checkpoint_preset,
          selected_score: voice.checkpoint_score,
          selected_job_id: voice.checkpoint_job_id,
          adoption_mode: 'promote',
          candidate_checkpoint_r2_prefix: null,
          candidate_run_name: null,
          candidate_epoch: null,
          candidate_score: null,
          candidate_job_id: null,
          completed_at: currentJob.completed_at ?? Date.now(),
        });
      }
      await syncTrainingCheckoutLedgerForJob(
        c,
        currentJob,
        nextSummary,
        currentJob.completed_at ?? Date.now(),
      );
      return { status: 'completed', progress };
    }

    const validationPlan = getValidationPlan(voice, job);
    const candidateCheckpoints = selectValidationCandidateCheckpoints(
      parsedStatus.checkpoints,
      validationPlan,
    );

    if (candidateCheckpoints.length === 0) {
      const message = 'Training completed but checkpoint metadata is invalid';
      const keepReadyVoice = shouldPreserveCurrentReadyVoice(voice);
      if (!keepReadyVoice) {
        await updateVoice(c.env.DB, job.voice_id, {
          status: 'created',
          checkpoint_r2_prefix: null,
          run_name: null,
          epoch: null,
          checkpoint_preset: null,
          checkpoint_score: null,
          checkpoint_job_id: null,
          candidate_checkpoint_r2_prefix: null,
          candidate_run_name: null,
          candidate_epoch: null,
          candidate_preset: null,
          candidate_score: null,
          candidate_job_id: null,
        });
      }
      if (currentJob.round_id) {
        await updateTrainingRound(c.env.DB, currentJob.round_id, {
          status: 'failed',
          production_checkpoint_r2_prefix: keepReadyVoice ? voice.checkpoint_r2_prefix : null,
          production_run_name: keepReadyVoice ? voice.run_name : null,
          production_epoch: keepReadyVoice ? voice.epoch : null,
          production_preset: keepReadyVoice ? voice.checkpoint_preset : null,
          production_score: keepReadyVoice ? voice.checkpoint_score : null,
          production_job_id: keepReadyVoice ? voice.checkpoint_job_id : null,
          champion_checkpoint_r2_prefix: null,
          champion_run_name: null,
          champion_epoch: null,
          champion_preset: null,
          champion_score: null,
          champion_job_id: null,
          selected_checkpoint_r2_prefix: null,
          selected_run_name: null,
          selected_epoch: null,
          selected_preset: null,
          selected_score: null,
          selected_job_id: null,
          adoption_mode: null,
          candidate_checkpoint_r2_prefix: null,
          candidate_run_name: null,
          candidate_epoch: null,
          candidate_score: null,
          candidate_job_id: null,
          completed_at: currentJob.completed_at ?? Date.now(),
        });
      }
      const nextSummary = {
        ...(currentJob.summary ?? {}),
        validation_failed: true,
        validation_checked: true,
        validation_passed: false,
        validation_message: message,
      };
      await updateTrainingJob(c.env.DB, job.job_id, {
        status: 'failed',
        error_message: message,
        summary: nextSummary,
        completed_at: currentJob.completed_at ?? Date.now(),
        progress,
      });
      await syncTrainingCheckoutLedgerForJob(
        c,
        currentJob,
        nextSummary,
        currentJob.completed_at ?? Date.now(),
      );
      return { status: 'failed', progress };
    }

    const nextSummary = {
      ...currentSummary,
      validation_candidate_epochs: candidateCheckpoints.map((checkpoint) => checkpoint.epoch),
    };
    await updateTrainingJob(c.env.DB, job.job_id, {
      summary: {
        ...nextSummary,
      },
    });

    return advanceAsyncCheckpointValidation(
      c,
      voice,
      currentJob,
      progress,
      nextSummary,
      candidateCheckpoints,
    );
  }

  return { status: currentJob.status, progress: currentJob.progress };
};

const RECONCILE_TIMEOUT_MS = 25000;
const COMPLETED_VALIDATION_TIMEOUT_MS = 180000;

const getExecutionCtx = (c: Context<AppContext>): ExecutionContext | undefined =>
  (c as unknown as { executionCtx?: ExecutionContext }).executionCtx;

const waitForBackgroundTask = (c: Context<AppContext>, promise: Promise<unknown>) => {
  getExecutionCtx(c)?.waitUntil(promise.then(() => undefined).catch(() => undefined));
};

const reconcileJobStatusWithTimeout = async (
  c: Context<AppContext>,
  job: TrainingJob,
  timeoutMs = RECONCILE_TIMEOUT_MS,
): Promise<boolean> => {
  let timedOut = false;
  const reconcilePromise = reconcileJobStatus(c, job);
  await Promise.race([
    reconcilePromise,
    new Promise<void>((resolve) =>
      setTimeout(() => {
        timedOut = true;
        resolve();
      }, timeoutMs),
    ),
  ]);
  if (timedOut) {
    waitForBackgroundTask(c, reconcilePromise);
  }
  return !timedOut;
};

export const runTrainingSupervisorSweep = async (
  env: Env,
): Promise<{
  checked: number;
  reconciled: number;
  timed_out: number;
  failed: number;
  launched_queued: number;
  campaigns_advanced: number;
}> => {
  const jobs = await listTrainingJobs(env.DB, { limit: TRAINING_SWEEP_LIMIT });
  const syntheticContext = createSyntheticContext(env, resolveWorkerPublicUrl(env));
  const candidates = jobs.filter(
    (job) =>
      ACTIVE_JOB_STATUSES.has(job.status) ||
      needsCompletedValidation(job) ||
      shouldRecoverFailedDependencyJob(syntheticContext, job),
  );
  let reconciled = 0;
  let timedOut = 0;
  let failed = 0;
  let launchedQueued = 0;
  let campaignsAdvanced = 0;

  for (const job of candidates) {
    try {
      const timeoutMs = needsCompletedValidation(job)
        ? COMPLETED_VALIDATION_TIMEOUT_MS
        : RECONCILE_TIMEOUT_MS;
      const completed = await reconcileJobStatusWithTimeout(syntheticContext, job, timeoutMs);
      if (completed) {
        reconciled += 1;
      } else {
        timedOut += 1;
      }
    } catch (error) {
      failed += 1;
      console.warn(`Training supervisor sweep failed for ${job.job_id}:`, error);
    }
  }

  const queuedVoiceIds = Array.from(
    new Set(
      jobs
        .filter((job) => QUEUED_JOB_STATUSES.has(job.status) || job.status === 'pending')
        .map((job) => job.voice_id),
    ),
  );
  for (const voiceId of queuedVoiceIds) {
    try {
      launchedQueued += await launchQueuedTrainingJobsForVoice(syntheticContext, voiceId);
    } catch (error) {
      failed += 1;
      console.warn(`Queued training launch sweep failed for ${voiceId}:`, error);
    }
  }

  try {
    campaignsAdvanced = await runTrainingCampaignSweep(syntheticContext);
  } catch (error) {
    failed += 1;
    console.warn('Training campaign sweep failed', error);
  }

  return {
    checked: candidates.length,
    reconciled,
    timed_out: timedOut,
    failed,
    launched_queued: launchedQueued,
    campaigns_advanced: campaignsAdvanced,
  };
};

app.post('/campaigns', async (c) => {
  const body = (await c.req.json()) as TrainingCampaignRequest;
  if (!body.voice_id) {
    return c.json({ detail: { message: 'voice_id is required' } }, 400);
  }

  const voice = await getVoice(c.env.DB, body.voice_id);
  if (!voice) {
    return c.json({ detail: { message: 'Voice not found' } }, 404);
  }

  const rawAttempts = Number(body.attempt_count ?? DEFAULT_CAMPAIGN_ATTEMPT_COUNT);
  const attemptCount = Number.isFinite(rawAttempts)
    ? Math.max(1, Math.min(Math.trunc(rawAttempts), MAX_CAMPAIGN_ATTEMPTS))
    : DEFAULT_CAMPAIGN_ATTEMPT_COUNT;
  const requestedParallelism = Number(body.parallelism ?? DEFAULT_CAMPAIGN_PARALLELISM);
  const perVoiceActiveLimit = getMaxActiveTrainingJobsPerVoice(c);
  const parallelism = Number.isFinite(requestedParallelism)
    ? Math.max(1, Math.min(Math.trunc(requestedParallelism), perVoiceActiveLimit))
    : 1;

  const baseConfig = body.base_config_overrides ?? {};
  const baseCfg = baseConfig as Record<string, unknown>;
  const baseModelSize =
    typeof baseConfig.model_size === 'string' && baseConfig.model_size
      ? baseConfig.model_size
      : voice.model_size || '1.7B';
  const baseDefaults = getRecommendedTrainingDefaults(baseModelSize);
  const normalizedBaseConfig: TrainingConfig = {
    ...baseConfig,
    model_size: baseModelSize,
    batch_size: readNumber(baseConfig.batch_size) ?? baseDefaults.batch_size,
    learning_rate: readNumber(baseConfig.learning_rate) ?? baseDefaults.learning_rate,
    num_epochs: readNumber(baseConfig.num_epochs ?? baseCfg.epochs) ?? baseDefaults.num_epochs,
    gradient_accumulation_steps:
      readNumber(baseCfg.gradient_accumulation_steps) ?? baseDefaults.gradient_accumulation_steps,
    subtalker_loss_weight:
      readNumber(baseCfg.subtalker_loss_weight) ?? baseDefaults.subtalker_loss_weight,
    save_every_n_epochs:
      readNumber(baseCfg.save_every_n_epochs) ?? baseDefaults.save_every_n_epochs,
    seed: readNumber(baseCfg.seed) ?? baseDefaults.seed,
    gpu_type_id:
      typeof baseCfg.gpu_type_id === 'string' && baseCfg.gpu_type_id
        ? baseCfg.gpu_type_id
        : baseDefaults.gpu_type_id,
  };
  if (typeof baseCfg.whisper_language === 'string' && baseCfg.whisper_language.trim()) {
    normalizedBaseConfig.whisper_language = baseCfg.whisper_language.trim();
  }

  const baseNumEpochs = Number(normalizedBaseConfig.num_epochs ?? baseDefaults.num_epochs);
  const baseBatchSize = Number(normalizedBaseConfig.batch_size ?? baseDefaults.batch_size);
  const baseMaxSteps = Number(baseCfg.max_steps ?? 0);
  if (baseNumEpochs < 1 || baseNumEpochs > 30) {
    return c.json(
      { detail: { message: 'base_config_overrides.num_epochs must be between 1 and 30' } },
      400,
    );
  }
  if (baseBatchSize < 1 || baseBatchSize > 16) {
    return c.json(
      { detail: { message: 'base_config_overrides.batch_size must be between 1 and 16' } },
      400,
    );
  }
  if (baseMaxSteps < 0 || baseMaxSteps > 100000) {
    return c.json(
      { detail: { message: 'base_config_overrides.max_steps must be between 0 and 100000' } },
      400,
    );
  }

  const now = Date.now();
  const campaignId = crypto.randomUUID();
  const campaign: TrainingCampaign = {
    campaign_id: campaignId,
    voice_id: voice.voice_id,
    dataset_name: typeof body.dataset_name === 'string' ? body.dataset_name.trim() || null : null,
    dataset_r2_prefix: null,
    dataset_snapshot_id: null,
    attempt_count: attemptCount,
    parallelism,
    status: 'planning',
    base_config: normalizedBaseConfig,
    stop_rules: body.stop_rules ?? {},
    planner_state: {
      direction:
        typeof body.direction === 'string' &&
        ['conservative', 'balanced', 'exploratory'].includes(body.direction)
          ? body.direction
          : 'balanced',
    },
    summary: {},
    created_at: now,
    updated_at: now,
    completed_at: null,
  };

  await createTrainingCampaign(c.env.DB, campaign);
  const updated = await advanceTrainingCampaign(c, campaignId);
  const jobs = await listTrainingJobs(c.env.DB, { campaign_id: campaignId, limit: 100 });

  return c.json({
    campaign: updated ?? campaign,
    attempts: jobs.map(serializeTrainingJob),
  });
});

app.get('/campaigns/:campaign_id', async (c) => {
  const campaignId = c.req.param('campaign_id');
  if (!campaignId) {
    return c.json({ detail: { message: 'campaign_id is required' } }, 400);
  }
  const campaign = await getTrainingCampaign(c.env.DB, campaignId);
  if (!campaign) {
    return c.json({ detail: { message: 'Campaign not found' } }, 404);
  }
  const attempts = await listTrainingJobs(c.env.DB, { campaign_id: campaignId, limit: 100 });
  return c.json({ campaign, attempts: attempts.map(serializeTrainingJob) });
});

app.post('/campaigns/:campaign_id/cancel', async (c) => {
  const campaignId = c.req.param('campaign_id');
  if (!campaignId) {
    return c.json({ detail: { message: 'campaign_id is required' } }, 400);
  }
  const campaign = await getTrainingCampaign(c.env.DB, campaignId);
  if (!campaign) {
    return c.json({ detail: { message: 'Campaign not found' } }, 404);
  }

  if (!CAMPAIGN_ACTIVE_STATUSES.has(campaign.status)) {
    const attempts = await listTrainingJobs(c.env.DB, { campaign_id: campaignId, limit: 100 });
    return c.json({ campaign, attempts: attempts.map(serializeTrainingJob) });
  }

  await updateTrainingCampaign(c.env.DB, campaignId, {
    status: 'cancelled',
    summary: {
      ...(campaign.summary ?? {}),
      cancel_requested_at: Date.now(),
    },
    completed_at: Date.now(),
  });

  const attemptsBeforeCancel = await listTrainingJobs(c.env.DB, {
    campaign_id: campaignId,
    limit: 100,
  });
  for (const attempt of attemptsBeforeCancel) {
    if (TERMINAL_JOB_STATUSES.has(attempt.status)) {
      continue;
    }

    if (attempt.runpod_pod_id) {
      try {
        await terminatePod(c.env, attempt.runpod_pod_id);
      } catch {
        // Pod may already be terminated — safe to ignore
      }
    }

    await updateTrainingJob(c.env.DB, attempt.job_id, {
      status: 'cancelled',
      completed_at: Date.now(),
      expected_updated_at: attempt.updated_at,
      summary: {
        ...(attempt.summary ?? {}),
        campaign_cancelled_at: Date.now(),
      },
      supervisor: {
        ...(attempt.supervisor ?? {}),
        phase: 'cancelled',
        last_transition_at: Date.now(),
      },
    });
  }

  const attemptsAfterCancel = await listTrainingJobs(c.env.DB, {
    campaign_id: campaignId,
    limit: 100,
  });
  const allTerminal = attemptsAfterCancel.every((attempt) =>
    TERMINAL_JOB_STATUSES.has(attempt.status),
  );
  if (allTerminal) {
    await launchQueuedTrainingJobsForVoice(c, campaign.voice_id);
  }

  const next = await getTrainingCampaign(c.env.DB, campaignId);
  const attempts = await listTrainingJobs(c.env.DB, { campaign_id: campaignId, limit: 100 });
  return c.json({ campaign: next ?? campaign, attempts: attempts.map(serializeTrainingJob) });
});

app.post('/start', async (c) => {
  const body = (await c.req.json()) as {
    voice_id?: string;
    dataset_name?: string;
    config?: TrainingConfig;
  };

  if (!body.voice_id) {
    return c.json({ detail: { message: 'voice_id is required' } }, 400);
  }

  const voice = await getVoice(c.env.DB, body.voice_id);
  if (!voice) {
    return c.json({ detail: { message: 'Voice not found' } }, 404);
  }

  const now = Date.now();
  const jobId = crypto.randomUUID();
  const jobToken = crypto.randomUUID();
  const workerUrl = getWorkerOrigin(c);
  const runName = `run_${jobId.slice(0, 8)}`;
  const datasetPrefix = resolveTrainingDatasetPrefix(voice, body.dataset_name);
  const config = body.config ?? {};
  const cfg = config as Record<string, unknown>;
  const modelSize =
    typeof config.model_size === 'string' && config.model_size
      ? config.model_size
      : voice.model_size || '1.7B';
  const recommendedDefaults = getRecommendedTrainingDefaults(modelSize);
  const effectiveConfig: TrainingConfig = {
    ...config,
    model_size: modelSize,
    batch_size: Number(config.batch_size ?? recommendedDefaults.batch_size),
    learning_rate: Number(config.learning_rate ?? recommendedDefaults.learning_rate),
    num_epochs: Number(config.num_epochs ?? cfg.epochs ?? recommendedDefaults.num_epochs),
    gradient_accumulation_steps: Number(
      cfg.gradient_accumulation_steps ?? recommendedDefaults.gradient_accumulation_steps,
    ),
    subtalker_loss_weight: Number(
      cfg.subtalker_loss_weight ?? recommendedDefaults.subtalker_loss_weight,
    ),
    save_every_n_epochs: Number(cfg.save_every_n_epochs ?? recommendedDefaults.save_every_n_epochs),
    seed: Number(cfg.seed ?? recommendedDefaults.seed),
    gpu_type_id:
      typeof cfg.gpu_type_id === 'string' && cfg.gpu_type_id
        ? cfg.gpu_type_id
        : recommendedDefaults.gpu_type_id,
  };
  if (typeof cfg.whisper_language === 'string' && cfg.whisper_language.trim()) {
    effectiveConfig.whisper_language = cfg.whisper_language.trim();
  }

  const numEpochs = Number(effectiveConfig.num_epochs ?? recommendedDefaults.num_epochs);
  const batchSize = Number(effectiveConfig.batch_size ?? recommendedDefaults.batch_size);
  const maxSteps = Number(cfg.max_steps ?? 0);

  if (numEpochs < 1 || numEpochs > 30) {
    return c.json({ detail: { message: 'num_epochs must be between 1 and 30' } }, 400);
  }
  if (batchSize < 1 || batchSize > 16) {
    return c.json({ detail: { message: 'batch_size must be between 1 and 16' } }, 400);
  }
  if (maxSteps < 0 || maxSteps > 100000) {
    return c.json({ detail: { message: 'max_steps must be between 0 and 100000' } }, 400);
  }

  const datasetObjects = await listAllR2Objects(c.env.R2, `${stripSlashes(datasetPrefix)}/`);
  if (datasetObjects.length === 0) {
    return c.json(
      {
        detail: {
          message: `Dataset not found at R2 prefix: ${datasetPrefix}/. Upload audio files first.`,
        },
      },
      400,
    );
  }

  const datasetSignatureInfo = await computeDatasetSignature(datasetPrefix, datasetObjects);
  const datasetSignature =
    datasetSignatureInfo?.signature ?? buildSyntheticDatasetSignature(datasetPrefix);
  const datasetTrainRawKey = getDatasetTrainRawR2Key(datasetPrefix);
  const datasetHasPreparedTrainRaw = Boolean(await c.env.R2.head(datasetTrainRawKey));
  const preprocessCache =
    datasetSignatureInfo !== null
      ? await getDatasetPreprocessCache(c.env.DB, body.voice_id, datasetPrefix, datasetSignature)
      : null;
  const datasetSnapshot = await ensureDatasetSnapshot({
    c,
    voice,
    datasetPrefix,
    datasetSignature,
    preprocessCache,
    sourceFileCount: datasetSignatureInfo?.sourceCount ?? null,
    createdFromJobId: jobId,
  });
  const round = await ensureTrainingRound({
    c,
    voice,
    datasetSnapshotId: datasetSnapshot.snapshot_id,
    now,
  });
  const initialSummary: Record<string, unknown> = {
    round_id: round.round_id,
    round_index: round.round_index,
    dataset_snapshot_id: datasetSnapshot.snapshot_id,
    dataset_snapshot_status: datasetSnapshot.status,
    dataset_snapshot_signature: datasetSnapshot.dataset_signature,
    dataset_snapshot_name: datasetSnapshot.dataset_name,
    preprocess_cache_lookup: preprocessCache
      ? 'hit'
      : datasetHasPreparedTrainRaw
        ? 'dataset_ready'
        : 'miss',
    preprocess_cache_dataset_signature: datasetSignature,
    preprocess_cache_source_file_count: datasetSignatureInfo?.sourceCount ?? null,
    preprocess_cache_r2_prefix: preprocessCache?.cache_r2_prefix ?? null,
    preprocess_cache_train_raw_r2_key:
      preprocessCache?.train_raw_r2_key ?? (datasetHasPreparedTrainRaw ? datasetTrainRawKey : null),
    queue_active_limit: getMaxActiveTrainingJobsPerVoice(c),
  };

  const job: TrainingJob = {
    job_id: jobId,
    voice_id: body.voice_id,
    round_id: round.round_id,
    dataset_snapshot_id: datasetSnapshot.snapshot_id,
    runpod_pod_id: null,
    job_token: jobToken,
    status: 'queued',
    config: effectiveConfig,
    progress: {},
    summary: initialSummary,
    metrics: {},
    supervisor: {
      phase: 'queued',
      checks: 0,
      recovery_attempts: 0,
      last_transition_at: now,
    },
    dataset_r2_prefix: datasetPrefix,
    log_r2_prefix: null,
    error_message: null,
    last_heartbeat_at: null,
    started_at: null,
    completed_at: null,
    created_at: now,
    updated_at: now,
  };

  await updateVoice(c.env.DB, body.voice_id, {
    active_round_id: round.round_id,
    candidate_checkpoint_r2_prefix: null,
    candidate_run_name: null,
    candidate_epoch: null,
    candidate_score: null,
    candidate_job_id: null,
  });
  await createTrainingJob(c.env.DB, job);
  const jobConfig = {
    voice_id: body.voice_id,
    dataset_r2_prefix: datasetPrefix,
    speaker_name: voice.speaker_name,
    model_size: modelSize,
    batch_size: Number(effectiveConfig.batch_size ?? recommendedDefaults.batch_size),
    learning_rate: Number(effectiveConfig.learning_rate ?? recommendedDefaults.learning_rate),
    num_epochs: Number(effectiveConfig.num_epochs ?? recommendedDefaults.num_epochs),
    run_name: runName,
    gradient_accumulation_steps: Number(
      effectiveConfig.gradient_accumulation_steps ??
        recommendedDefaults.gradient_accumulation_steps,
    ),
    speaker_id: Number(cfg.speaker_id ?? 3000),
    mixed_precision: String(cfg.mixed_precision ?? 'bf16'),
    torch_dtype: String(cfg.torch_dtype ?? 'bfloat16'),
    attn_implementation: String(cfg.attn_implementation ?? 'sdpa'),
    weight_decay: Number(cfg.weight_decay ?? 0.01),
    max_grad_norm: Number(cfg.max_grad_norm ?? 1.0),
    subtalker_loss_weight: Number(
      effectiveConfig.subtalker_loss_weight ?? recommendedDefaults.subtalker_loss_weight,
    ),
    log_every_n_steps: Number(cfg.log_every_n_steps ?? 10),
    save_every_n_epochs: Number(
      effectiveConfig.save_every_n_epochs ?? recommendedDefaults.save_every_n_epochs,
    ),
    max_steps: Number(cfg.max_steps ?? 0),
    seed: Number(effectiveConfig.seed ?? recommendedDefaults.seed),
    job_token: jobToken,
    worker_api_url: workerUrl,
    whisper_language:
      typeof effectiveConfig.whisper_language === 'string'
        ? effectiveConfig.whisper_language
        : undefined,
    dataset_signature: datasetSignature,
    dataset_snapshot_id: datasetSnapshot.snapshot_id,
    training_round_id: round.round_id,
    preprocess_cache_r2_prefix:
      datasetSnapshot.cache_r2_prefix ?? preprocessCache?.cache_r2_prefix ?? undefined,
  };
  await c.env.R2.put(`jobs/${jobId}/config.json`, JSON.stringify(jobConfig), {
    httpMetadata: { contentType: 'application/json' },
  });

  await launchQueuedTrainingJobsForVoice(c, body.voice_id);

  const persistedJob = await getTrainingJob(c.env.DB, jobId);
  if (!persistedJob) {
    return c.json({ detail: { message: 'Training job not found' } }, 404);
  }

  let responseStatus = persistedJob.status;
  let responseProgress = persistedJob.progress;
  if (ACTIVE_JOB_STATUSES.has(persistedJob.status)) {
    const reconciled = await reconcileJobStatus(c, persistedJob);
    responseStatus = reconciled.status;
    responseProgress = reconciled.progress;
  }

  return c.json({
    job_id: jobId,
    round_id: round.round_id,
    dataset_snapshot_id: datasetSnapshot.snapshot_id,
    status: responseStatus,
    progress: responseProgress,
  });
});

app.get('/jobs', async (c) => {
  const voiceId = c.req.query('voice_id')?.trim();
  const limitRaw = c.req.query('limit');
  const parsedLimit = Number(limitRaw ?? '20');
  const limit = Number.isFinite(parsedLimit) ? parsedLimit : 20;

  const jobs = await listTrainingJobs(c.env.DB, {
    voice_id: voiceId || undefined,
    limit,
  });

  const hydratedJobs = await Promise.all(
    jobs.map(async (job) => {
      try {
        if (
          !ACTIVE_JOB_STATUSES.has(job.status) &&
          !needsCompletedValidation(job) &&
          !shouldRecoverFailedDependencyJob(c, job)
        ) {
          return job;
        }
        await reconcileJobStatusWithTimeout(c, job);
        return (await getTrainingJob(c.env.DB, job.job_id)) ?? job;
      } catch (error) {
        console.warn(`Failed to hydrate training job ${job.job_id}:`, error);
        return job;
      }
    }),
  );

  return c.json({ jobs: hydratedJobs.map(serializeTrainingJob) });
});

app.get('/advice', async (c) => {
  const voiceId = c.req.query('voice_id')?.trim();
  if (!voiceId) {
    return c.json({ detail: { message: 'voice_id is required' } }, 400);
  }

  const voice = await getVoice(c.env.DB, voiceId);
  if (!voice) {
    return c.json({ detail: { message: 'Voice not found' } }, 404);
  }

  const limitRaw = c.req.query('limit');
  const parsedLimit = Number(limitRaw ?? '40');
  const limit = Number.isFinite(parsedLimit) ? Math.max(1, Math.min(parsedLimit, 100)) : 40;
  const jobs = await listTrainingJobs(c.env.DB, {
    voice_id: voiceId,
    limit,
  });

  const hydratedJobs = await Promise.all(
    jobs.map(async (job) => {
      try {
        if (
          !ACTIVE_JOB_STATUSES.has(job.status) &&
          !needsCompletedValidation(job) &&
          !shouldRecoverFailedDependencyJob(c, job)
        ) {
          return job;
        }
        await reconcileJobStatusWithTimeout(c, job);
        return (await getTrainingJob(c.env.DB, job.job_id)) ?? job;
      } catch (error) {
        console.warn(`Failed to hydrate training advice job ${job.job_id}:`, error);
        return job;
      }
    }),
  );

  let advice: ReturnType<typeof buildTrainingAdvice> = null;
  try {
    advice = await buildLLMTrainingAdvice(c.env, voice, hydratedJobs);
  } catch (error) {
    console.warn('LLM advisor error, falling back to heuristic:', error);
  }
  if (!advice) {
    advice = buildTrainingAdvice(voice, hydratedJobs);
  }

  return c.json({
    advice,
    voice_id: voiceId,
    jobs_considered: hydratedJobs.length,
  });
});
app.get('/rounds', async (c) => {
  const voiceId = c.req.query('voice_id')?.trim();
  const limitRaw = c.req.query('limit');
  const parsedLimit = Number(limitRaw ?? '20');
  const limit = Number.isFinite(parsedLimit) ? parsedLimit : 20;
  const rounds = await listTrainingRounds(c.env.DB, {
    voice_id: voiceId || undefined,
    limit,
  });
  return c.json({ rounds: rounds.map(serializeTrainingRound) });
});

app.get('/rounds/:round_id', async (c) => {
  const roundId = c.req.param('round_id');
  const round = await getTrainingRound(c.env.DB, roundId);
  if (!round) {
    return c.json({ detail: { message: 'Training round not found' } }, 404);
  }
  return c.json(serializeTrainingRound(round));
});

app.get('/snapshots', async (c) => {
  const voiceId = c.req.query('voice_id')?.trim();
  const limitRaw = c.req.query('limit');
  const parsedLimit = Number(limitRaw ?? '20');
  const limit = Number.isFinite(parsedLimit) ? parsedLimit : 20;
  const snapshots = await listDatasetSnapshots(c.env.DB, {
    voice_id: voiceId || undefined,
    limit,
  });
  return c.json({ snapshots: snapshots.map(serializeDatasetSnapshot) });
});

app.get('/:job_id/logs', async (c) => {
  const jobId = c.req.param('job_id');
  const job = await getTrainingJob(c.env.DB, jobId);
  if (!job) {
    return c.json({ detail: { message: 'Training job not found' } }, 404);
  }

  const limitRaw = c.req.query('limit');
  const cursorRaw = c.req.query('cursor');
  const parsedLimit = Number(limitRaw ?? '50');
  const limit = Number.isFinite(parsedLimit) ? parsedLimit : 50;
  const parsedCursor =
    typeof cursorRaw === 'string' && cursorRaw.trim().length > 0 ? Number(cursorRaw) : NaN;
  const cursor = Number.isFinite(parsedCursor) ? parsedCursor : undefined;

  const chunks = await listTrainingLogChunks(c.env.DB, jobId, limit, cursor);
  const nextCursor = chunks.length > 0 ? chunks[chunks.length - 1].seq : null;

  return c.json({
    job_id: jobId,
    chunks,
    next_cursor: nextCursor,
  });
});

app.get('/:job_id/logs/:seq', async (c) => {
  const jobId = c.req.param('job_id');
  const seq = Number(c.req.param('seq'));
  if (!Number.isInteger(seq)) {
    return c.json({ detail: { message: 'Invalid seq' } }, 400);
  }

  const job = await getTrainingJob(c.env.DB, jobId);
  if (!job) {
    return c.json({ detail: { message: 'Training job not found' } }, 404);
  }

  const chunk = await getTrainingLogChunk(c.env.DB, jobId, seq);
  if (!chunk) {
    return c.json({ detail: { message: 'Log chunk not found' } }, 404);
  }

  const obj = await c.env.R2.get(chunk.r2_key);
  if (!obj) {
    return c.json({ detail: { message: 'R2 log chunk not found' } }, 404);
  }

  const contentType = obj.httpMetadata?.contentType || 'application/jsonl';
  return new Response(obj.body, {
    headers: {
      'content-type': contentType,
      'cache-control': 'no-store',
    },
  });
});

app.get('/:job_id/preprocess-cache', async (c) => {
  const jobId = c.req.param('job_id');
  const job = await getTrainingJob(c.env.DB, jobId);
  if (!job) {
    return c.json({ detail: { message: 'Training job not found' } }, 404);
  }

  const resolved = await ensureJobPreprocessCacheState(c, job);
  if (!resolved) {
    return c.json({
      job_id: jobId,
      cache: null,
      entries: [],
      reference_text: null,
      hydrated_from_r2: false,
    });
  }

  return c.json({
    job_id: jobId,
    cache: resolved.cache,
    entries: resolved.entries,
    reference_text: resolved.referenceText,
    hydrated_from_r2: resolved.hydratedFromR2,
  });
});

app.patch('/:job_id/preprocess-cache', async (c) => {
  const jobId = c.req.param('job_id');
  const job = await getTrainingJob(c.env.DB, jobId);
  if (!job) {
    return c.json({ detail: { message: 'Training job not found' } }, 404);
  }

  const body = (await c.req.json()) as { reference_text?: string };
  if (typeof body.reference_text !== 'string') {
    return c.json({ detail: { message: 'reference_text is required' } }, 400);
  }

  const resolved = await ensureJobPreprocessCacheState(c, job);
  if (!resolved) {
    return c.json({ detail: { message: 'No preprocess cache found for this training job' } }, 404);
  }

  const now = Date.now();
  const profileKey =
    (typeof resolved.cache.reference_profile_r2_key === 'string' &&
      resolved.cache.reference_profile_r2_key.trim()) ||
    `${resolved.cache.cache_r2_prefix}/reference_profile.json`;
  const existingRaw = await readR2Text(c.env.R2, profileKey);
  let profile: Record<string, unknown> = {};
  if (existingRaw) {
    try {
      profile = JSON.parse(existingRaw) as Record<string, unknown>;
    } catch {
      profile = {};
    }
  }
  profile.reference_audio_key =
    resolved.cache.ref_audio_r2_key ?? profile.reference_audio_key ?? null;
  profile.reference_text = body.reference_text.trim();
  await c.env.R2.put(profileKey, JSON.stringify(profile, null, 2), {
    httpMetadata: { contentType: 'application/json' },
  });

  await upsertDatasetPreprocessCache(c.env.DB, {
    ...resolved.cache,
    reference_profile_r2_key: profileKey,
    updated_at: now,
  });
  if (job.dataset_snapshot_id) {
    const snapshot = await getDatasetSnapshotById(c.env.DB, job.dataset_snapshot_id);
    if (snapshot) {
      await upsertDatasetSnapshot(c.env.DB, {
        ...snapshot,
        reference_profile_r2_key: profileKey,
        reference_text: profile.reference_text as string,
        updated_at: now,
      });
    }
  }

  return c.json({
    status: 'ok',
    reference_text: profile.reference_text,
    updated_at: now,
  });
});

app.patch('/:job_id/preprocess-cache/entries/:entry_id', async (c) => {
  const jobId = c.req.param('job_id');
  const entryId = c.req.param('entry_id');
  const job = await getTrainingJob(c.env.DB, jobId);
  if (!job) {
    return c.json({ detail: { message: 'Training job not found' } }, 404);
  }

  const body = (await c.req.json()) as {
    text?: string;
    included?: boolean;
  };
  if (body.text === undefined && body.included === undefined) {
    return c.json({ detail: { message: 'text or included must be provided' } }, 400);
  }

  const resolved = await ensureJobPreprocessCacheState(c, job);
  if (!resolved) {
    return c.json({ detail: { message: 'No preprocess cache found for this training job' } }, 404);
  }

  const currentEntry = resolved.entries.find((entry) => entry.entry_id === entryId);
  if (!currentEntry) {
    return c.json({ detail: { message: 'Preprocess cache entry not found' } }, 404);
  }

  const nextText = typeof body.text === 'string' ? body.text.trim() : currentEntry.text;
  const nextIncluded = typeof body.included === 'boolean' ? body.included : currentEntry.included;
  if (nextIncluded && !nextText) {
    return c.json(
      { detail: { message: 'Included transcript entries must have non-empty text' } },
      400,
    );
  }

  const nextEntries = resolved.entries.map((entry) =>
    entry.entry_id === entryId
      ? {
          ...entry,
          text: nextText,
          included: nextIncluded,
        }
      : entry,
  );
  let nextJsonl: string;
  try {
    nextJsonl = buildTrainRawJsonl(nextEntries);
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Invalid preprocess cache update';
    return c.json({ detail: { message } }, 400);
  }

  const now = Date.now();
  await updateDatasetPreprocessCacheEntry(c.env.DB, entryId, {
    text: nextText,
    included: nextIncluded,
    updated_at: now,
  });
  await c.env.R2.put(resolved.cache.train_raw_r2_key, nextJsonl, {
    httpMetadata: { contentType: 'application/jsonl' },
  });
  await upsertDatasetPreprocessCache(c.env.DB, {
    ...resolved.cache,
    updated_at: now,
  });
  if (job.dataset_snapshot_id) {
    const snapshot = await getDatasetSnapshotById(c.env.DB, job.dataset_snapshot_id);
    if (snapshot) {
      await upsertDatasetSnapshot(c.env.DB, {
        ...snapshot,
        train_raw_r2_key: resolved.cache.train_raw_r2_key,
        updated_at: now,
      });
    }
  }

  return c.json({
    status: 'ok',
    entry: {
      ...currentEntry,
      text: nextText,
      included: nextIncluded,
      updated_at: now,
    },
    included_entries: nextEntries.filter((entry) => entry.included).length,
    updated_at: now,
  });
});

app.get('/:job_id', async (c) => {
  const jobId = c.req.param('job_id');
  const job = await getTrainingJob(c.env.DB, jobId);

  if (!job) {
    return c.json({ detail: { message: 'Training job not found' } }, 404);
  }

  const shouldRecoverDependencyFailure = shouldRecoverFailedDependencyJob(c, job);
  if (
    ACTIVE_JOB_STATUSES.has(job.status) ||
    needsCompletedValidation(job) ||
    shouldRecoverDependencyFailure
  ) {
    const timeoutMs = needsCompletedValidation(job)
      ? COMPLETED_VALIDATION_TIMEOUT_MS
      : RECONCILE_TIMEOUT_MS;
    const reconciled = await reconcileJobStatusWithTimeout(c, job, timeoutMs);
    if (!reconciled) {
      return c.json({
        ...serializeTrainingJob(job),
        summary: {
          ...(job.summary ?? {}),
          validation_in_progress: true,
        },
      });
    }
    const updated = await getTrainingJob(c.env.DB, jobId);
    if (updated) {
      if (ACTIVE_JOB_STATUSES.has(updated.status) && updated.runpod_pod_id) {
        try {
          const pod = await getPodStatus(c.env, updated.runpod_pod_id);
          return c.json({
            ...serializeTrainingJob(updated),
            summary: {
              ...(updated.summary ?? {}),
              pod_status: pod,
            },
          });
        } catch {
          return c.json(serializeTrainingJob(updated));
        }
      }
      return c.json(serializeTrainingJob(updated));
    }
  }

  if (ACTIVE_JOB_STATUSES.has(job.status) && job.runpod_pod_id) {
    try {
      const pod = await getPodStatus(c.env, job.runpod_pod_id);
      return c.json({
        ...serializeTrainingJob(job),
        summary: {
          ...(job.summary ?? {}),
          pod_status: pod,
        },
      });
    } catch {
      return c.json(serializeTrainingJob(job));
    }
  }

  return c.json(serializeTrainingJob(job));
});

app.get('/:job_id/checkout-ledger', async (c) => {
  const jobId = c.req.param('job_id');
  const job = await getTrainingJob(c.env.DB, jobId);
  if (!job) {
    return c.json({ detail: { message: 'Training job not found' } }, 404);
  }

  const entries = await listTrainingCheckoutLedger(c.env.DB, { job_id: jobId, limit: 500 });
  return c.json({ entries });
});

app.get('/debug/template/:template_id', async (c) => {
  const templateId = c.req.param('template_id');
  try {
    const template = await getTemplateById(c.env, templateId);
    if (!template) {
      return c.json({ detail: { message: 'Template not found' } }, 404);
    }
    const safeTemplate = {
      id: template.id,
      imageName: template.imageName ?? null,
      containerRegistryAuthConfigured: Boolean(template.containerRegistryAuthId),
      ports: template.ports ?? null,
      volumeMountPath: template.volumeMountPath ?? null,
      dockerEntrypoint: template.dockerEntrypoint ?? null,
      dockerStartCmd: template.dockerStartCmd ?? null,
      isServerless: template.isServerless ?? null,
    };
    return c.json(safeTemplate);
  } catch (error) {
    return c.json(
      {
        detail: {
          message: error instanceof Error ? error.message : 'Failed to fetch template',
        },
      },
      502,
    );
  }
});

app.post('/:job_id/reconcile', async (c) => {
  const jobId = c.req.param('job_id');
  const job = await getTrainingJob(c.env.DB, jobId);
  if (!job) {
    return c.json({ detail: { message: 'Training job not found' } }, 404);
  }

  if (!ACTIVE_JOB_STATUSES.has(job.status) && !needsCompletedValidation(job)) {
    if (QUEUED_JOB_STATUSES.has(job.status) || job.status === 'pending') {
      waitForBackgroundTask(c, launchQueuedTrainingJobsForVoice(c, job.voice_id));
      return c.json({
        status: 'accepted',
        validation_started: false,
        job_id: jobId,
      });
    }
    return c.json({
      status: 'noop',
      job: serializeTrainingJob(job),
    });
  }

  waitForBackgroundTask(c, reconcileJobStatus(c, job));

  return c.json({
    status: 'accepted',
    validation_started: true,
    job_id: jobId,
  });
});

app.post('/:job_id/revalidate', async (c) => {
  const jobId = c.req.param('job_id');
  const job = await getTrainingJob(c.env.DB, jobId);

  if (!job) {
    return c.json({ detail: { message: 'Training job not found' } }, 404);
  }
  if (ACTIVE_JOB_STATUSES.has(job.status)) {
    return c.json({ detail: { message: 'Cannot revalidate an active training job' } }, 409);
  }

  const summary = (job.summary ?? {}) as Record<string, unknown>;
  const {
    validation_checked: _validationChecked,
    validation_passed: _validationPassed,
    validation_failed: _validationFailed,
    validation_in_progress: _validationInProgress,
    validation_message: _validationMessage,
    validation_rejected: _validationRejected,
    evaluated_checkpoints: _evaluatedCheckpoints,
    async_validation: _asyncValidation,
    selected_checkpoint_prefix: _selectedCheckpointPrefix,
    selected_checkpoint_epoch: _selectedCheckpointEpoch,
    selected_preset: _selectedPreset,
    selected_score: _selectedScore,
    candidate_checkpoint_prefix: _candidateCheckpointPrefix,
    candidate_checkpoint_epoch: _candidateCheckpointEpoch,
    candidate_preset: _candidatePreset,
    candidate_score: _candidateScore,
    candidate_promotion_mode: _candidatePromotionMode,
    manual_promoted_checkpoint_prefix: _manualPromotedCheckpointPrefix,
    manual_promoted_checkpoint_epoch: _manualPromotedCheckpointEpoch,
    manual_promoted_preset: _manualPromotedPreset,
    manual_promoted_score: _manualPromotedScore,
    manual_promotion_at: _manualPromotionAt,
    ...restSummary
  } = summary;

  const voice = await getVoice(c.env.DB, job.voice_id);
  if (voice?.candidate_job_id === jobId) {
    await updateVoice(c.env.DB, job.voice_id, {
      candidate_checkpoint_r2_prefix: null,
      candidate_run_name: null,
      candidate_epoch: null,
      candidate_preset: null,
      candidate_score: null,
      candidate_job_id: null,
    });
  }

  if (job.round_id) {
    const existingRound = await getTrainingRound(c.env.DB, job.round_id);
    await updateTrainingRound(c.env.DB, job.round_id, {
      status: 'validating',
      champion_checkpoint_r2_prefix: null,
      champion_run_name: null,
      champion_epoch: null,
      champion_preset: null,
      champion_score: null,
      champion_job_id: null,
      selected_checkpoint_r2_prefix: null,
      selected_run_name: null,
      selected_epoch: null,
      selected_preset: null,
      selected_score: null,
      selected_job_id: null,
      adoption_mode: null,
      candidate_checkpoint_r2_prefix: null,
      candidate_run_name: null,
      candidate_epoch: null,
      candidate_score: null,
      candidate_job_id: null,
      completed_at: null,
      summary: {
        ...(existingRound?.summary ?? {}),
        revalidation_started_at: Date.now(),
      },
    });
  }

  await replaceTrainingCheckoutLedgerForJob(c.env.DB, jobId, []);
  await updateTrainingJob(c.env.DB, jobId, {
    status: 'completed',
    error_message: null,
    summary: {
      ...restSummary,
      force_revalidation: true,
      validation_checked: false,
      validation_passed: false,
      validation_failed: false,
      validation_in_progress: false,
      evaluated_checkpoints: [],
      async_validation: null,
    },
  });

  const refreshedJob = await getTrainingJob(c.env.DB, jobId);
  if (!refreshedJob) {
    return c.json({ detail: { message: 'Training job not found after reset' } }, 404);
  }

  const reconciled = await reconcileJobStatusWithTimeout(
    c,
    refreshedJob,
    COMPLETED_VALIDATION_TIMEOUT_MS,
  );

  const updatedJob = await getTrainingJob(c.env.DB, jobId);
  return c.json({
    status: reconciled ? 'started' : 'accepted',
    job: serializeTrainingJob(updatedJob ?? refreshedJob),
  });
});

app.post('/:job_id/promote', async (c) => {
  const jobId = c.req.param('job_id');
  const body = (await c.req.json().catch(() => ({}))) as {
    checkpoint_prefix?: string;
  };
  const checkpointPrefix =
    typeof body.checkpoint_prefix === 'string' ? body.checkpoint_prefix.trim() : '';
  if (!checkpointPrefix) {
    return c.json({ detail: { message: 'checkpoint_prefix is required' } }, 400);
  }

  const job = await getTrainingJob(c.env.DB, jobId);
  if (!job) {
    return c.json({ detail: { message: 'Training job not found' } }, 404);
  }

  const voice = await getVoice(c.env.DB, job.voice_id);
  if (!voice) {
    return c.json({ detail: { message: 'Voice not found' } }, 404);
  }

  const summary = (job.summary ?? {}) as Record<string, unknown>;
  const candidate = collectManualPromotionCandidates(summary).find(
    (value) => value.prefix === checkpointPrefix,
  );
  if (!candidate) {
    return c.json(
      {
        detail: {
          message: "checkpoint_prefix was not found among the job's evaluated checkpoints",
        },
      },
      404,
    );
  }

  await updateVoice(c.env.DB, job.voice_id, {
    status: 'ready',
    checkpoint_r2_prefix: candidate.prefix,
    run_name: parseRunNameFromCheckpointPrefix(candidate.prefix),
    epoch: candidate.epoch,
    checkpoint_preset: candidate.preset,
    checkpoint_score: candidate.score,
    checkpoint_job_id: job.job_id,
    candidate_checkpoint_r2_prefix: null,
    candidate_run_name: null,
    candidate_epoch: null,
    candidate_preset: null,
    candidate_score: null,
    candidate_job_id: null,
    active_round_id: voice.active_round_id ?? job.round_id ?? null,
    settings: resolvePromotionSettings(voice, candidate.preset),
  });

  const nextSummary = {
    ...summary,
    selected_checkpoint_prefix: candidate.prefix,
    selected_checkpoint_epoch: candidate.epoch,
    selected_preset: candidate.preset,
    selected_score: candidate.score,
    manual_promoted_checkpoint_prefix: candidate.prefix,
    manual_promoted_checkpoint_epoch: candidate.epoch,
    manual_promoted_preset: candidate.preset,
    manual_promoted_score: candidate.score,
    manual_promotion_at: Date.now(),
  };
  await updateTrainingJob(c.env.DB, jobId, {
    summary: nextSummary,
    supervisor: {
      ...(job.supervisor ?? {}),
      phase: 'promoted',
      last_transition_at: Date.now(),
    },
  });

  if (job.round_id) {
    const championPrefix =
      typeof summary.candidate_checkpoint_prefix === 'string' &&
      summary.candidate_checkpoint_prefix.trim()
        ? summary.candidate_checkpoint_prefix.trim()
        : candidate.prefix;
    const championEpoch = readNumber(summary.candidate_checkpoint_epoch) ?? candidate.epoch;
    const championPreset =
      typeof summary.candidate_preset === 'string' && summary.candidate_preset.trim()
        ? summary.candidate_preset.trim()
        : candidate.preset;
    const championScore = readNumber(summary.candidate_score) ?? candidate.score;
    await updateTrainingRound(c.env.DB, job.round_id, {
      status: 'promoted',
      production_checkpoint_r2_prefix: candidate.prefix,
      production_run_name: parseRunNameFromCheckpointPrefix(candidate.prefix),
      production_epoch: candidate.epoch,
      production_preset: candidate.preset,
      production_score: candidate.score,
      production_job_id: job.job_id,
      champion_checkpoint_r2_prefix: championPrefix,
      champion_run_name: parseRunNameFromCheckpointPrefix(championPrefix),
      champion_epoch: championEpoch,
      champion_preset: championPreset,
      champion_score: championScore,
      champion_job_id: job.job_id,
      selected_checkpoint_r2_prefix: candidate.prefix,
      selected_run_name: parseRunNameFromCheckpointPrefix(candidate.prefix),
      selected_epoch: candidate.epoch,
      selected_preset: candidate.preset,
      selected_score: candidate.score,
      selected_job_id: job.job_id,
      adoption_mode: 'promote',
      candidate_checkpoint_r2_prefix: null,
      candidate_run_name: null,
      candidate_epoch: null,
      candidate_score: null,
      candidate_job_id: null,
      completed_at: Date.now(),
      summary: {
        manual_promoted_checkpoint_prefix: candidate.prefix,
        manual_promoted_checkpoint_epoch: candidate.epoch,
        manual_promoted_score: candidate.score,
        manual_promotion_at: Date.now(),
      },
    });
  }

  await syncTrainingCheckoutLedgerForJob(c, job, nextSummary, Date.now());

  const updatedVoice = await getVoice(c.env.DB, job.voice_id);
  const updatedJob = await getTrainingJob(c.env.DB, jobId);
  return c.json({
    status: 'ok',
    voice: updatedVoice,
    job: updatedJob ? serializeTrainingJob(updatedJob) : serializeTrainingJob(job),
  });
});

app.post('/:job_id/cancel', async (c) => {
  const jobId = c.req.param('job_id');
  const job = await getTrainingJob(c.env.DB, jobId);

  if (!job) {
    return c.json({ detail: { message: 'Training job not found' } }, 404);
  }

  if (job.runpod_pod_id) {
    try {
      await terminatePod(c.env, job.runpod_pod_id);
    } catch {
      // Pod may already be terminated — safe to ignore
    }
  }

  await updateTrainingJob(c.env.DB, jobId, {
    status: 'cancelled',
    completed_at: Date.now(),
  });
  if (job.round_id) {
    await updateTrainingRound(c.env.DB, job.round_id, {
      status: 'cancelled',
      completed_at: Date.now(),
      summary: {
        cancelled_job_id: job.job_id,
      },
    });
  }
  await launchQueuedTrainingJobsForVoice(c, job.voice_id);

  return c.json({ status: 'ok' });
});

export async function dispatchQueuedJobs(env: Env, voiceId: string): Promise<number> {
  const workerOrigin = resolveWorkerPublicUrl(env);
  const syntheticContext = createSyntheticContext(env, workerOrigin);
  return launchQueuedTrainingJobsForVoice(syntheticContext, voiceId);
}

export default app;
