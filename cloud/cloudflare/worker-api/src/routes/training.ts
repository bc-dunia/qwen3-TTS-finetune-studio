import { Hono } from "hono";
import type { Context } from "hono";
import {
  createTrainingJob,
  deleteTrainingLogChunks,
  getTrainingJob,
  getTrainingLogChunk,
  getVoice,
  listTrainingJobs,
  listTrainingLogChunks,
  updateTrainingJob,
  updateVoice,
} from "../lib/d1";
import {
  createPod,
  createPodDirect,
  getServerlessStatus,
  getPodStatus,
  getTemplateById,
  invokeServerless,
  invokeServerlessAsync,
  terminatePod,
} from "../lib/runpod";
import { enrichOutputWithReviewAsr } from "../lib/review-asr";
import { authMiddleware } from "../middleware/auth";
import type { AppContext, Env, TrainingConfig, TrainingJob, TrainingProgress } from "../types";

const app = new Hono<AppContext>();
app.use("*", authMiddleware);

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
  presetSettings?: ValidationPreset["settings"];
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
  speaker_samples: number;
  tone_samples: number;
  speed_samples: number;
  first_failure_message: string | null;
};

type AsyncValidationChampion = {
  epoch: number;
  prefix: string;
  score: number;
  message: string;
  preset_name: string;
  preset_settings?: ValidationPreset["settings"];
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
  mode: "checkpoint_async";
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
  mode: "fast_06b_async";
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
  failureMessage: string | null;
};

const ACTIVE_JOB_STATUSES = new Set([
  "pending",
  "running",
  "provisioning",
  "downloading",
  "preprocessing",
  "preparing",
  "training",
  "uploading",
]);
const ACTIVE_RUNTIME_RECOVERY_STATUSES = new Set([
  "pending",
  "running",
  "downloading",
  "preprocessing",
  "preparing",
  "training",
  "uploading",
]);
const ACTIVE_STAGE_STALE_MS: Record<string, number> = {
  pending: 5 * 60 * 1000,
  running: 12 * 60 * 1000,
  downloading: 10 * 60 * 1000,
  preprocessing: 30 * 60 * 1000,
  preparing: 20 * 60 * 1000,
  training: 12 * 60 * 1000,
  uploading: 20 * 60 * 1000,
};
const MAX_STALL_RECOVERY_ATTEMPTS = 3;
const DEFAULT_WORKER_PUBLIC_URL = "https://qwen-tts-api.brian-367.workers.dev";
const TRAINING_SWEEP_LIMIT = 100;

const REFERENCE_AUDIO_KEY_RE = /\/ref_audio\.[^/]+$/i;

const needsCompletedValidation = (job: TrainingJob): boolean => {
  if (job.status !== "completed") {
    return false;
  }
  const summary = (job.summary ?? {}) as Record<string, unknown>;
  return summary.validation_checked !== true;
};

const extractDatasetPrefixFromRefAudioKey = (
  voice: NonNullable<Awaited<ReturnType<typeof getVoice>>>
): string | null => {
  const refAudioKey = typeof voice.ref_audio_r2_key === "string" ? voice.ref_audio_r2_key.trim() : "";
  if (!REFERENCE_AUDIO_KEY_RE.test(refAudioKey)) {
    return null;
  }

  const datasetPrefix = refAudioKey.replace(REFERENCE_AUDIO_KEY_RE, "").replace(/\/+$/, "");
  const expectedPrefix = `datasets/${voice.voice_id}/`;
  if (!datasetPrefix.startsWith(expectedPrefix)) {
    return null;
  }

  return datasetPrefix;
};

const resolveTrainingDatasetPrefix = (
  voice: NonNullable<Awaited<ReturnType<typeof getVoice>>>,
  datasetName: string | undefined
): string => {
  const requestedDatasetName = typeof datasetName === "string" ? datasetName.trim() : "";
  if (requestedDatasetName) {
    return `datasets/${voice.voice_id}/${requestedDatasetName}`;
  }

  return extractDatasetPrefixFromRefAudioKey(voice) ?? `datasets/${voice.voice_id}`;
};

const getCurrentReadyVoiceScore = async (
  c: Context<AppContext>,
  voice: NonNullable<Awaited<ReturnType<typeof getVoice>>>
): Promise<number | null> => {
  const currentPrefix = typeof voice.checkpoint_r2_prefix === "string" ? voice.checkpoint_r2_prefix.trim() : "";
  if (voice.status !== "ready" || !currentPrefix) {
    return null;
  }

  const jobs = await listTrainingJobs(c.env.DB, { voice_id: voice.voice_id, limit: 100 });
  for (const candidate of jobs) {
    const summary = candidate.summary ?? {};
    const selectedPrefix =
      typeof summary.selected_checkpoint_prefix === "string" ? summary.selected_checkpoint_prefix.trim() : "";
    if (selectedPrefix === currentPrefix) {
      const selectedScore = Number(summary.selected_score);
      if (Number.isFinite(selectedScore)) {
        return selectedScore;
      }
    }

    const manualPrefix =
      typeof summary.manual_promoted_checkpoint_prefix === "string"
        ? summary.manual_promoted_checkpoint_prefix.trim()
        : "";
    if (manualPrefix === currentPrefix) {
      const manualScore = Number(summary.manual_promoted_score);
      if (Number.isFinite(manualScore)) {
        return manualScore;
      }
    }
  }

  return null;
};

const chooseCheckpointPromotion = async ({
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
  promote: boolean;
  preservedPrefix: string | null;
  preservedEpoch: number | null;
  preservedScore: number | null;
}> => {
  const currentPrefix = typeof voice.checkpoint_r2_prefix === "string" ? voice.checkpoint_r2_prefix.trim() : "";
  if (voice.status !== "ready" || !currentPrefix || currentPrefix === candidatePrefix) {
    return {
      promote: true,
      preservedPrefix: currentPrefix || null,
      preservedEpoch: voice.epoch,
      preservedScore: await getCurrentReadyVoiceScore(c, voice),
    };
  }

  const currentScore = await getCurrentReadyVoiceScore(c, voice);
  if (currentScore !== null && candidateScore <= currentScore) {
    return {
      promote: false,
      preservedPrefix: currentPrefix,
      preservedEpoch: voice.epoch,
      preservedScore: currentScore,
    };
  }

  return {
    promote: true,
    preservedPrefix: currentPrefix,
    preservedEpoch: voice.epoch,
    preservedScore: currentScore,
  };
};

const parseRunNameFromCheckpointPrefix = (prefix: string): string | null => {
  const parts = prefix.split("/");
  if (parts.length < 4 || parts[0] !== "checkpoints") {
    return null;
  }
  return parts[2] || null;
};

type ManualPromotionCandidate = {
  prefix: string;
  epoch: number | null;
  preset: string | null;
  score: number | null;
};

const collectManualPromotionCandidates = (
  summary: Record<string, unknown>
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

  const evaluated = Array.isArray(summary.evaluated_checkpoints) ? summary.evaluated_checkpoints : [];
  for (const value of evaluated) {
    if (!value || typeof value !== "object") {
      continue;
    }
    const record = value as Record<string, unknown>;
    const prefix = typeof record.prefix === "string" ? record.prefix.trim() : "";
    register({
      prefix,
      epoch: readNumber(record.epoch),
      preset: typeof record.preset === "string" ? record.preset.trim() : null,
      score: readNumber(record.score),
    });
  }

  const selectedPrefix =
    typeof summary.selected_checkpoint_prefix === "string" ? summary.selected_checkpoint_prefix.trim() : "";
  if (selectedPrefix) {
    register({
      prefix: selectedPrefix,
      epoch: readNumber(summary.selected_checkpoint_epoch),
      preset: typeof summary.selected_preset === "string" ? summary.selected_preset.trim() : null,
      score: readNumber(summary.selected_score),
    });
  }

  const candidatePrefix =
    typeof summary.candidate_checkpoint_prefix === "string" ? summary.candidate_checkpoint_prefix.trim() : "";
  if (candidatePrefix) {
    register({
      prefix: candidatePrefix,
      epoch: readNumber(summary.candidate_checkpoint_epoch),
      preset: typeof summary.candidate_preset === "string" ? summary.candidate_preset.trim() : null,
      score: readNumber(summary.candidate_score),
    });
  }

  return [...byPrefix.values()];
};

const resolvePromotionSettings = (
  voice: NonNullable<Awaited<ReturnType<typeof getVoice>>>,
  presetName: string | null
) => {
  const normalizedPreset = typeof presetName === "string" ? presetName.trim() : "";
  const presets = getValidationPresets(
    voice.model_id ?? (voice.model_size || "1.7B"),
    String(voice.labels?.language ?? "")
  );
  const preset = presets.find((value) => value.name === normalizedPreset);
  return preset?.settings ?? voice.settings ?? {};
};

const isMissingRunpodRequestError = (error: unknown): error is Error => {
  if (!(error instanceof Error)) {
    return false;
  }
  return (
    error.message.includes("RunPod status request failed (404)") &&
    error.message.includes("request does not exist")
  );
};

const getServerlessStatusOrSyntheticFailure = async (
  env: AppContext["Bindings"],
  endpointId: string,
  runId: string
): Promise<Record<string, unknown>> => {
  try {
    return await getServerlessStatus(env, endpointId, runId);
  } catch (error) {
    if (!isMissingRunpodRequestError(error)) {
      throw error;
    }
    return {
      status: "FAILED",
      error: error.message,
    };
  }
};

const shouldPreserveCurrentReadyVoice = (
  voice: NonNullable<Awaited<ReturnType<typeof getVoice>>>
): boolean =>
  voice.status === "ready" &&
  typeof voice.checkpoint_r2_prefix === "string" &&
  voice.checkpoint_r2_prefix.trim().length > 0;

const shouldKeepReadyVoiceOnValidationFailure = (
  voice: NonNullable<Awaited<ReturnType<typeof getVoice>>>,
  summary: Record<string, unknown>,
  options?: {
    evaluatedCheckpoints?: CheckpointEvaluation[];
    validationRunName?: string | null;
    forceRevalidation?: boolean;
  }
): boolean => {
  if (!shouldPreserveCurrentReadyVoice(voice)) {
    return false;
  }

  const currentPrefix =
    typeof voice.checkpoint_r2_prefix === "string" ? voice.checkpoint_r2_prefix.trim() : "";
  if (!currentPrefix) {
    return false;
  }

  const getRunNameFromPrefix = (prefix: string): string | null => {
    const parts = prefix.split("/");
    if (parts.length < 4 || parts[0] !== "checkpoints") {
      return null;
    }
    return parts[2] || null;
  };
  const currentRunName = getRunNameFromPrefix(currentPrefix);
  const referencedPrefixes = new Set<string>();
  const referencedRunNames = new Set<string>();
  const registerPrefix = (value: unknown) => {
    if (typeof value !== "string") {
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
    if (typeof value !== "string") {
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
    summary.async_validation && typeof summary.async_validation === "object"
      ? (summary.async_validation as Record<string, unknown>)
      : null;
  registerPrefix(asyncValidation?.checkpoint_prefix);

  const evaluated = Array.isArray(summary.evaluated_checkpoints) ? summary.evaluated_checkpoints : [];
  for (const value of evaluated) {
    if (!value || typeof value !== "object") {
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
const FAST_VALIDATION_SEEDS_OFFSET = [123456, 223456] as const;
const MAX_CHECKPOINTS_TO_EVAL = 4;
const MAX_CHECKPOINTS_TO_EVAL_06B = 4;
const VALIDATION_RETRY_ATTEMPTS = 3;
const MIN_PASS_RATE_06B = 5 / 6;
const MIN_PASS_RATE_17B = 5 / 6;
const PROVISIONING_STALE_MS = 4 * 60 * 1000;
const VALIDATION_RUN_STALE_MS = 6 * 60 * 1000;

const getRecommendedTrainingDefaults = (modelSize: string): {
  batch_size: number;
  learning_rate: number;
  num_epochs: number;
  gradient_accumulation_steps: number;
  subtalker_loss_weight: number;
  save_every_n_epochs: number;
  seed: number;
  gpu_type_id: string;
} => {
  if (modelSize.includes("0.6")) {
    return {
      batch_size: 2,
      // Keep 0.6B close to the official finetuning recipe; prior higher-LR/lower-subtalker runs were unstable.
      learning_rate: 2.5e-6,
      num_epochs: 12,
      gradient_accumulation_steps: 4,
      subtalker_loss_weight: 0.3,
      save_every_n_epochs: 1,
      seed: 303,
      gpu_type_id: "NVIDIA L40S",
    };
  }

  return {
    batch_size: 2,
    learning_rate: 2e-5,
    num_epochs: 15,
    gradient_accumulation_steps: 4,
    subtalker_loss_weight: 0.3,
    save_every_n_epochs: 5,
    seed: 42,
    gpu_type_id: "NVIDIA A100-SXM4-80GB",
  };
};
const MAX_PROVISIONING_RECOVERY_ATTEMPTS = 2;

const OVERALL_SCORE_ERROR_RE = /overall_score=([0-9.]+)/i;
const VALIDATION_ASR_ERROR_KEY = "openai_asr_error";

const resolveWorkerPublicUrl = (env: Pick<Env, "WORKER_PUBLIC_URL">, requestUrl?: string | null): string => {
  if (typeof requestUrl === "string" && requestUrl.trim()) {
    return new URL(requestUrl).origin;
  }
  const configured = env.WORKER_PUBLIC_URL?.trim();
  return configured ? configured.replace(/\/+$/, "") : DEFAULT_WORKER_PUBLIC_URL;
};

const getWorkerOrigin = (c: Context<AppContext>): string => resolveWorkerPublicUrl(c.env, c.req.url);
const createSyntheticContext = (env: Env, workerOrigin: string): Context<AppContext> =>
  ({
    env,
    req: {
      url: `${workerOrigin.replace(/\/+$/, "")}/`,
    },
  } as unknown as Context<AppContext>);
const GHCR_INDEX_ACCEPT =
  "application/vnd.oci.image.index.v1+json, application/vnd.docker.distribution.manifest.list.v2+json, application/vnd.docker.distribution.manifest.v2+json, application/vnd.oci.image.manifest.v1+json";

const readNumber = (value: unknown): number | null => {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
};

const readTimestamp = (value: unknown): number | null => {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === "string" && value.trim()) {
    const parsed = Date.parse(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
};

const getValidationRunStartedAt = (
  persistedState: Record<string, unknown> | null,
  job: TrainingJob
): number => {
  const fromState = readNumber(persistedState?.run_started_at);
  if (fromState !== null) {
    return fromState;
  }
  const fallback = readNumber(job.completed_at) ?? readNumber(job.updated_at) ?? Date.now();
  return fallback;
};

const resolveGhcrAmd64Image = async (imageName: string): Promise<string> => {
  if (!imageName.startsWith("ghcr.io/") || imageName.includes("@")) {
    return imageName;
  }

  const imageRef = imageName.slice("ghcr.io/".length);
  const tagSeparator = imageRef.lastIndexOf(":");
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

  const contentType = String(manifestResp.headers.get("content-type") ?? "").toLowerCase();
  const registryDigest = manifestResp.headers.get("docker-content-digest");

  if (contentType.includes("image.index") || contentType.includes("manifest.list")) {
    const payload = (await manifestResp.json()) as {
      manifests?: Array<{
        digest?: string;
        platform?: { architecture?: string; os?: string };
      }>;
    };
    const amd64 = payload.manifests?.find(
      (manifest) =>
        manifest.platform?.os === "linux" &&
        manifest.platform?.architecture === "amd64" &&
        typeof manifest.digest === "string"
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
  return typeof config.model_size === "string" && config.model_size
    ? config.model_size
    : "1.7B";
};

const getTrainingGpuType = (job: TrainingJob): string => {
  const config = job.config as Record<string, unknown>;
  if (typeof config.gpu_type_id === "string" && config.gpu_type_id) {
    return config.gpu_type_id;
  }
  return getConfiguredModelSize(job).includes("0.6")
    ? "NVIDIA GeForce RTX 4090"
    : "NVIDIA L40S";
};

const buildTrainingPodEnv = (
  c: Context<AppContext>,
  job: TrainingJob
): Array<{ key: string; value: string }> => {
  const workerUrl = getWorkerOrigin(c);
  return [
    { key: "JOB_ID", value: job.job_id },
    { key: "VOICE_ID", value: job.voice_id },
    { key: "WORKER_API_URL", value: workerUrl },
    { key: "JOB_TOKEN", value: job.job_token ?? "" },
    { key: "R2_ENDPOINT_URL", value: c.env.R2_ENDPOINT_URL },
    { key: "R2_ACCESS_KEY_ID", value: c.env.R2_ACCESS_KEY_ID },
    { key: "R2_SECRET_ACCESS_KEY", value: c.env.R2_SECRET_ACCESS_KEY },
    { key: "R2_BUCKET", value: "qwen-tts-studio" },
    { key: "RUNPOD_API_KEY", value: c.env.RUNPOD_API_KEY },
    { key: "HF_HUB_ENABLE_HF_TRANSFER", value: "0" },
  ];
};

const getConfiguredTrainingImageName = (c: Context<AppContext>): string | null => {
  const imageName = c.env.RUNPOD_TRAINING_IMAGE_NAME?.trim();
  return imageName ? imageName : null;
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

const createTrainingPodForJob = async (
  c: Context<AppContext>,
  job: TrainingJob
): Promise<{
  pod: { podId: string; desiredStatus: string };
  summary: Record<string, unknown>;
}> => {
  const imageName = getConfiguredTrainingImageName(c);
  if (imageName) {
    const dockerArgs = getConfiguredTrainingDockerArgs(c);
    const volumeMountPath = getConfiguredTrainingVolumeMountPath(c);
    const pod = await createPodDirect(c.env, {
      gpuTypeId: getTrainingGpuType(job),
      envVars: buildTrainingPodEnv(c, job),
      imageName,
      dockerArgs,
      name: `qwen3-tts-training-${job.job_id.slice(0, 8)}`,
      cloudType: "ALL",
      volumeMountPath: volumeMountPath ?? undefined,
    });
    return {
      pod,
      summary: {
        training_launch_mode: "direct_image",
        training_image_name: imageName,
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
      buildTrainingPodEnv(c, job)
    );
    return {
      pod,
      summary: {
        training_launch_mode: "template",
        training_template_id: templateId,
      },
    };
  }

  throw new Error("No training template or direct image configured");
};

const getDirectFallbackDockerArgs = (
  pod: PodStatusDetail,
  template: Awaited<ReturnType<typeof getTemplateById>>
): string | null => {
  if (pod.dockerArgs && pod.dockerArgs.trim()) {
    return pod.dockerArgs.trim();
  }
  const entrypoint = Array.isArray(template?.dockerEntrypoint)
    ? template.dockerEntrypoint.join(" ").trim()
    : "";
  const startCmd = Array.isArray(template?.dockerStartCmd)
    ? template.dockerStartCmd.join(" ").trim()
    : "";
  const joined = [entrypoint, startCmd].filter(Boolean).join(" ").trim();
  return joined || null;
};

const getProvisioningState = (pod: PodStatusDetail | null): string => {
  return String(pod?.latestTelemetry?.state ?? pod?.runtimeStatus ?? "").trim().toLowerCase();
};

const isProvisioningPodStalled = (pod: PodStatusDetail | null): boolean => {
  if (!pod) {
    return false;
  }

  const state = getProvisioningState(pod);
  const uptimeSeconds = readNumber(pod.uptimeSeconds ?? pod.runtime?.uptimeInSeconds);
  const cpuUtil = readNumber(pod.latestTelemetry?.cpuUtilization ?? pod.runtime?.container?.cpuPercent);
  const memoryUtil = readNumber(
    pod.latestTelemetry?.memoryUtilization ?? pod.runtime?.container?.memoryPercent
  );
  const gpuUtil = (pod.runtime?.gpus ?? [])
    .map((gpu) => readNumber(gpu.gpuUtilPercent))
    .filter((value): value is number => value !== null)
    .reduce((max, value) => Math.max(max, value), 0);
  const noUtilization = [cpuUtil, memoryUtil, gpuUtil].every(
    (value) => value === null || value === 0
  );

  if (state === "created" || state === "pending" || state === "starting") {
    return noUtilization;
  }

  return (state === "" || state === "running") && uptimeSeconds === 0 && noUtilization;
};

const recoverStalledProvisioningJob = async (
  c: Context<AppContext>,
  job: TrainingJob
): Promise<TrainingJob> => {
  if (job.status !== "provisioning" || !job.runpod_pod_id) {
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
  const podState = podStatus ? (getProvisioningState(podStatus) || "unknown") : "missing";
  const previousPodIds = Array.isArray(summary.previous_runpod_pod_ids)
    ? summary.previous_runpod_pod_ids.filter((value): value is string => typeof value === "string")
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
    `state=${podState} uptime=${podStatus?.uptimeSeconds ?? podStatus?.runtime?.uptimeInSeconds ?? "n/a"}s ` +
    `image=${podStatus?.imageName ?? "unknown"} template=${podStatus?.templateId ?? "unknown"}`;

  if (!job.job_token) {
    await updateTrainingJob(c.env.DB, job.job_id, {
      status: "failed",
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
        !hasTriedDirectFallback ||
        (!hasTriedDigestFallback && resolvedImage !== fallbackImage);
      if (shouldTryDirect) {
        await terminatePod(c.env, job.runpod_pod_id).catch(() => false);

        const directSummary = {
          ...nextSummary,
          provisioning_direct_fallback_attempted: true,
          provisioning_direct_fallback_image: resolvedImage,
          provisioning_direct_fallback_docker_args: dockerArgs,
          provisioning_digest_fallback_attempted: resolvedImage !== fallbackImage,
        };

        const tryCreateDirect = async (cloudType: "COMMUNITY" | "ALL") =>
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
          let newPod;
          try {
            newPod = await tryCreateDirect("COMMUNITY");
          } catch {
            newPod = await tryCreateDirect("ALL");
          }

          await updateTrainingJob(c.env.DB, job.job_id, {
            runpod_pod_id: newPod.podId,
            status: "provisioning",
            error_message: null,
            started_at: Date.now(),
            summary: {
              ...directSummary,
              last_provisioning_recovery_mode: "direct_image",
            },
          });
          return (await getTrainingJob(c.env.DB, job.job_id)) ?? job;
        } catch (error) {
          await updateTrainingJob(c.env.DB, job.job_id, {
            status: "failed",
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
      status: "failed",
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
      status: "failed",
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
      buildTrainingPodEnv(c, job)
    );
    await updateTrainingJob(c.env.DB, job.job_id, {
      runpod_pod_id: newPod.podId,
      status: "provisioning",
      error_message: null,
      started_at: Date.now(),
      summary: nextSummary,
    });
    return (await getTrainingJob(c.env.DB, job.job_id)) ?? job;
  } catch (error) {
    await updateTrainingJob(c.env.DB, job.job_id, {
      status: "failed",
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
  parsedStatus: TrainingStatusBlob | null
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

const recoverStalledActiveJob = async (
  c: Context<AppContext>,
  job: TrainingJob,
  parsedStatus: TrainingStatusBlob | null
): Promise<TrainingJob> => {
  const effectiveStatus =
    typeof parsedStatus?.status === "string" && parsedStatus.status.trim()
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
  const lastMessage = typeof summary.last_message === "string" ? summary.last_message.trim() : "";
  const previousPodIds = Array.isArray(summary.previous_runpod_pod_ids)
    ? summary.previous_runpod_pod_ids.filter((value): value is string => typeof value === "string")
    : [];

  let podStatus: PodStatusDetail | null = null;
  try {
    podStatus = job.runpod_pod_id ? await getPodStatus(c.env, job.runpod_pod_id) : null;
  } catch (error) {
    console.warn(`Failed to inspect pod ${job.runpod_pod_id ?? "unknown"} for stalled job ${job.job_id}:`, error);
  }

  const podState = getProvisioningState(podStatus) || String(podStatus?.runtimeStatus ?? "").trim().toLowerCase() || "unknown";
  const baseSummary = {
    ...summary,
    stall_recovery_attempts: attempts + 1,
    last_stall_stage: effectiveStatus,
    last_stall_activity_at: latestActivityAt,
    last_stall_inactive_ms: inactiveMs,
    last_stall_pod_state: podState,
    last_recovery_at: Date.now(),
    previous_runpod_pod_ids: Array.from(
      new Set([...previousPodIds, ...(job.runpod_pod_id ? [job.runpod_pod_id] : [])])
    ),
  };
  const reason =
    `Training job stalled in ${effectiveStatus} for ${Math.round(inactiveMs / 60000)} minute(s)` +
    (lastMessage ? `; last_message=${lastMessage}` : "") +
    (job.runpod_pod_id ? `; pod=${job.runpod_pod_id}` : "") +
    (podState ? `; pod_state=${podState}` : "");

  if (attempts >= MAX_STALL_RECOVERY_ATTEMPTS) {
    if (job.runpod_pod_id) {
      await terminatePod(c.env, job.runpod_pod_id).catch(() => false);
    }
    await updateTrainingJob(c.env.DB, job.job_id, {
      status: "failed",
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
      status: "failed",
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
      status: "provisioning",
      progress: {},
      error_message: null,
      last_heartbeat_at: null,
      started_at: Date.now(),
      completed_at: null,
      summary: {
        ...baseSummary,
        ...launchResult.summary,
        last_recovery_mode: "restart_same_job",
      },
    });
    return (await getTrainingJob(c.env.DB, job.job_id)) ?? job;
  } catch (error) {
    await updateTrainingJob(c.env.DB, job.job_id, {
      status: "failed",
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
      "안녕하세요.",
      "안녕하세요. 오늘 회의는 오후 두 시에 시작합니다.",
      "안녕하세요. 오늘 회의는 오후 두 시에 시작하고, 발표 자료는 메일로 공유드리겠습니다.",
    ],
    en: [
      "Hello.",
      "Hello. The meeting starts at two o'clock this afternoon.",
      "Hello. The meeting starts at two o'clock this afternoon, and I will share the presentation materials via email.",
    ],
    zh: [
      "你好。",
      "你好。今天的会议下午两点开始。",
      "你好。今天的会议下午两点开始，我会通过邮件分享演示文稿。",
    ],
    ja: [
      "こんにちは。",
      "こんにちは。今日の会議は午後二時に始まります。",
      "こんにちは。今日の会議は午後二時に始まります。プレゼン資料はメールでお送りします。",
    ],
  };

  const fallback = validationTextsByLang[lang] ?? [
    "Hello.",
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

const getSignatureStyleInstruction = (lang: string): string => {
  switch (lang) {
    case "ko":
      return "참고 음성의 특유의 말투와 호흡, 문장 리듬, 억양의 오르내림을 최대한 유지하고 과장하지 말고 자연스럽게 말하세요.";
    case "ja":
      return "参照音声の話し方、間の取り方、文のリズム、抑揚をできるだけ保ち、誇張せず自然に話してください。";
    case "zh":
      return "尽量保留参考音频特有的说话方式、停连节奏、句子律动和语调起伏，不要夸张，保持自然。";
    default:
      return "Preserve the reference voice's natural cadence, pauses, emphasis, and conversational rhythm. Keep the delivery natural and not exaggerated.";
  }
};

const getValidationPresets = (modelId: string, lang = ""): ValidationPreset[] => {
  const is06b = modelId.toLowerCase().includes("0.6b");
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
        name: "balanced",
        payload: { voice_settings: balancedSettings },
        settings: balancedSettings,
      },
      {
        name: "high_similarity",
        payload: { voice_settings: conservativeSettings },
        settings: conservativeSettings,
      },
      {
        name: "signature_style",
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
      name: "balanced",
      payload: { voice_settings: balancedSettings },
      settings: balancedSettings,
    },
    {
      name: "high_similarity",
      payload: { voice_settings: conservativeSettings },
      settings: conservativeSettings,
    },
  ];
};

const getValidationPlan = (
  voice: NonNullable<Awaited<ReturnType<typeof getVoice>>>,
  job: TrainingJob
): ValidationPlan => {
  const modelId = voice.model_id ?? "qwen3-tts-1.7b";
  const is06b = modelId.toLowerCase().includes("0.6b");
  const jobConfig = job.config as Record<string, unknown>;
  const lang =
    typeof jobConfig.whisper_language === "string" ? jobConfig.whisper_language.toLowerCase() : "";
  const validationTexts = getValidationTexts(lang, is06b);
  const validationSeedOffsets = is06b ? FAST_VALIDATION_SEEDS_OFFSET : FULL_VALIDATION_SEEDS_OFFSET;
  return {
    is06b,
    presets: getValidationPresets(modelId, lang),
    validationTexts,
    validationSeedOffsets,
    totalSamples: validationTexts.length * validationSeedOffsets.length,
    minOverall: is06b ? 0.9 : 0.85,
    minPassRate: is06b ? MIN_PASS_RATE_06B : MIN_PASS_RATE_17B,
    minAsrScore: 0.8,
    minToneScore: is06b ? 0.45 : 0.55,
    maxCheckpointsToEval: is06b ? MAX_CHECKPOINTS_TO_EVAL_06B : MAX_CHECKPOINTS_TO_EVAL,
    prioritizeLatestPassingCheckpoint: is06b,
  };
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
}): Array<{ value: number; weight: number }> => {
  const baseWeights = is06b
    ? { overall: 0.38, asr: 0.22, health: 0.14, duration: 0.08, passRate: 0.08, speaker: 0.20, tone: 0.06, speed: 0.06 }
    : { overall: 0.34, asr: 0.18, health: 0.12, duration: 0.06, passRate: 0.06, speaker: 0.18, tone: 0.16, speed: 0.10 };

  const parts: Array<{ value: number; weight: number }> = [
    { value: overall, weight: baseWeights.overall },
    { value: asr, weight: baseWeights.asr },
    { value: health, weight: baseWeights.health },
    { value: duration, weight: baseWeights.duration },
    { value: passRate, weight: baseWeights.passRate },
  ];

  if (Number.isFinite(speaker)) {
    parts.push({ value: speaker, weight: baseWeights.speaker });
  }
  if (Number.isFinite(tone)) {
    parts.push({ value: tone, weight: baseWeights.tone });
  }
  if (Number.isFinite(speed)) {
    parts.push({ value: speed, weight: baseWeights.speed });
  }

  return parts;
};

const serializeTrainingJob = (job: TrainingJob): Omit<TrainingJob, "job_token"> => {
  const { job_token: _jobToken, ...safeJob } = job;
  return safeJob;
};

const normalizeCheckpointEvaluation = (value: unknown): CheckpointEvaluation | null => {
  if (!value || typeof value !== "object") {
    return null;
  }
  const v = value as Record<string, unknown>;
  if (
    typeof v.epoch !== "number" ||
    typeof v.prefix !== "string" ||
    typeof v.ok !== "boolean" ||
    typeof v.score !== "number" ||
    typeof v.message !== "string" ||
    typeof v.preset !== "string" ||
    typeof v.passed_samples !== "number" ||
    typeof v.total_samples !== "number"
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

const normalizePresetSettings = (value: unknown): ValidationPreset["settings"] | undefined => {
  if (!value || typeof value !== "object") {
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
  speaker_samples: 0,
  tone_samples: 0,
  speed_samples: 0,
  first_failure_message: null,
});

const normalizeValidationAccumulator = (value: unknown): AsyncValidationAccumulator => {
  if (!value || typeof value !== "object") {
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
    speaker_samples: Number.isFinite(Number(src.speaker_samples)) ? Number(src.speaker_samples) : 0,
    tone_samples: Number.isFinite(Number(src.tone_samples)) ? Number(src.tone_samples) : 0,
    speed_samples: Number.isFinite(Number(src.speed_samples)) ? Number(src.speed_samples) : 0,
    first_failure_message:
      typeof src.first_failure_message === "string" ? src.first_failure_message : null,
  };
};

const normalizeValidationChampion = (value: unknown): AsyncValidationChampion | null => {
  if (!value || typeof value !== "object") {
    return null;
  }
  const src = value as Record<string, unknown>;
  if (
    typeof src.epoch !== "number" ||
    typeof src.prefix !== "string" ||
    typeof src.score !== "number" ||
    typeof src.message !== "string" ||
    typeof src.preset_name !== "string" ||
    typeof src.passed_samples !== "number" ||
    typeof src.total_samples !== "number"
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
  if (!value || typeof value !== "object") {
    return null;
  }
  const src = value as Record<string, unknown>;
  if (
    typeof src.passed_samples !== "number" ||
    typeof src.score !== "number" ||
    typeof src.message !== "string" ||
    typeof src.preset_name !== "string" ||
    typeof src.total_samples !== "number"
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
  job: TrainingJob
): Promise<{ referenceAudioKey: string | null; referenceText: string }> => {
  const datasetPrefix = String(job.dataset_r2_prefix ?? "").replace(/\/+$/, "");
  let referenceAudioKey = voice.ref_audio_r2_key ?? (datasetPrefix ? `${datasetPrefix}/ref_audio.wav` : null);
  let referenceText = "";

  if (datasetPrefix) {
    try {
      const profileObj = await c.env.R2.get(`${datasetPrefix}/reference_profile.json`);
      if (profileObj) {
        const profile = (await profileObj.json()) as Record<string, unknown>;
        if (typeof profile.reference_audio_key === "string" && profile.reference_audio_key.trim()) {
          referenceAudioKey = profile.reference_audio_key.trim();
        }
        if (typeof profile.reference_text === "string") {
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
}: {
  voice: NonNullable<Awaited<ReturnType<typeof getVoice>>>;
  checkpointPrefix: string;
  preset: ValidationPreset;
  validationText: string;
  seed: number;
  referenceAudioKey: string | null;
  referenceText: string;
}): Record<string, unknown> => ({
  text: validationText,
  voice_id: voice.voice_id,
  speaker_name: voice.speaker_name,
  model_id: voice.model_id ?? "qwen3-tts-1.7b",
  language: "auto",
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
    type: "full",
  },
  ...preset.payload,
});

const getValidationLanguageHint = (
  voice: NonNullable<Awaited<ReturnType<typeof getVoice>>>,
  job: TrainingJob
): string => {
  const jobConfig = job.config as Record<string, unknown>;
  if (typeof jobConfig.whisper_language === "string" && jobConfig.whisper_language.trim()) {
    return jobConfig.whisper_language.trim();
  }
  if (voice.labels && typeof voice.labels.language === "string" && voice.labels.language.trim()) {
    return voice.labels.language.trim();
  }
  return "auto";
};

const annotateAsrFailure = (
  output: { quality?: Record<string, unknown>; audio?: string; error?: unknown } | null,
  error: unknown
): { quality?: Record<string, unknown>; audio?: string; error?: unknown } | null => {
  if (!output) {
    return output;
  }
  const detail = error instanceof Error ? error.message : "OpenAI ASR enrichment failed";
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
  seed: number
): string => {
  const detail = typeof quality[VALIDATION_ASR_ERROR_KEY] === "string" ? quality[VALIDATION_ASR_ERROR_KEY] : null;
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
      typeof output?.error === "string"
        ? output.error
        : fallbackError && fallbackError.trim()
          ? fallbackError
          : "status=unknown no-audio";
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
  if (duration < 0.45) {
    return fail(`sample ${sampleIndex} seed ${seed} duration_score=${duration.toFixed(3)}`);
  }
  if (health < 0.72) {
    return fail(`sample ${sampleIndex} seed ${seed} health_score=${health.toFixed(3)}`);
  }
  if (asr < minAsrScore) {
    return fail(`sample ${sampleIndex} seed ${seed} asr_score=${asr.toFixed(3)}`);
  }
  if (referenceAudioKey && Number.isFinite(speaker) && speaker < 0.75) {
    return fail(`sample ${sampleIndex} seed ${seed} speaker_score=${speaker.toFixed(3)}`);
  }
  if (referenceAudioKey && Number.isFinite(tone) && tone < minToneScore) {
    return fail(`sample ${sampleIndex} seed ${seed} tone_score=${tone.toFixed(3)}`);
  }
  if (referenceAudioKey && referenceText && Number.isFinite(speed) && speed < 0.55) {
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
    failureMessage: null,
  };
};

const applyValidationSampleOutcome = (
  accumulator: AsyncValidationAccumulator,
  outcome: ValidationSampleOutcome
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
  if (typeof outcome.speaker === "number") {
    next.sum_speaker += outcome.speaker;
    next.speaker_samples += 1;
  }
  if (typeof outcome.tone === "number") {
    next.sum_tone += outcome.tone;
    next.tone_samples += 1;
  }
  if (typeof outcome.speed === "number") {
    next.sum_speed += outcome.speed;
    next.speed_samples += 1;
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
    const meanTone = accumulator.tone_samples > 0 ? accumulator.sum_tone / accumulator.tone_samples : NaN;
    const meanSpeed = accumulator.speed_samples > 0 ? accumulator.sum_speed / accumulator.speed_samples : NaN;
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
    });
    const totalWeight = scoreParts.reduce((acc, part) => acc + part.weight, 0) || 1;
    const score =
      scoreParts.reduce((acc, part) => acc + (part.value * part.weight), 0) / totalWeight;
    const similarityNote =
      Number.isFinite(meanSpeaker) && Number.isFinite(meanTone)
        ? `speaker=${meanSpeaker.toFixed(3)} tone=${meanTone.toFixed(3)} `
        : "";
    const speedNote = Number.isFinite(meanSpeed) ? `speed=${meanSpeed.toFixed(3)} ` : "";
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

  return {
    ok: false,
    message:
      `All presets failed: ` +
      `preset=${preset.name} ` +
      (accumulator.first_failure_message ??
        `samples=${accumulator.passed}/${totalSamples} ` +
          `pass_rate=${passRate.toFixed(3)} ` +
          `no_audio=${accumulator.no_audio} infra=${accumulator.infra_issues}`),
    aggregateScore: 0,
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
  const minOverall = is06b ? 0.9 : 0.85;
  const minAsrScore = 0.8;
  const totalSamples = 1;

  if (!output?.audio) {
    const lastErrorDetail =
      typeof output?.error === "string"
        ? output.error
        : fallbackError && fallbackError.trim()
          ? fallbackError
          : "status=unknown no-audio";
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

  const fail = (detail: string): CheckpointValidationResult => ({
    ok: false,
    message: `All presets failed: preset=${preset.name} ${detail}`,
    aggregateScore: 0,
    presetName: preset.name,
    passedSamples: 0,
    totalSamples,
  });

  if (!Number.isFinite(overall) || !Number.isFinite(duration) || !Number.isFinite(health)) {
    return fail("sample 1 seed " + seed + " invalid quality metrics");
  }
  if (!Number.isFinite(asr)) {
    return fail(getMissingAsrMessage(quality, 1, seed));
  }
  if (overall < minOverall) {
    return fail(`sample 1 seed ${seed} overall_score=${overall.toFixed(3)}`);
  }
  if (duration < 0.45) {
    return fail(`sample 1 seed ${seed} duration_score=${duration.toFixed(3)}`);
  }
  if (health < 0.72) {
    return fail(`sample 1 seed ${seed} health_score=${health.toFixed(3)}`);
  }
  if (asr < minAsrScore) {
    return fail(`sample 1 seed ${seed} asr_score=${asr.toFixed(3)}`);
  }
  if (referenceAudioKey && Number.isFinite(speaker) && speaker < 0.75) {
    return fail(`sample 1 seed ${seed} speaker_score=${speaker.toFixed(3)}`);
  }
  if (referenceAudioKey && Number.isFinite(tone) && tone < minToneScore) {
    return fail(`sample 1 seed ${seed} tone_score=${tone.toFixed(3)}`);
  }
  if (referenceAudioKey && referenceText && Number.isFinite(speed) && speed < 0.55) {
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
  });
  const totalWeight = scoreParts.reduce((acc, part) => acc + part.weight, 0) || 1;
  const score = scoreParts.reduce((acc, part) => acc + (part.value * part.weight), 0) / totalWeight;
  const similarityNote = Number.isFinite(speaker) && Number.isFinite(tone)
    ? `speaker=${speaker.toFixed(3)} tone=${tone.toFixed(3)} `
    : "";
  const speedNote = Number.isFinite(speed)
    ? `speed=${speed.toFixed(3)} `
    : "";
  return {
    ok: true,
    message:
      `preset=${preset.name} ` +
      `score=${score.toFixed(3)} overall=${overall.toFixed(3)} ` +
      `asr=${asr.toFixed(3)} ` +
      similarityNote +
      speedNote +
      `health=${health.toFixed(3)} duration=${duration.toFixed(3)} ` +
      "samples=1/1 no_audio=0",
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
  candidateCheckpoints: CheckpointCandidate[]
): Promise<{ status: string; progress: TrainingProgress }> => {
  const plan = getValidationPlan(voice, job);
  if (plan.presets.length === 0 || plan.validationTexts.length === 0 || plan.totalSamples === 0) {
    return {
      status: "completed",
      progress,
    };
  }

  const persistedState =
    currentSummary.async_validation && typeof currentSummary.async_validation === "object"
      ? (currentSummary.async_validation as Record<string, unknown>)
      : null;
  const existingEvaluations = Array.isArray(persistedState?.evaluations)
    ? persistedState.evaluations
        .map(normalizeCheckpointEvaluation)
        .filter((value): value is CheckpointEvaluation => value !== null)
    : [];
  const persistedReferenceAudioKey =
    typeof persistedState?.reference_audio_key === "string" || persistedState?.reference_audio_key === null
      ? (persistedState.reference_audio_key as string | null)
      : null;
  const persistedReferenceText =
    typeof persistedState?.reference_text === "string" ? persistedState.reference_text : "";
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
    evaluations: CheckpointEvaluation[]
  ): Promise<{ status: string; progress: TrainingProgress }> => {
    const promotion = await chooseCheckpointPromotion({
      c,
      voice,
      candidatePrefix: champion.prefix,
      candidateScore: champion.score,
    });

    if (promotion.promote) {
      await updateVoice(c.env.DB, job.voice_id, {
        status: "ready",
        checkpoint_r2_prefix: champion.prefix,
        run_name: parseRunNameFromCheckpointPrefix(champion.prefix),
        epoch: champion.epoch,
        settings: champion.preset_settings,
      });
    }

    const validationMessage = promotion.promote
      ? champion.message
      : `${champion.message} | kept current ready checkpoint score=${(promotion.preservedScore ?? 0).toFixed(3)} >= candidate_score=${champion.score.toFixed(3)}`;
    await updateTrainingJob(c.env.DB, job.job_id, {
      status: "completed",
      completed_at: job.completed_at ?? Date.now(),
      progress,
      summary: {
        ...currentSummary,
        validation_in_progress: false,
        validation_checked: true,
        validation_passed: true,
        validation_message: validationMessage,
        selected_checkpoint_prefix: promotion.promote ? champion.prefix : promotion.preservedPrefix,
        selected_checkpoint_epoch: promotion.promote ? champion.epoch : promotion.preservedEpoch,
        selected_preset: promotion.promote ? champion.preset_name : "kept_existing_best",
        selected_score: promotion.promote ? champion.score : promotion.preservedScore,
        candidate_checkpoint_prefix: champion.prefix,
        candidate_checkpoint_epoch: champion.epoch,
        candidate_preset: champion.preset_name,
        candidate_score: champion.score,
        evaluated_checkpoints: evaluations,
        async_validation: null,
      },
    });
    return { status: "completed", progress };
  };

  const completeFailedValidation = async (
    message: string,
    evaluations: CheckpointEvaluation[]
  ): Promise<{ status: string; progress: TrainingProgress }> => {
    if (
      !shouldKeepReadyVoiceOnValidationFailure(voice, currentSummary, {
        evaluatedCheckpoints: evaluations,
        validationRunName: parseRunNameFromCheckpointPrefix(candidateCheckpoints[0]?.r2_prefix ?? ""),
        forceRevalidation: currentSummary.force_revalidation === true,
      })
    ) {
      await updateVoice(c.env.DB, job.voice_id, {
        status: "created",
        checkpoint_r2_prefix: null,
        run_name: null,
        epoch: null,
      });
    }

    await updateTrainingJob(c.env.DB, job.job_id, {
      status: "completed",
      error_message: null,
      completed_at: job.completed_at ?? Date.now(),
      progress,
      summary: {
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
      },
    });
    return { status: "completed", progress };
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
      return completeFailedValidation("No remaining checkpoints to validate", evaluations);
    }

    const preset = plan.presets[presetIndex];
    const validationText = plan.validationTexts[textIndex];
    const seedOffset = plan.validationSeedOffsets[seedIndex];
    if (!preset || !validationText || typeof seedOffset !== "number") {
      return completeFailedValidation("Validation plan is invalid", evaluations);
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
    });
    const runpodResponse = await invokeServerlessAsync(c.env, c.env.RUNPOD_ENDPOINT_ID, payload);
    const sampleOrdinal = textIndex * plan.validationSeedOffsets.length + seedIndex + 1;
    const nextState: AsyncCheckpointValidationState = {
      mode: "checkpoint_async",
      run_id: String(runpodResponse.id ?? ""),
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
      status: "completed",
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
    return { status: "completed", progress };
  };

  if (!persistedState || String(persistedState.mode ?? "") !== "checkpoint_async") {
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
    typeof persistedState.checkpoint_index === "number" ? persistedState.checkpoint_index : 0;
  const checkpoint = candidateCheckpoints[checkpointIndex];
  if (!checkpoint) {
    const champion = normalizeValidationChampion(persistedState.champion);
    if (champion) {
      return completeSuccessfulValidation(champion, existingEvaluations);
    }
    return completeFailedValidation("No remaining checkpoints to validate", existingEvaluations);
  }

  const presetIndex = typeof persistedState.preset_index === "number" ? persistedState.preset_index : 0;
  const textIndex = typeof persistedState.text_index === "number" ? persistedState.text_index : 0;
  const seedIndex = typeof persistedState.seed_index === "number" ? persistedState.seed_index : 0;
  const preset = plan.presets[presetIndex];
  const seedOffset = plan.validationSeedOffsets[seedIndex];
  if (!preset || typeof seedOffset !== "number" || !plan.validationTexts[textIndex]) {
    return completeFailedValidation("Validation plan is invalid", existingEvaluations);
  }

  const seed = seedOffset + textIndex;
  const runId = typeof persistedState.run_id === "string" ? persistedState.run_id : "";
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
    runId
  );
  const runStartedAt = getValidationRunStartedAt(persistedState, job);
  const runAgeMs = Math.max(0, Date.now() - runStartedAt);
  const rawRunStatus = String(runpodResponse.status ?? "UNKNOWN");
  const runTimedOut =
    rawRunStatus !== "COMPLETED" &&
    rawRunStatus !== "FAILED" &&
    runAgeMs > VALIDATION_RUN_STALE_MS;
  const runStatus = runTimedOut ? "FAILED" : rawRunStatus;
  const sampleOrdinal = textIndex * plan.validationSeedOffsets.length + seedIndex + 1;
  if (runStatus !== "COMPLETED" && runStatus !== "FAILED") {
    await updateTrainingJob(c.env.DB, job.job_id, {
      status: "completed",
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
    return { status: "completed", progress };
  }

  const output = (runpodResponse.output ?? null) as
    | { quality?: Record<string, unknown>; audio?: string; error?: unknown }
    | null;
  let enrichedOutput = runStatus === "COMPLETED" ? output : null;
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
  const fallbackError =
    runTimedOut
      ? `validation request timed out after ${Math.round(runAgeMs / 1000)}s`
      : typeof enrichedOutput?.error === "string"
      ? enrichedOutput.error
      : typeof runpodResponse.error === "string"
        ? runpodResponse.error
        : `runpod_status=${runStatus.toLowerCase()}`;
  const sampleOutcome = evaluateValidationSample({
    output: runStatus === "COMPLETED" ? enrichedOutput : null,
    fallbackError,
    sampleIndex: textIndex + 1,
    seed,
    referenceAudioKey:
      typeof persistedState.reference_audio_key === "string" || persistedState.reference_audio_key === null
        ? (persistedState.reference_audio_key as string | null)
        : null,
    referenceText: typeof persistedState.reference_text === "string" ? persistedState.reference_text : "",
    minOverall: plan.minOverall,
    minAsrScore: plan.minAsrScore,
    minToneScore: plan.minToneScore,
  });
  const nextPresetStats = applyValidationSampleOutcome(
    normalizeValidationAccumulator(persistedState.preset_stats),
    sampleOutcome
  );
  const currentCheckpointBestPassing = normalizeValidationChampion(persistedState.checkpoint_best_passing);
  const currentCheckpointBestFailure = normalizeValidationFailure(persistedState.checkpoint_best_failure);
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
    (!currentCheckpointBestPassing || presetResult.aggregateScore > currentCheckpointBestPassing.score)
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
        message: nextCheckpointBestFailure?.message ?? "Validation failed for all presets",
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
  candidateCheckpoints: CheckpointCandidate[]
): Promise<{ status: string; progress: TrainingProgress }> => {
  const preset = getValidationPresets(
    voice.model_id ?? "qwen3-tts-0.6b",
    typeof (job.config as Record<string, unknown>).whisper_language === "string"
      ? String((job.config as Record<string, unknown>).whisper_language).toLowerCase()
      : String(voice.labels?.language ?? "")
  )[0];
  const validationText = getValidationTexts(
    typeof (job.config as Record<string, unknown>).whisper_language === "string"
      ? String((job.config as Record<string, unknown>).whisper_language).toLowerCase()
      : "",
    true
  )[0];
  const seed = FAST_VALIDATION_SEEDS_OFFSET[0];

  if (!preset || !validationText) {
    return {
      status: "completed",
      progress,
    };
  }

  const persistedState =
    currentSummary.async_validation && typeof currentSummary.async_validation === "object"
      ? currentSummary.async_validation as Record<string, unknown>
      : null;
  const existingEvaluations = Array.isArray(persistedState?.evaluations)
    ? persistedState.evaluations
        .map(normalizeCheckpointEvaluation)
        .filter((value): value is CheckpointEvaluation => value !== null)
    : [];

  const referenceAudioKey =
    typeof persistedState?.reference_audio_key === "string" || persistedState?.reference_audio_key === null
      ? (persistedState.reference_audio_key as string | null)
      : null;
  const referenceText =
    typeof persistedState?.reference_text === "string"
      ? persistedState.reference_text
      : "";

  const startCheckpointValidation = async (
    checkpointIndex: number,
    evaluations: CheckpointEvaluation[]
  ): Promise<{ status: string; progress: TrainingProgress }> => {
    const checkpoint = candidateCheckpoints[checkpointIndex];
    if (!checkpoint) {
      await updateTrainingJob(c.env.DB, job.job_id, {
        status: "completed",
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
          validation_message: "No remaining checkpoints to validate",
          selected_checkpoint_prefix: null,
          selected_checkpoint_epoch: null,
          selected_preset: null,
          selected_score: null,
          evaluated_checkpoints: evaluations,
          async_validation: null,
        },
      });
      return { status: "completed", progress };
    }
    const reference = referenceAudioKey !== null || referenceText
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
    });
    const runpodResponse = await invokeServerlessAsync(c.env, c.env.RUNPOD_ENDPOINT_ID, payload);
    const nextState: Async06bValidationState = {
      mode: "fast_06b_async",
      run_id: String(runpodResponse.id ?? ""),
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
      status: "completed",
      completed_at: job.completed_at ?? Date.now(),
      progress,
      summary: {
        ...currentSummary,
        validation_in_progress: true,
        validation_message: `Validating checkpoint epoch ${checkpoint.epoch} for 0.6B`,
        async_validation: nextState,
      },
    });
    return { status: "completed", progress };
  };

  if (!persistedState || String(persistedState.mode ?? "") !== "fast_06b_async" || typeof persistedState.run_id !== "string") {
    return startCheckpointValidation(existingEvaluations.length, existingEvaluations);
  }

  const currentIndex = typeof persistedState.checkpoint_index === "number" ? persistedState.checkpoint_index : 0;
  const currentEpoch = typeof persistedState.checkpoint_epoch === "number" ? persistedState.checkpoint_epoch : candidateCheckpoints[currentIndex]?.epoch;
  const currentPrefix =
    typeof persistedState.checkpoint_prefix === "string"
      ? persistedState.checkpoint_prefix
      : candidateCheckpoints[currentIndex]?.r2_prefix;

  const runpodResponse = await getServerlessStatusOrSyntheticFailure(
    c.env,
    c.env.RUNPOD_ENDPOINT_ID,
    persistedState.run_id
  );
  const runStartedAt = getValidationRunStartedAt(persistedState, job);
  const runAgeMs = Math.max(0, Date.now() - runStartedAt);
  const rawRunStatus = String(runpodResponse.status ?? "UNKNOWN");
  const runTimedOut =
    rawRunStatus !== "COMPLETED" &&
    rawRunStatus !== "FAILED" &&
    runAgeMs > VALIDATION_RUN_STALE_MS;
  const runStatus = runTimedOut ? "FAILED" : rawRunStatus;
  if (runStatus !== "COMPLETED" && runStatus !== "FAILED") {
    await updateTrainingJob(c.env.DB, job.job_id, {
      status: "completed",
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
    return { status: "completed", progress };
  }

  const output = (runpodResponse.output ?? null) as { quality?: Record<string, unknown>; audio?: string; error?: unknown } | null;
  let enrichedOutput = runStatus === "COMPLETED" ? output : null;
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
  const asyncFailureDetail =
    runTimedOut
      ? `validation request timed out after ${Math.round(runAgeMs / 1000)}s`
      : typeof enrichedOutput?.error === "string"
      ? enrichedOutput.error
      : typeof runpodResponse.error === "string"
        ? runpodResponse.error
        : `runpod_status=${runStatus.toLowerCase()}`;
  const result =
    runStatus === "COMPLETED"
      ? scoreSingleValidationOutput({
          preset,
          seed,
          output: enrichedOutput,
          fallbackError: typeof runpodResponse.error === "string" ? runpodResponse.error : null,
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
      epoch: typeof currentEpoch === "number" ? currentEpoch : candidateCheckpoints[currentIndex].epoch,
      prefix: typeof currentPrefix === "string" ? currentPrefix : candidateCheckpoints[currentIndex].r2_prefix,
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
      typeof currentPrefix === "string" ? currentPrefix : candidateCheckpoints[currentIndex].r2_prefix;
    const promotedEpoch =
      typeof currentEpoch === "number" ? currentEpoch : candidateCheckpoints[currentIndex].epoch;
    const promotion = await chooseCheckpointPromotion({
      c,
      voice,
      candidatePrefix: promotedPrefix,
      candidateScore: result.aggregateScore,
    });

    if (promotion.promote) {
      await updateVoice(c.env.DB, job.voice_id, {
        status: "ready",
        checkpoint_r2_prefix: promotedPrefix,
        run_name: parseRunNameFromCheckpointPrefix(promotedPrefix),
        epoch: promotedEpoch,
        settings: result.presetSettings,
      });
    }

    const validationMessage = promotion.promote
      ? result.message
      : `${result.message} | kept current ready checkpoint score=${(promotion.preservedScore ?? 0).toFixed(3)} >= candidate_score=${result.aggregateScore.toFixed(3)}`;
    await updateTrainingJob(c.env.DB, job.job_id, {
      status: "completed",
      completed_at: job.completed_at ?? Date.now(),
      progress,
        summary: {
          ...currentSummary,
          force_revalidation: false,
          validation_in_progress: false,
          validation_checked: true,
          validation_passed: true,
        validation_message: validationMessage,
        selected_checkpoint_prefix: promotion.promote ? promotedPrefix : promotion.preservedPrefix,
        selected_checkpoint_epoch: promotion.promote ? promotedEpoch : promotion.preservedEpoch,
        selected_preset: promotion.promote ? result.presetName : "kept_existing_best",
        selected_score: promotion.promote ? result.aggregateScore : promotion.preservedScore,
        candidate_checkpoint_prefix: promotedPrefix,
        candidate_checkpoint_epoch: promotedEpoch,
        candidate_preset: result.presetName,
        candidate_score: result.aggregateScore,
        evaluated_checkpoints: nextEvaluations,
        async_validation: null,
      },
    });
    return { status: "completed", progress };
  }

  const nextIndex = currentIndex + 1;
  if (nextIndex < candidateCheckpoints.length) {
    await updateTrainingJob(c.env.DB, job.job_id, {
      status: "completed",
      completed_at: job.completed_at ?? Date.now(),
      progress,
      summary: {
        ...currentSummary,
        evaluated_checkpoints: nextEvaluations,
      },
    });
    return startCheckpointValidation(nextIndex, nextEvaluations);
  }

  if (
    !shouldKeepReadyVoiceOnValidationFailure(voice, currentSummary, {
      evaluatedCheckpoints: nextEvaluations,
      validationRunName: parseRunNameFromCheckpointPrefix(candidateCheckpoints[0]?.r2_prefix ?? ""),
      forceRevalidation: currentSummary.force_revalidation === true,
    })
  ) {
    await updateVoice(c.env.DB, job.voice_id, {
      status: "created",
      checkpoint_r2_prefix: null,
      run_name: null,
      epoch: null,
    });
  }

  await updateTrainingJob(c.env.DB, job.job_id, {
    status: "completed",
    error_message: null,
    completed_at: job.completed_at ?? Date.now(),
    progress,
    summary: {
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
    },
  });
  return { status: "completed", progress };
};

const validateTrainedCheckpoint = async (
  c: Context<AppContext>,
  voice: NonNullable<Awaited<ReturnType<typeof getVoice>>>,
  job: TrainingJob,
  checkpointPrefix: string
): Promise<CheckpointValidationResult> => {
  const jobConfig = job.config as Record<string, unknown>;
  const lang = typeof jobConfig.whisper_language === "string" ? jobConfig.whisper_language.toLowerCase() : "";
  const datasetPrefix = String(job.dataset_r2_prefix ?? "").replace(/\/+$/, "");
  const presets = getValidationPresets(voice.model_id ?? "qwen3-tts-1.7b", lang);
  const is06b = String(voice.model_id ?? "").toLowerCase().includes("0.6b");
  const validationTexts = getValidationTexts(lang, is06b);
  const validationSeedOffsets = is06b ? FAST_VALIDATION_SEEDS_OFFSET : FULL_VALIDATION_SEEDS_OFFSET;
  const totalSamples = validationTexts.length * validationSeedOffsets.length;
  const minOverall = is06b ? 0.9 : 0.85;
  const minPassRate = is06b ? MIN_PASS_RATE_06B : MIN_PASS_RATE_17B;
  const minAsrScore = 0.8;
  let referenceAudioKey = voice.ref_audio_r2_key ?? (datasetPrefix ? `${datasetPrefix}/ref_audio.wav` : null);
  let referenceText = "";

  if (datasetPrefix) {
    try {
      const profileObj = await c.env.R2.get(`${datasetPrefix}/reference_profile.json`);
      if (profileObj) {
        const profile = (await profileObj.json()) as Record<string, unknown>;
        if (typeof profile.reference_audio_key === "string" && profile.reference_audio_key.trim()) {
          referenceAudioKey = profile.reference_audio_key.trim();
        }
        if (typeof profile.reference_text === "string") {
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
      let speakerSamples = 0;
      let toneSamples = 0;
      let speedSamples = 0;
      let firstFailureMessage: string | null = null;

      for (let i = 0; i < validationTexts.length; i += 1) {
        for (const seedOffset of validationSeedOffsets) {
          const seed = seedOffset + i;
          const payload: Record<string, unknown> = {
            text: validationTexts[i],
            voice_id: voice.voice_id,
            speaker_name: voice.speaker_name,
            model_id: voice.model_id ?? "qwen3-tts-1.7b",
            language: "auto",
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
              type: "full",
            },
            ...preset.payload,
          };

          let response: Record<string, unknown> | null = null;
          let output: { quality?: Record<string, unknown>; audio?: string; error?: unknown } | null = null;
          let lastErrorDetail = "unknown";

          for (let attempt = 1; attempt <= VALIDATION_RETRY_ATTEMPTS; attempt += 1) {
            response = await invokeServerless(c.env, c.env.RUNPOD_ENDPOINT_ID, payload);
            output = (response.output ?? {}) as { quality?: Record<string, unknown>; audio?: string; error?: unknown };

            if (output.audio) {
              break;
            }

            const statusText = String(response.status ?? "unknown");
            const outputError =
              typeof output.error === "string"
                ? output.error
                : typeof response.error === "string"
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
            output = (await enrichOutputWithReviewAsr({
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
          if (duration < 0.45) {
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
          if (referenceAudioKey && Number.isFinite(speaker) && speaker < 0.75) {
            if (!firstFailureMessage) {
              firstFailureMessage = `sample ${i + 1} seed ${seed} speaker_score=${speaker.toFixed(3)}`;
            }
            continue;
          }
          if (referenceAudioKey && Number.isFinite(tone) && tone < 0.45) {
            if (!firstFailureMessage) {
              firstFailureMessage = `sample ${i + 1} seed ${seed} tone_score=${tone.toFixed(3)}`;
            }
            continue;
          }
          if (referenceAudioKey && referenceText && Number.isFinite(speed) && speed < 0.55) {
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
        }
      }

      const passRate = totalSamples > 0 ? passed / totalSamples : 0;
      if (passed > 0 && passRate >= minPassRate && infraIssues === 0) {
        const n = Math.max(1, passed);
        const meanOverall = sumOverall / n;
        const meanDuration = sumDuration / n;
        const meanHealth = sumHealth / n;
        const meanAsr = sumAsr / n;
        const meanSpeaker = speakerSamples > 0 ? (sumSpeaker / speakerSamples) : NaN;
        const meanTone = toneSamples > 0 ? (sumTone / toneSamples) : NaN;
        const meanSpeed = speedSamples > 0 ? (sumSpeed / speedSamples) : NaN;
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
        });
        const totalWeight = scoreParts.reduce((acc, part) => acc + part.weight, 0) || 1;
        const score = scoreParts.reduce((acc, part) => acc + (part.value * part.weight), 0) / totalWeight;
        const similarityNote = Number.isFinite(meanSpeaker) && Number.isFinite(meanTone)
          ? `speaker=${meanSpeaker.toFixed(3)} tone=${meanTone.toFixed(3)} `
          : "";
        const speedNote = Number.isFinite(meanSpeed)
          ? `speed=${meanSpeed.toFixed(3)} `
          : "";
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
        bestFailure = { passed, message: `preset=${preset.name} ${failureSummary}`, presetName: preset.name };
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
      message: `All presets failed: ${bestFailure?.message ?? "unknown"}`,
      aggregateScore: 0,
      presetName: bestFailure?.presetName ?? "default",
      passedSamples: bestFailure?.passed ?? 0,
      totalSamples,
    };
  } catch (error) {
    return {
      ok: false,
      message: `Validation invocation failed: ${error instanceof Error ? error.message : "unknown"}`,
      aggregateScore: 0,
      presetName: "default",
      passedSamples: 0,
      totalSamples,
    };
  }
};

const reconcileJobStatus = async (
  c: Context<AppContext>,
  job: TrainingJob
): Promise<{ status: string; progress: TrainingProgress }> => {
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
      completed_at: status === "completed" ? job.completed_at ?? Date.now() : job.completed_at,
    });
    currentJob = {
      ...job,
      status,
      progress,
      completed_at: status === "completed" ? job.completed_at ?? Date.now() : job.completed_at,
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

  if (currentJob.status === "completed" && !currentJob.summary?.validation_checked) {
    const voice = await getVoice(c.env.DB, job.voice_id);
    if (!voice) {
      const message = "Voice record missing for validation";
      await updateTrainingJob(c.env.DB, job.job_id, {
        status: "failed",
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
      return { status: "failed", progress };
    }

    if (!Array.isArray(parsedStatus.checkpoints) || parsedStatus.checkpoints.length === 0) {
      const message = "Training completed but no checkpoints found in status payload";
      await updateVoice(c.env.DB, job.voice_id, {
        status: "created",
      });
      await updateTrainingJob(c.env.DB, job.job_id, {
        status: "failed",
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
      return { status: "failed", progress };
    }

    const currentSummary = (currentJob.summary ?? {}) as Record<string, unknown>;
    const forceRevalidation = currentSummary.force_revalidation === true;
    const persistedReadyCheckpoint =
      voice.status === "ready" &&
      typeof voice.checkpoint_r2_prefix === "string" &&
      parsedStatus.checkpoints.some((cp) => cp?.r2_prefix === voice.checkpoint_r2_prefix)
        ? voice.checkpoint_r2_prefix
        : null;

    if (persistedReadyCheckpoint && !forceRevalidation) {
      await updateTrainingJob(c.env.DB, job.job_id, {
        status: "completed",
        completed_at: currentJob.completed_at ?? Date.now(),
        progress,
        summary: {
          ...currentSummary,
          validation_checked: true,
          validation_passed: true,
          validation_message:
            typeof currentSummary.validation_message === "string" && currentSummary.validation_message.trim()
              ? currentSummary.validation_message
              : "Recovered validation result from persisted ready voice checkpoint",
          selected_checkpoint_prefix: persistedReadyCheckpoint,
          selected_checkpoint_epoch: voice.epoch,
          selected_preset:
            typeof currentSummary.selected_preset === "string" && currentSummary.selected_preset.trim()
              ? currentSummary.selected_preset
              : "high_similarity",
        },
      });
      return { status: "completed", progress };
    }

    const validationPlan = getValidationPlan(voice, job);
    const candidateCheckpoints = parsedStatus.checkpoints
      .filter(
        (cp): cp is { epoch: number; r2_prefix: string } =>
          typeof cp.epoch === "number" && typeof cp.r2_prefix === "string"
      )
      .sort((a, b) => b.epoch - a.epoch)
      .slice(0, validationPlan.maxCheckpointsToEval);

    if (candidateCheckpoints.length === 0) {
      const message = "Training completed but checkpoint metadata is invalid";
      await updateVoice(c.env.DB, job.voice_id, {
        status: "created",
      });
      await updateTrainingJob(c.env.DB, job.job_id, {
        status: "failed",
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
      return { status: "failed", progress };
    }

    return advanceAsyncCheckpointValidation(c, voice, currentJob, progress, currentSummary, candidateCheckpoints);
  }

  return { status: currentJob.status, progress: currentJob.progress };
};

const RECONCILE_TIMEOUT_MS = 25000;
const COMPLETED_VALIDATION_TIMEOUT_MS = 180000;

const getExecutionCtx = (c: Context<AppContext>): ExecutionContext | undefined =>
  (c as unknown as { executionCtx?: ExecutionContext }).executionCtx;

const waitForBackgroundTask = (c: Context<AppContext>, promise: Promise<unknown>) => {
  getExecutionCtx(c)?.waitUntil(
    promise
      .then(() => undefined)
      .catch(() => undefined)
  );
};

const reconcileJobStatusWithTimeout = async (
  c: Context<AppContext>,
  job: TrainingJob,
  timeoutMs = RECONCILE_TIMEOUT_MS
): Promise<boolean> => {
  let timedOut = false;
  const reconcilePromise = reconcileJobStatus(c, job);
  await Promise.race([
    reconcilePromise,
    new Promise<void>((resolve) =>
      setTimeout(() => {
        timedOut = true;
        resolve();
      }, timeoutMs)
    ),
  ]);
  if (timedOut) {
    waitForBackgroundTask(c, reconcilePromise);
  }
  return !timedOut;
};

export const runTrainingSupervisorSweep = async (
  env: Env
): Promise<{ checked: number; reconciled: number; timed_out: number; failed: number }> => {
  const jobs = await listTrainingJobs(env.DB, { limit: TRAINING_SWEEP_LIMIT });
  const candidates = jobs.filter((job) => ACTIVE_JOB_STATUSES.has(job.status) || needsCompletedValidation(job));
  const syntheticContext = createSyntheticContext(env, resolveWorkerPublicUrl(env));
  let reconciled = 0;
  let timedOut = 0;
  let failed = 0;

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

  return {
    checked: candidates.length,
    reconciled,
    timed_out: timedOut,
    failed,
  };
};

app.post("/start", async (c) => {
  const body = (await c.req.json()) as {
    voice_id?: string;
    dataset_name?: string;
    config?: TrainingConfig;
  };

  if (!body.voice_id) {
    return c.json({ detail: { message: "voice_id is required" } }, 400);
  }

  const voice = await getVoice(c.env.DB, body.voice_id);
  if (!voice) {
    return c.json({ detail: { message: "Voice not found" } }, 404);
  }

  const activeJobs = await listTrainingJobs(c.env.DB, { voice_id: body.voice_id, limit: 10 });
  const hasActiveJob = activeJobs.some((j) => ACTIVE_JOB_STATUSES.has(j.status));
  if (hasActiveJob) {
    return c.json(
      { detail: { message: "A training job is already active for this voice. Cancel it first or wait for completion." } },
      409
    );
  }

  const now = Date.now();
  const jobId = crypto.randomUUID();
  const jobToken = crypto.randomUUID();
  const workerUrl = getWorkerOrigin(c);
  const runName = `run_${jobId.slice(0, 8)}`;
  const datasetPrefix = resolveTrainingDatasetPrefix(voice, body.dataset_name);
  const config = body.config ?? {};
  const cfg = config as Record<string, unknown>;
  const modelSize = typeof config.model_size === "string" && config.model_size
    ? config.model_size
    : (voice.model_size || "1.7B");
  const recommendedDefaults = getRecommendedTrainingDefaults(modelSize);
  const effectiveConfig: TrainingConfig = {
    ...config,
    model_size: modelSize,
    batch_size: Number(config.batch_size ?? recommendedDefaults.batch_size),
    learning_rate: Number(config.learning_rate ?? recommendedDefaults.learning_rate),
    num_epochs: Number(config.num_epochs ?? cfg.epochs ?? recommendedDefaults.num_epochs),
    gradient_accumulation_steps: Number(cfg.gradient_accumulation_steps ?? recommendedDefaults.gradient_accumulation_steps),
    subtalker_loss_weight: Number(cfg.subtalker_loss_weight ?? recommendedDefaults.subtalker_loss_weight),
    save_every_n_epochs: Number(cfg.save_every_n_epochs ?? recommendedDefaults.save_every_n_epochs),
    seed: Number(cfg.seed ?? recommendedDefaults.seed),
    gpu_type_id:
      typeof cfg.gpu_type_id === "string" && cfg.gpu_type_id
        ? cfg.gpu_type_id
        : recommendedDefaults.gpu_type_id,
  };
  if (typeof cfg.whisper_language === "string" && cfg.whisper_language.trim()) {
    effectiveConfig.whisper_language = cfg.whisper_language.trim();
  }

  const numEpochs = Number(effectiveConfig.num_epochs ?? recommendedDefaults.num_epochs);
  const batchSize = Number(effectiveConfig.batch_size ?? recommendedDefaults.batch_size);
  const maxSteps = Number(cfg.max_steps ?? 0);

  if (numEpochs < 1 || numEpochs > 30) {
    return c.json({ detail: { message: "num_epochs must be between 1 and 30" } }, 400);
  }
  if (batchSize < 1 || batchSize > 16) {
    return c.json({ detail: { message: "batch_size must be between 1 and 16" } }, 400);
  }
  if (maxSteps < 0 || maxSteps > 100000) {
    return c.json({ detail: { message: "max_steps must be between 0 and 100000" } }, 400);
  }

  const job: TrainingJob = {
    job_id: jobId,
    voice_id: body.voice_id,
    runpod_pod_id: null,
    job_token: jobToken,
    status: "pending",
    config: effectiveConfig,
    progress: {},
    summary: {},
    metrics: {},
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
    voice_id: body.voice_id,
    dataset_r2_prefix: datasetPrefix,
    speaker_name: voice.speaker_name,
    model_size: modelSize,
    batch_size: Number(effectiveConfig.batch_size ?? recommendedDefaults.batch_size),
    learning_rate: Number(effectiveConfig.learning_rate ?? recommendedDefaults.learning_rate),
    num_epochs: Number(effectiveConfig.num_epochs ?? recommendedDefaults.num_epochs),
    run_name: runName,
    gradient_accumulation_steps: Number(effectiveConfig.gradient_accumulation_steps ?? recommendedDefaults.gradient_accumulation_steps),
    speaker_id: Number(cfg.speaker_id ?? 3000),
    mixed_precision: String(cfg.mixed_precision ?? "bf16"),
    torch_dtype: String(cfg.torch_dtype ?? "bfloat16"),
    attn_implementation: String(cfg.attn_implementation ?? "sdpa"),
    weight_decay: Number(cfg.weight_decay ?? 0.01),
    max_grad_norm: Number(cfg.max_grad_norm ?? 1.0),
    subtalker_loss_weight: Number(effectiveConfig.subtalker_loss_weight ?? recommendedDefaults.subtalker_loss_weight),
    log_every_n_steps: Number(cfg.log_every_n_steps ?? 10),
    save_every_n_epochs: Number(effectiveConfig.save_every_n_epochs ?? recommendedDefaults.save_every_n_epochs),
    max_steps: Number(cfg.max_steps ?? 0),
    seed: Number(effectiveConfig.seed ?? recommendedDefaults.seed),
    job_token: jobToken,
    worker_api_url: workerUrl,
    whisper_language: typeof effectiveConfig.whisper_language === "string" ? effectiveConfig.whisper_language : undefined,
  };
  await c.env.R2.put(`jobs/${jobId}/config.json`, JSON.stringify(jobConfig), {
    httpMetadata: { contentType: "application/json" },
  });

  // Default GPU based on model size: 1.7B needs ≥40GB VRAM, 0.6B fits on 24GB
  const gpuTypeId = getTrainingGpuType(job);

  const datasetListing = await c.env.R2.list({ prefix: `${datasetPrefix}/`, limit: 1 });
  if (!datasetListing.objects || datasetListing.objects.length === 0) {
    await updateTrainingJob(c.env.DB, jobId, {
      status: "failed",
      error_message: `Dataset not found at R2 prefix: ${datasetPrefix}/`,
      completed_at: Date.now(),
    });
    return c.json(
      { detail: { message: `Dataset not found at R2 prefix: ${datasetPrefix}/. Upload audio files first.` } },
      400
    );
  }

  let launchResult: Awaited<ReturnType<typeof createTrainingPodForJob>>;
  try {
    launchResult = await createTrainingPodForJob(c, job);
  } catch (podError) {
    // Pod creation failed (e.g., GPU supply constraint) — mark job as failed
    const errMsg = podError instanceof Error ? podError.message : String(podError);
    await updateTrainingJob(c.env.DB, jobId, {
      status: "failed",
      error_message: errMsg,
      completed_at: Date.now(),
    });
    // Re-throw to let the global error handler return appropriate status
    throw podError;
  }

  await updateTrainingJob(c.env.DB, jobId, {
    runpod_pod_id: launchResult.pod.podId,
    status: "provisioning",
    started_at: Date.now(),
    summary: {
      ...(job.summary ?? {}),
      ...launchResult.summary,
    },
  });

  const persistedJob = await getTrainingJob(c.env.DB, jobId);
  if (!persistedJob) {
    return c.json({ detail: { message: "Training job not found" } }, 404);
  }

  const reconciled = await reconcileJobStatus(c, persistedJob);

  return c.json({
    job_id: jobId,
    status: reconciled.status,
    progress: reconciled.progress,
  });
});

app.get("/jobs", async (c) => {
  const voiceId = c.req.query("voice_id")?.trim();
  const limitRaw = c.req.query("limit");
  const parsedLimit = Number(limitRaw ?? "20");
  const limit = Number.isFinite(parsedLimit) ? parsedLimit : 20;

  const jobs = await listTrainingJobs(c.env.DB, {
    voice_id: voiceId || undefined,
    limit,
  });

  const hydratedJobs = await Promise.all(
    jobs.map(async (job) => {
      try {
        if (!ACTIVE_JOB_STATUSES.has(job.status)) {
          if (!needsCompletedValidation(job)) {
            return job;
          }
        }
        await reconcileJobStatusWithTimeout(c, job);
        return (await getTrainingJob(c.env.DB, job.job_id)) ?? job;
      } catch (error) {
        console.warn(`Failed to hydrate training job ${job.job_id}:`, error);
        return job;
      }
    })
  );

  return c.json({ jobs: hydratedJobs.map(serializeTrainingJob) });
});

app.get("/:job_id/logs", async (c) => {
  const jobId = c.req.param("job_id");
  const job = await getTrainingJob(c.env.DB, jobId);
  if (!job) {
    return c.json({ detail: { message: "Training job not found" } }, 404);
  }

  const limitRaw = c.req.query("limit");
  const cursorRaw = c.req.query("cursor");
  const parsedLimit = Number(limitRaw ?? "50");
  const limit = Number.isFinite(parsedLimit) ? parsedLimit : 50;
  const parsedCursor =
    typeof cursorRaw === "string" && cursorRaw.trim().length > 0 ? Number(cursorRaw) : NaN;
  const cursor = Number.isFinite(parsedCursor) ? parsedCursor : undefined;

  const chunks = await listTrainingLogChunks(c.env.DB, jobId, limit, cursor);
  const nextCursor = chunks.length > 0 ? chunks[chunks.length - 1].seq : null;

  return c.json({
    job_id: jobId,
    chunks,
    next_cursor: nextCursor,
  });
});

app.get("/:job_id/logs/:seq", async (c) => {
  const jobId = c.req.param("job_id");
  const seq = Number(c.req.param("seq"));
  if (!Number.isInteger(seq)) {
    return c.json({ detail: { message: "Invalid seq" } }, 400);
  }

  const job = await getTrainingJob(c.env.DB, jobId);
  if (!job) {
    return c.json({ detail: { message: "Training job not found" } }, 404);
  }

  const chunk = await getTrainingLogChunk(c.env.DB, jobId, seq);
  if (!chunk) {
    return c.json({ detail: { message: "Log chunk not found" } }, 404);
  }

  const obj = await c.env.R2.get(chunk.r2_key);
  if (!obj) {
    return c.json({ detail: { message: "R2 log chunk not found" } }, 404);
  }

  const contentType = obj.httpMetadata?.contentType || "application/jsonl";
  return new Response(obj.body, {
    headers: {
      "content-type": contentType,
      "cache-control": "no-store",
    },
  });
});

app.get("/:job_id", async (c) => {
  const jobId = c.req.param("job_id");
  const job = await getTrainingJob(c.env.DB, jobId);

  if (!job) {
    return c.json({ detail: { message: "Training job not found" } }, 404);
  }

  if (ACTIVE_JOB_STATUSES.has(job.status) || needsCompletedValidation(job)) {
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

app.get("/debug/template/:template_id", async (c) => {
  const templateId = c.req.param("template_id");
  try {
    const template = await getTemplateById(c.env, templateId);
    if (!template) {
      return c.json({ detail: { message: "Template not found" } }, 404);
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
          message: error instanceof Error ? error.message : "Failed to fetch template",
        },
      },
      502
    );
  }
});

app.post("/:job_id/reconcile", async (c) => {
  const jobId = c.req.param("job_id");
  const job = await getTrainingJob(c.env.DB, jobId);
  if (!job) {
    return c.json({ detail: { message: "Training job not found" } }, 404);
  }

  if (!ACTIVE_JOB_STATUSES.has(job.status) && !needsCompletedValidation(job)) {
    return c.json({
      status: "noop",
      job: serializeTrainingJob(job),
    });
  }

  waitForBackgroundTask(c, reconcileJobStatus(c, job));

  return c.json({
    status: "accepted",
    validation_started: true,
    job_id: jobId,
  });
});

app.post("/:job_id/revalidate", async (c) => {
  const jobId = c.req.param("job_id");
  const job = await getTrainingJob(c.env.DB, jobId);

  if (!job) {
    return c.json({ detail: { message: "Training job not found" } }, 404);
  }
  if (ACTIVE_JOB_STATUSES.has(job.status)) {
    return c.json({ detail: { message: "Cannot revalidate an active training job" } }, 409);
  }

  const summary = (job.summary ?? {}) as Record<string, unknown>;
  const {
    validation_checked: _validationChecked,
    validation_passed: _validationPassed,
    validation_failed: _validationFailed,
    validation_in_progress: _validationInProgress,
    validation_message: _validationMessage,
    evaluated_checkpoints: _evaluatedCheckpoints,
    async_validation: _asyncValidation,
    selected_checkpoint_prefix: _selectedCheckpointPrefix,
    selected_checkpoint_epoch: _selectedCheckpointEpoch,
    selected_preset: _selectedPreset,
    selected_score: _selectedScore,
    ...restSummary
  } = summary;

  await updateTrainingJob(c.env.DB, jobId, {
    status: "completed",
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
    return c.json({ detail: { message: "Training job not found after reset" } }, 404);
  }

  const reconciled = await reconcileJobStatusWithTimeout(c, refreshedJob, COMPLETED_VALIDATION_TIMEOUT_MS);

  const updatedJob = await getTrainingJob(c.env.DB, jobId);
  return c.json({
    status: reconciled ? "started" : "accepted",
    job: serializeTrainingJob(updatedJob ?? refreshedJob),
  });
});

app.post("/:job_id/promote", async (c) => {
  const jobId = c.req.param("job_id");
  const body = (await c.req.json().catch(() => ({}))) as {
    checkpoint_prefix?: string;
  };
  const checkpointPrefix = typeof body.checkpoint_prefix === "string" ? body.checkpoint_prefix.trim() : "";
  if (!checkpointPrefix) {
    return c.json({ detail: { message: "checkpoint_prefix is required" } }, 400);
  }

  const job = await getTrainingJob(c.env.DB, jobId);
  if (!job) {
    return c.json({ detail: { message: "Training job not found" } }, 404);
  }

  const voice = await getVoice(c.env.DB, job.voice_id);
  if (!voice) {
    return c.json({ detail: { message: "Voice not found" } }, 404);
  }

  const summary = (job.summary ?? {}) as Record<string, unknown>;
  const candidate = collectManualPromotionCandidates(summary).find((value) => value.prefix === checkpointPrefix);
  if (!candidate) {
    return c.json(
      { detail: { message: "checkpoint_prefix was not found among the job's evaluated checkpoints" } },
      404
    );
  }

  await updateVoice(c.env.DB, job.voice_id, {
    status: "ready",
    checkpoint_r2_prefix: candidate.prefix,
    run_name: parseRunNameFromCheckpointPrefix(candidate.prefix),
    epoch: candidate.epoch,
    settings: resolvePromotionSettings(voice, candidate.preset),
  });

  await updateTrainingJob(c.env.DB, jobId, {
    summary: {
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
    },
  });

  const updatedVoice = await getVoice(c.env.DB, job.voice_id);
  const updatedJob = await getTrainingJob(c.env.DB, jobId);
  return c.json({
    status: "ok",
    voice: updatedVoice,
    job: updatedJob ? serializeTrainingJob(updatedJob) : serializeTrainingJob(job),
  });
});

app.post("/:job_id/cancel", async (c) => {
  const jobId = c.req.param("job_id");
  const job = await getTrainingJob(c.env.DB, jobId);

  if (!job) {
    return c.json({ detail: { message: "Training job not found" } }, 404);
  }

  if (job.runpod_pod_id) {
    try {
      await terminatePod(c.env, job.runpod_pod_id);
    } catch {
      // Pod may already be terminated — safe to ignore
    }
  }

  await updateTrainingJob(c.env.DB, jobId, {
    status: "cancelled",
    completed_at: Date.now(),
  });

  return c.json({ status: "ok" });
});

export default app;
