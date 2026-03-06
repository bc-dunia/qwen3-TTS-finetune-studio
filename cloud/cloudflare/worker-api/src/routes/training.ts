import { Hono } from "hono";
import type { Context } from "hono";
import {
  createTrainingJob,
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
  getPodStatus,
  getTemplateById,
  invokeServerless,
  terminatePod,
} from "../lib/runpod";
import { authMiddleware } from "../middleware/auth";
import type { AppContext, TrainingConfig, TrainingJob, TrainingProgress } from "../types";

const app = new Hono<AppContext>();
app.use("*", authMiddleware);

type TrainingStatusBlob = {
  status?: string;
  progress?: TrainingProgress;
  checkpoints?: Array<{ epoch?: number; r2_prefix?: string }>;
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

const needsCompletedValidation = (job: TrainingJob): boolean => {
  if (job.status !== "completed") {
    return false;
  }
  const summary = (job.summary ?? {}) as Record<string, unknown>;
  return summary.validation_checked !== true;
};

const parseRunNameFromCheckpointPrefix = (prefix: string): string | null => {
  const parts = prefix.split("/");
  if (parts.length < 4 || parts[0] !== "checkpoints") {
    return null;
  }
  return parts[2] || null;
};

type PodStatusDetail = NonNullable<Awaited<ReturnType<typeof getPodStatus>>>;

const VALIDATION_SEEDS_OFFSET = [123456, 223456] as const;
const MAX_CHECKPOINTS_TO_EVAL = 4;
const VALIDATION_RETRY_ATTEMPTS = 3;
const MIN_PASS_RATE_06B = 5 / 6;
const MIN_PASS_RATE_17B = 5 / 6;
const PROVISIONING_STALE_MS = 4 * 60 * 1000;
const MAX_PROVISIONING_RECOVERY_ATTEMPTS = 2;

const OVERALL_SCORE_ERROR_RE = /overall_score=([0-9.]+)/i;

const getWorkerOrigin = (c: Context<AppContext>): string => new URL(c.req.url).origin;
const GHCR_INDEX_ACCEPT =
  "application/vnd.oci.image.index.v1+json, application/vnd.docker.distribution.manifest.list.v2+json, application/vnd.docker.distribution.manifest.v2+json, application/vnd.oci.image.manifest.v1+json";

const readNumber = (value: unknown): number | null => {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
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

  const pod = await createPod(
    c.env,
    c.env.RUNPOD_TRAINING_TEMPLATE_ID,
    getTrainingGpuType(job),
    buildTrainingPodEnv(c, job)
  );
  return {
    pod,
    summary: {
      training_launch_mode: "template",
      training_template_id: c.env.RUNPOD_TRAINING_TEMPLATE_ID,
    },
  };
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

  if (!isProvisioningPodStalled(podStatus)) {
    return job;
  }
  if (!podStatus) {
    return job;
  }

  const summary = (job.summary ?? {}) as Record<string, unknown>;
  const attempts = readNumber(summary.provisioning_recovery_attempts) ?? 0;
  const podState = getProvisioningState(podStatus) || "unknown";
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
    `state=${podState} uptime=${podStatus.uptimeSeconds ?? podStatus.runtime?.uptimeInSeconds ?? "n/a"}s ` +
    `image=${podStatus.imageName ?? "unknown"} template=${podStatus.templateId ?? "unknown"}`;

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

  const hasTriedDirectFallback = summary.provisioning_direct_fallback_attempted === true;
  const hasTriedDigestFallback = summary.provisioning_digest_fallback_attempted === true;
  if (podStatus.imageName) {
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

  await terminatePod(c.env, job.runpod_pod_id).catch(() => false);

  try {
    const newPod = await createPod(
      c.env,
      c.env.RUNPOD_TRAINING_TEMPLATE_ID,
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

const parseOverallFromError = (message: string): number | null => {
  const m = OVERALL_SCORE_ERROR_RE.exec(message);
  if (!m || !m[1]) return null;
  const v = Number(m[1]);
  return Number.isFinite(v) ? v : null;
};

const getValidationTexts = (lang: string): string[] => {
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

  return validationTextsByLang[lang] ?? [
    "Hello.",
    "Hello. The meeting starts at two o'clock this afternoon.",
    "Hello. The meeting starts at two o'clock this afternoon, and I will share the presentation materials via email.",
  ];
};

const getValidationPresets = (modelId: string): ValidationPreset[] => {
  const is06b = modelId.toLowerCase().includes("0.6b");
  const conservativeSettings = {
    stability: 0.9,
    similarity_boost: 0.9,
    style: 0.05,
    speed: 1.0,
  };

  if (!is06b) {
    return [
      { name: "default", payload: {} },
      {
        name: "high_similarity",
        payload: { voice_settings: conservativeSettings },
        settings: conservativeSettings,
      },
    ];
  }

  return [
    { name: "default", payload: {} },
    {
      name: "conservative",
      payload: { voice_settings: conservativeSettings },
      settings: conservativeSettings,
    },
  ];
};

const serializeTrainingJob = (job: TrainingJob): Omit<TrainingJob, "job_token"> => {
  const { job_token: _jobToken, ...safeJob } = job;
  return safeJob;
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
  const validationTexts = getValidationTexts(lang);
  const presets = getValidationPresets(voice.model_id ?? "qwen3-tts-1.7b");
  const totalSamples = validationTexts.length * VALIDATION_SEEDS_OFFSET.length;
  const is06b = String(voice.model_id ?? "").toLowerCase().includes("0.6b");
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
        for (const seedOffset of VALIDATION_SEEDS_OFFSET) {
          const seed = seedOffset + i;
          const payload: Record<string, unknown> = {
            text: validationTexts[i],
            voice_id: voice.voice_id,
            speaker_name: voice.speaker_name,
            model_id: voice.model_id ?? "qwen3-tts-1.7b",
            language: "auto",
            seed,
            quality_review: {
              enable_asr: true,
              enable_speaker: Boolean(referenceAudioKey),
              enable_style: Boolean(referenceAudioKey),
              enable_speed: Boolean(referenceAudioKey && referenceText),
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

          const quality = output.quality ?? {};
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
              firstFailureMessage = `sample ${i + 1} seed ${seed} missing asr_score`;
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
        const scoreParts: Array<{ value: number; weight: number }> = [
          { value: meanOverall, weight: 0.38 },
          { value: meanAsr, weight: 0.22 },
          { value: meanHealth, weight: 0.14 },
          { value: meanDuration, weight: 0.08 },
          { value: passRate, weight: 0.08 },
        ];
        if (Number.isFinite(meanSpeaker)) {
          scoreParts.push({ value: meanSpeaker, weight: 0.20 });
        }
        if (Number.isFinite(meanTone)) {
          scoreParts.push({ value: meanTone, weight: 0.06 });
        }
        if (Number.isFinite(meanSpeed)) {
          scoreParts.push({ value: meanSpeed, weight: 0.06 });
        }
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
  }

  if (status === "completed" && !job.summary?.validation_checked) {
    const voice = await getVoice(c.env.DB, job.voice_id);
    if (!voice) {
      const message = "Voice record missing for validation";
      await updateTrainingJob(c.env.DB, job.job_id, {
        status: "failed",
        error_message: message,
        summary: {
          ...(job.summary ?? {}),
          validation_failed: true,
          validation_checked: true,
          validation_passed: false,
          validation_message: message,
        },
        completed_at: job.completed_at ?? Date.now(),
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
          ...(job.summary ?? {}),
          validation_failed: true,
          validation_checked: true,
          validation_passed: false,
          validation_message: message,
        },
        completed_at: job.completed_at ?? Date.now(),
        progress,
      });
      return { status: "failed", progress };
    }

    const candidateCheckpoints = parsedStatus.checkpoints
      .filter(
        (cp): cp is { epoch: number; r2_prefix: string } =>
          typeof cp.epoch === "number" && typeof cp.r2_prefix === "string"
      )
      .sort((a, b) => b.epoch - a.epoch)
      .slice(0, MAX_CHECKPOINTS_TO_EVAL);

    if (candidateCheckpoints.length === 0) {
      const message = "Training completed but checkpoint metadata is invalid";
      await updateVoice(c.env.DB, job.voice_id, {
        status: "created",
      });
      await updateTrainingJob(c.env.DB, job.job_id, {
        status: "failed",
        error_message: message,
        summary: {
          ...(job.summary ?? {}),
          validation_failed: true,
          validation_checked: true,
          validation_passed: false,
          validation_message: message,
        },
        completed_at: job.completed_at ?? Date.now(),
        progress,
      });
      return { status: "failed", progress };
    }

    const checkpointEvaluations: Array<{
      epoch: number;
      prefix: string;
      ok: boolean;
      score: number;
      message: string;
      preset: string;
      passed_samples: number;
      total_samples: number;
    }> = [];
    let champion: {
      checkpoint: { epoch: number; r2_prefix: string };
      result: CheckpointValidationResult;
    } | null = null;

    for (const checkpoint of candidateCheckpoints) {
      const validation = await validateTrainedCheckpoint(c, voice, job, checkpoint.r2_prefix);
      checkpointEvaluations.push({
        epoch: checkpoint.epoch,
        prefix: checkpoint.r2_prefix,
        ok: validation.ok,
        score: validation.aggregateScore,
        message: validation.message,
        preset: validation.presetName,
        passed_samples: validation.passedSamples,
        total_samples: validation.totalSamples,
      });

      if (validation.ok && (!champion || validation.aggregateScore > champion.result.aggregateScore)) {
        champion = { checkpoint, result: validation };
      }
    }

    if (champion) {
      const voiceUpdate: {
        status: string;
        checkpoint_r2_prefix: string;
        run_name: string | null;
        epoch: number;
        settings?: {
          stability: number;
          similarity_boost: number;
          style: number;
          speed: number;
        };
      } = {
        status: "ready",
        checkpoint_r2_prefix: champion.checkpoint.r2_prefix,
        run_name: parseRunNameFromCheckpointPrefix(champion.checkpoint.r2_prefix),
        epoch: champion.checkpoint.epoch,
      };
      if (champion.result.presetSettings) {
        voiceUpdate.settings = champion.result.presetSettings;
      }
      await updateVoice(c.env.DB, job.voice_id, voiceUpdate);

      await updateTrainingJob(c.env.DB, job.job_id, {
        status: "completed",
        completed_at: job.completed_at ?? Date.now(),
        progress,
        summary: {
          ...(job.summary ?? {}),
          validation_checked: true,
          validation_passed: true,
          validation_message: champion.result.message,
          selected_checkpoint_prefix: champion.checkpoint.r2_prefix,
          selected_checkpoint_epoch: champion.checkpoint.epoch,
          selected_preset: champion.result.presetName,
          selected_score: champion.result.aggregateScore,
          evaluated_checkpoints: checkpointEvaluations,
        },
      });
    } else {
      const bestFailure = checkpointEvaluations
        .sort((a, b) => b.passed_samples - a.passed_samples || b.score - a.score)
        .at(0);
      const failureMessage =
        bestFailure?.message ??
        "Validation failed for all candidate checkpoints";

      await updateVoice(c.env.DB, job.voice_id, {
        status: "created",
        checkpoint_r2_prefix: null,
        run_name: null,
        epoch: null,
      });
      await updateTrainingJob(c.env.DB, job.job_id, {
        status: "failed",
        error_message: failureMessage,
        summary: {
          ...(job.summary ?? {}),
          validation_failed: true,
          validation_checked: true,
          validation_passed: false,
          validation_message: failureMessage,
          evaluated_checkpoints: checkpointEvaluations,
        },
        completed_at: job.completed_at ?? Date.now(),
        progress,
      });
      return { status: "failed", progress };
    }
  }

  return { status, progress };
};

const RECONCILE_TIMEOUT_MS = 25000;

const reconcileJobStatusWithTimeout = async (
  c: Context<AppContext>,
  job: TrainingJob
): Promise<boolean> => {
  let timedOut = false;
  await Promise.race([
    reconcileJobStatus(c, job),
    new Promise<void>((resolve) =>
      setTimeout(() => {
        timedOut = true;
        resolve();
      }, RECONCILE_TIMEOUT_MS)
    ),
  ]);
  return !timedOut;
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
  const datasetPrefix = body.dataset_name ? `datasets/${body.voice_id}/${body.dataset_name}` : `datasets/${body.voice_id}`;
  const config = body.config ?? {};
  const cfg = config as Record<string, unknown>;

  const numEpochs = Number(config.num_epochs ?? cfg.epochs ?? 8);
  const batchSize = Number(config.batch_size ?? 4);
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
    config,
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
    model_size: typeof config.model_size === "string" ? config.model_size : (voice.model_size || "1.7B"),
    batch_size: Number(config.batch_size ?? 4),
    learning_rate: Number(config.learning_rate ?? 1e-5),
    num_epochs: Number(config.num_epochs ?? cfg.epochs ?? 8),
    run_name: runName,
    gradient_accumulation_steps: Number(cfg.gradient_accumulation_steps ?? 4),
    speaker_id: Number(cfg.speaker_id ?? 3000),
    mixed_precision: String(cfg.mixed_precision ?? "bf16"),
    torch_dtype: String(cfg.torch_dtype ?? "bfloat16"),
      attn_implementation: String(cfg.attn_implementation ?? "sdpa"),
    weight_decay: Number(cfg.weight_decay ?? 0.01),
    max_grad_norm: Number(cfg.max_grad_norm ?? 1.0),
    subtalker_loss_weight: Number(cfg.subtalker_loss_weight ?? 0.3),
    log_every_n_steps: Number(cfg.log_every_n_steps ?? 10),
    save_every_n_epochs: Number(cfg.save_every_n_epochs ?? 1),
    max_steps: Number(cfg.max_steps ?? 0),
    seed: Number(cfg.seed ?? 42),
    job_token: jobToken,
    worker_api_url: workerUrl,
    whisper_language: typeof cfg.whisper_language === "string" ? cfg.whisper_language : undefined,
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
      if (!ACTIVE_JOB_STATUSES.has(job.status)) {
        if (!needsCompletedValidation(job)) {
          return job;
        }
      }
      await reconcileJobStatusWithTimeout(c, job);
      return (await getTrainingJob(c.env.DB, job.job_id)) ?? job;
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
    const reconciled = await reconcileJobStatusWithTimeout(c, job);
    if (!reconciled) {
      const execCtx = (c as unknown as { executionCtx?: ExecutionContext }).executionCtx;
      execCtx?.waitUntil(
        reconcileJobStatus(c, job)
          .then(() => undefined)
          .catch(() => undefined)
      );
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

  const execCtx = (c as unknown as { executionCtx?: ExecutionContext }).executionCtx;
  execCtx?.waitUntil(
    reconcileJobStatus(c, job)
      .then(() => undefined)
      .catch(() => undefined)
  );

  return c.json({
    status: "accepted",
    validation_started: true,
    job_id: jobId,
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
