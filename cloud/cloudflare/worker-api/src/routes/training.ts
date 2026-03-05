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
import { createPod, invokeServerless, terminatePod } from "../lib/runpod";
import { authMiddleware } from "../middleware/auth";
import type { AppContext, TrainingConfig, TrainingJob, TrainingProgress } from "../types";

const app = new Hono<AppContext>();
app.use("*", authMiddleware);

type TrainingStatusBlob = {
  status?: string;
  progress?: TrainingProgress;
  checkpoints?: Array<{ epoch?: number; r2_prefix?: string }>;
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

const serializeTrainingJob = (job: TrainingJob): Omit<TrainingJob, "job_token"> => {
  const { job_token: _jobToken, ...safeJob } = job;
  return safeJob;
};

const validateTrainedCheckpoint = async (
  c: Context<AppContext>,
  job: TrainingJob,
  checkpointPrefix: string
): Promise<{ ok: boolean; message: string }> => {
  const voice = await getVoice(c.env.DB, job.voice_id);
  if (!voice) {
    return { ok: false, message: "Voice record missing for validation" };
  }

  // Multi-language validation texts — select based on job config or default to mixed set
  const jobConfig = job.config as Record<string, unknown>;
  const lang = typeof jobConfig.whisper_language === "string" ? jobConfig.whisper_language.toLowerCase() : "";

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

  // Use language-specific texts, or fall back to mixed set (short/medium/long)
  const validationTexts = validationTextsByLang[lang] ?? [
    "Hello.",
    "Hello. The meeting starts at two o'clock this afternoon.",
    "Hello. The meeting starts at two o'clock this afternoon, and I will share the presentation materials via email.",
  ];

  try {
    for (let i = 0; i < validationTexts.length; i += 1) {
      for (const seed of [123456 + i, 223456 + i]) {
        const payload: Record<string, unknown> = {
          text: validationTexts[i],
          voice_id: voice.voice_id,
          speaker_name: voice.speaker_name,
          model_id: voice.model_id ?? "qwen3-tts-1.7b",
          language: "auto",
          seed,
          checkpoint_info: {
            r2_prefix: checkpointPrefix,
            type: "full",
          },
        };

        const response = await invokeServerless(c.env, c.env.RUNPOD_ENDPOINT_ID, payload);
        const output = (response.output ?? {}) as { quality?: Record<string, unknown>; audio?: string };
        if (!output.audio) {
          return { ok: false, message: `Validation sample ${i + 1} seed ${seed} returned no audio` };
        }

        const quality = output.quality ?? {};
        const overall = Number(quality.overall_score ?? NaN);
        const duration = Number(quality.duration_score ?? NaN);
        const health = Number(quality.health_score ?? NaN);

        if (Number.isFinite(overall) && overall < 0.85) {
          return { ok: false, message: `Validation sample ${i + 1} seed ${seed} failed: overall_score=${overall.toFixed(2)}` };
        }
        if (Number.isFinite(duration) && duration < 0.45) {
          return { ok: false, message: `Validation sample ${i + 1} seed ${seed} failed: duration_score=${duration.toFixed(2)}` };
        }
        if (Number.isFinite(health) && health < 0.72) {
          return { ok: false, message: `Validation sample ${i + 1} seed ${seed} failed: health_score=${health.toFixed(2)}` };
        }
      }
    }

    return { ok: true, message: "Validation passed on all samples" };
  } catch (error) {
    return {
      ok: false,
      message: `Validation invocation failed: ${error instanceof Error ? error.message : "unknown"}`,
    };
  }
};

const reconcileJobStatus = async (
  c: Context<AppContext>,
  job: TrainingJob
): Promise<{ status: string; progress: TrainingProgress }> => {
  let status = job.status;
  let progress: TrainingProgress = job.progress;
  const statusBlob = await c.env.R2.get(`jobs/${job.job_id}/status.json`);

  if (!statusBlob) {
    return { status, progress };
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

  if (status === "completed" && !job.summary?.validation_checked && Array.isArray(parsedStatus.checkpoints) && parsedStatus.checkpoints.length > 0) {
    const lastCheckpoint = parsedStatus.checkpoints[parsedStatus.checkpoints.length - 1];
    if (typeof lastCheckpoint.epoch === "number" && typeof lastCheckpoint.r2_prefix === "string") {
      const validation = await validateTrainedCheckpoint(c, job, lastCheckpoint.r2_prefix);
      if (validation.ok) {
        await updateVoice(c.env.DB, job.voice_id, {
          status: "ready",
          checkpoint_r2_prefix: lastCheckpoint.r2_prefix,
          run_name: parseRunNameFromCheckpointPrefix(lastCheckpoint.r2_prefix),
          epoch: lastCheckpoint.epoch,
        });
        await updateTrainingJob(c.env.DB, job.job_id, {
          status: "completed",
          completed_at: job.completed_at ?? Date.now(),
          progress,
          summary: {
            ...(job.summary ?? {}),
            validation_checked: true,
            validation_passed: true,
            validation_message: validation.message,
          },
        });
      } else {
        await updateVoice(c.env.DB, job.voice_id, {
          status: "created",
        });
        await updateTrainingJob(c.env.DB, job.job_id, {
          status: "failed",
          error_message: validation.message,
          summary: {
            ...(job.summary ?? {}),
            validation_failed: true,
            validation_checked: true,
            validation_passed: false,
            validation_message: validation.message,
          },
          completed_at: job.completed_at ?? Date.now(),
          progress,
        });
        return { status: "failed", progress };
      }
    }

    await updateTrainingJob(c.env.DB, job.job_id, {
      status: "completed",
      completed_at: job.completed_at ?? Date.now(),
      progress,
    });
  }

  return { status, progress };
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
  const workerUrl = new URL(c.req.url).origin;
  const runName = `run_${jobId.slice(0, 8)}`;
  const datasetPrefix = body.dataset_name ? `datasets/${body.voice_id}/${body.dataset_name}` : `datasets/${body.voice_id}`;
  const config = body.config ?? {};
  const cfg = config as Record<string, unknown>;

  const numEpochs = Number(config.num_epochs ?? cfg.epochs ?? 5);
  const batchSize = Number(config.batch_size ?? 2);
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
    batch_size: Number(config.batch_size ?? 2),
    learning_rate: Number(config.learning_rate ?? 2e-5),
    num_epochs: Number(config.num_epochs ?? cfg.epochs ?? 5),
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
  const modelSize = typeof config.model_size === "string" ? config.model_size : "1.7B";
  const defaultGpu = modelSize.includes("0.6") ? "NVIDIA GeForce RTX 4090" : "NVIDIA L40S";
  const gpuTypeId = typeof config.gpu_type_id === "string" && config.gpu_type_id ? config.gpu_type_id : defaultGpu;

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

  let pod: { podId: string; desiredStatus: string };
  try {
    pod = await createPod(c.env, c.env.RUNPOD_TRAINING_TEMPLATE_ID, gpuTypeId, [
      { key: "JOB_ID", value: jobId },
      { key: "VOICE_ID", value: body.voice_id },
      { key: "WORKER_API_URL", value: workerUrl },
      { key: "JOB_TOKEN", value: jobToken },
      { key: "R2_ENDPOINT_URL", value: c.env.R2_ENDPOINT_URL },
      { key: "R2_ACCESS_KEY_ID", value: c.env.R2_ACCESS_KEY_ID },
      { key: "R2_SECRET_ACCESS_KEY", value: c.env.R2_SECRET_ACCESS_KEY },
      { key: "R2_BUCKET", value: "qwen-tts-studio" },
      { key: "RUNPOD_API_KEY", value: c.env.RUNPOD_API_KEY },
      { key: "HF_HUB_ENABLE_HF_TRANSFER", value: "0" },
    ]);
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
    runpod_pod_id: pod.podId,
    status: "provisioning",
    started_at: Date.now(),
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
      await reconcileJobStatus(c, job);
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
  const parsedCursor = Number(cursorRaw ?? "");
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
    await reconcileJobStatus(c, job);
    const updated = await getTrainingJob(c.env.DB, jobId);
    if (updated) {
      return c.json(serializeTrainingJob(updated));
    }
  }

  return c.json(serializeTrainingJob(job));
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
