import { Hono } from "hono";
import type { Context } from "hono";
import {
  createTrainingLogChunk,
  getDatasetPreprocessCache,
  getDatasetSnapshot,
  upsertDatasetPreprocessCache,
  upsertDatasetSnapshot,
  getTrainingJobByToken,
  updateTrainingRound,
  updateTrainingJob,
  updateVoice,
} from "../lib/d1";
import type { AppContext, TrainingProgress } from "../types";
import { stripSlashes, extractDatasetNameFromPrefix } from "../lib/training-domain";
import { dispatchQueuedJobs } from "./training";

const app = new Hono<AppContext>();

type CheckpointRecord = { epoch?: number; r2_prefix?: string };

const resolveCanonicalDatasetRefAudioKey = (
  datasetPrefixRaw: string,
  ...candidates: Array<string | null | undefined>
): string | null => {
  const datasetPrefix = stripSlashes(datasetPrefixRaw);
  let hasCandidate = false;
  for (const candidate of candidates) {
    const trimmed = typeof candidate === "string" ? candidate.trim() : "";
    if (!trimmed) {
      continue;
    }
    hasCandidate = true;
    if (datasetPrefix && trimmed.startsWith(`${datasetPrefix}/`)) {
      return trimmed;
    }
  }
  if (datasetPrefix && hasCandidate) {
    return `${datasetPrefix}/ref_audio.wav`;
  }
  for (const candidate of candidates) {
    const trimmed = typeof candidate === "string" ? candidate.trim() : "";
    if (trimmed) {
      return trimmed;
    }
  }
  return null;
};

const parseBearerToken = (authorizationHeader: string | undefined): string | null => {
  if (!authorizationHeader) {
    return null;
  }
  const [scheme, token] = authorizationHeader.split(" ");
  if (!scheme || !token || scheme.toLowerCase() !== "bearer") {
    return null;
  }
  return token.trim() || null;
};

const loadAuthedJob = async (c: Context<AppContext>) => {
  const jobId = c.req.param("job_id");
  const token = parseBearerToken(c.req.header("authorization"));
  if (!token) {
    return { error: c.json({ detail: { message: "Missing or invalid bearer token" } }, 401) };
  }

  const job = await getTrainingJobByToken(c.env.DB, jobId, token);
  if (!job) {
    return { error: c.json({ detail: { message: "Invalid job token" } }, 401) };
  }
  return { job };
};

app.post("/:job_id/heartbeat", async (c) => {
  const auth = await loadAuthedJob(c);
  if ("error" in auth) {
    return auth.error;
  }

  const body = (await c.req.json()) as {
    progress?: TrainingProgress;
    message?: string;
  };

  const summary = body.message
    ? {
        ...auth.job.summary,
        last_message: body.message,
      }
    : undefined;

  await updateTrainingJob(c.env.DB, auth.job.job_id, {
    progress: body.progress,
    summary,
    last_heartbeat_at: Date.now(),
    supervisor: {
      ...(auth.job.supervisor ?? {}),
      phase: auth.job.status,
      last_message: typeof body.message === "string" ? body.message : auth.job.supervisor?.last_message,
      last_heartbeat_at: Date.now(),
    },
  });

  return c.json({ status: "ok" });
});

app.post("/:job_id/report", async (c) => {
  const auth = await loadAuthedJob(c);
  if ("error" in auth) {
    return auth.error;
  }

  const body = (await c.req.json()) as {
    status: string;
    progress?: TrainingProgress;
    metrics?: Record<string, unknown>;
    message?: string;
    error?: string;
    checkpoints?: CheckpointRecord[];
  };

  if (!body.status) {
    return c.json({ detail: { message: "status is required" } }, 400);
  }

  const now = Date.now();
  const summary: Record<string, unknown> = {
    ...auth.job.summary,
  };

  if (body.message) {
    summary.last_message = body.message;
  }
  if (body.status === "completed") {
    summary.completed_at = now;
    if (typeof auth.job.started_at === "number") {
      summary.duration_ms = Math.max(0, now - auth.job.started_at);
    }
    if (typeof body.progress?.epoch === "number") {
      summary.final_epoch = body.progress.epoch;
    }
    if (typeof body.progress?.loss === "number") {
      summary.final_loss = body.progress.loss;
    }
    if (typeof body.progress?.total_epochs === "number") {
      summary.total_epochs = body.progress.total_epochs;
    }
  }

  await updateTrainingJob(c.env.DB, auth.job.job_id, {
    status: body.status,
    progress: body.progress,
    metrics: body.metrics,
    summary,
    supervisor: {
      ...(auth.job.supervisor ?? {}),
      phase: body.status,
      last_message: typeof body.message === "string" ? body.message : auth.job.supervisor?.last_message,
      last_reported_status: body.status,
      last_transition_at: now,
    },
    error_message: body.error ?? (body.status === "failed" ? "Training failed" : auth.job.error_message),
    completed_at: body.status === "completed" ? now : auth.job.completed_at,
    last_heartbeat_at: now,
  });

  if (auth.job.round_id) {
    await updateTrainingRound(c.env.DB, auth.job.round_id, {
      status:
        body.status === "completed"
          ? "validating"
          : body.status === "failed"
            ? "failed"
            : "running",
      completed_at: body.status === "failed" ? now : null,
      summary: {
        last_job_id: auth.job.job_id,
        last_job_status: body.status,
        last_message: typeof body.message === "string" ? body.message : null,
      },
    });
  }

  if (body.status === "completed" && Array.isArray(body.checkpoints) && body.checkpoints.length > 0) {
    const lastCheckpoint = body.checkpoints[body.checkpoints.length - 1];
    if (typeof lastCheckpoint.epoch === "number" && typeof lastCheckpoint.r2_prefix === "string") {
      await updateTrainingJob(c.env.DB, auth.job.job_id, {
        summary: {
          ...summary,
          callback_last_checkpoint_epoch: lastCheckpoint.epoch,
          callback_last_checkpoint_prefix: lastCheckpoint.r2_prefix,
          callback_reported_completed_at: now,
        },
      });
    }
  }

  if (body.status === "completed" || body.status === "failed") {
    const ctx = (c as unknown as { executionCtx?: ExecutionContext }).executionCtx;
    ctx?.waitUntil(dispatchQueuedJobs(c.env, auth.job.voice_id));
  }

  return c.json({ status: "ok" });
});

app.post("/:job_id/log", async (c) => {
  const auth = await loadAuthedJob(c);
  if ("error" in auth) {
    return auth.error;
  }

  const body = (await c.req.json()) as {
    seq?: number;
    r2_key?: string;
    bytes?: number;
    lines?: number;
  };

  if (typeof body.seq !== "number" || !body.r2_key) {
    return c.json({ detail: { message: "seq and r2_key are required" } }, 400);
  }

  await createTrainingLogChunk(c.env.DB, {
    job_id: auth.job.job_id,
    seq: body.seq,
    r2_key: body.r2_key,
    created_at: Date.now(),
    bytes: body.bytes,
    lines: body.lines,
  });

  await updateTrainingJob(c.env.DB, auth.job.job_id, {
    log_r2_prefix: `jobs/${auth.job.job_id}/logs`,
    last_heartbeat_at: Date.now(),
  });

  return c.json({ status: "ok" });
});

app.post("/:job_id/preprocess-cache", async (c) => {
  const auth = await loadAuthedJob(c);
  if ("error" in auth) {
    return auth.error;
  }

  const body = (await c.req.json()) as {
    dataset_signature?: string;
    cache_r2_prefix?: string;
    train_raw_r2_key?: string;
    train_clipped_r2_key?: string | null;
    control_holdout_r2_key?: string | null;
    ref_audio_r2_key?: string | null;
    reference_profile_r2_key?: string | null;
    source_file_count?: number;
    segments_created?: number;
    segments_accepted?: number;
    accepted_duration_min?: number;
    train_segments?: number;
    holdout_segments?: number;
    holdout_ratio?: number;
  };

  const datasetSignature =
    typeof body.dataset_signature === "string" ? body.dataset_signature.trim() : "";
  const cacheR2Prefix =
    typeof body.cache_r2_prefix === "string" ? body.cache_r2_prefix.trim() : "";
  const trainRawR2Key =
    typeof body.train_raw_r2_key === "string" ? body.train_raw_r2_key.trim() : "";
  const reportedRefAudioR2Key =
    typeof body.ref_audio_r2_key === "string" && body.ref_audio_r2_key.trim()
      ? body.ref_audio_r2_key.trim()
      : null;
  if (!datasetSignature || !cacheR2Prefix || !trainRawR2Key) {
    return c.json(
      {
        detail: {
          message:
            "dataset_signature, cache_r2_prefix, and train_raw_r2_key are required",
        },
      },
      400
    );
  }

  const now = Date.now();
  await upsertDatasetPreprocessCache(c.env.DB, {
    cache_id: crypto.randomUUID(),
    voice_id: auth.job.voice_id,
    dataset_r2_prefix: auth.job.dataset_r2_prefix,
    dataset_signature: datasetSignature,
    cache_r2_prefix: cacheR2Prefix,
    train_raw_r2_key: trainRawR2Key,
    ref_audio_r2_key:
      typeof body.ref_audio_r2_key === "string" && body.ref_audio_r2_key.trim()
        ? body.ref_audio_r2_key.trim()
        : null,
    reference_profile_r2_key:
      typeof body.reference_profile_r2_key === "string" &&
      body.reference_profile_r2_key.trim()
        ? body.reference_profile_r2_key.trim()
        : null,
    source_file_count:
      typeof body.source_file_count === "number" &&
      Number.isFinite(body.source_file_count)
        ? Math.trunc(body.source_file_count)
        : null,
    segments_created:
      typeof body.segments_created === "number" &&
      Number.isFinite(body.segments_created)
        ? Math.trunc(body.segments_created)
        : null,
    segments_accepted:
      typeof body.segments_accepted === "number" &&
      Number.isFinite(body.segments_accepted)
        ? Math.trunc(body.segments_accepted)
        : null,
    accepted_duration_min:
      typeof body.accepted_duration_min === "number" &&
      Number.isFinite(body.accepted_duration_min)
        ? body.accepted_duration_min
        : null,
    created_at: now,
    updated_at: now,
  });

  const persistedCache = await getDatasetPreprocessCache(
    c.env.DB,
    auth.job.voice_id,
    auth.job.dataset_r2_prefix,
    datasetSignature
  );
  const existingSnapshot = await getDatasetSnapshot(
    c.env.DB,
    auth.job.voice_id,
    auth.job.dataset_r2_prefix,
    datasetSignature
  );
  const canonicalRefAudioR2Key = resolveCanonicalDatasetRefAudioKey(
    auth.job.dataset_r2_prefix,
    existingSnapshot?.ref_audio_r2_key,
    reportedRefAudioR2Key
  );
  await upsertDatasetSnapshot(c.env.DB, {
    snapshot_id: auth.job.dataset_snapshot_id ?? existingSnapshot?.snapshot_id ?? crypto.randomUUID(),
    voice_id: auth.job.voice_id,
    dataset_name: extractDatasetNameFromPrefix(auth.job.dataset_r2_prefix),
    dataset_r2_prefix: auth.job.dataset_r2_prefix,
    dataset_signature: datasetSignature,
    status: "frozen",
    source_cache_id: persistedCache?.cache_id ?? existingSnapshot?.source_cache_id ?? null,
    cache_r2_prefix: cacheR2Prefix,
    train_raw_r2_key: trainRawR2Key,
    ref_audio_r2_key: canonicalRefAudioR2Key,
    reference_profile_r2_key:
      typeof body.reference_profile_r2_key === "string" && body.reference_profile_r2_key.trim()
        ? body.reference_profile_r2_key.trim()
        : null,
    reference_text: existingSnapshot?.reference_text ?? null,
    source_file_count:
      typeof body.source_file_count === "number" && Number.isFinite(body.source_file_count)
        ? Math.trunc(body.source_file_count)
        : null,
    segments_created:
      typeof body.segments_created === "number" && Number.isFinite(body.segments_created)
        ? Math.trunc(body.segments_created)
        : null,
    segments_accepted:
      typeof body.segments_accepted === "number" && Number.isFinite(body.segments_accepted)
        ? Math.trunc(body.segments_accepted)
        : null,
    accepted_duration_min:
      typeof body.accepted_duration_min === "number" && Number.isFinite(body.accepted_duration_min)
        ? body.accepted_duration_min
        : null,
    created_from_job_id: auth.job.job_id,
    created_at: now,
    updated_at: now,
  });

  if (canonicalRefAudioR2Key) {
    await updateVoice(c.env.DB, auth.job.voice_id, {
      ref_audio_r2_key: canonicalRefAudioR2Key,
    });
  }

  await updateTrainingJob(c.env.DB, auth.job.job_id, {
    summary: {
      ...auth.job.summary,
      preprocess_cache_lookup: "stored",
      preprocess_cache_dataset_signature: datasetSignature,
      preprocess_cache_r2_prefix: cacheR2Prefix,
      preprocess_cache_train_raw_r2_key: trainRawR2Key,
      preprocess_cache_train_clipped_r2_key:
        typeof body.train_clipped_r2_key === "string"
          ? body.train_clipped_r2_key.trim()
          : null,
      preprocess_cache_control_holdout_r2_key:
        typeof body.control_holdout_r2_key === "string"
          ? body.control_holdout_r2_key.trim()
          : null,
      preprocess_cache_ref_audio_r2_key:
        typeof body.ref_audio_r2_key === "string" ? body.ref_audio_r2_key.trim() : null,
      preprocess_cache_reference_profile_r2_key:
        typeof body.reference_profile_r2_key === "string"
          ? body.reference_profile_r2_key.trim()
          : null,
      preprocess_cache_source_file_count:
        typeof body.source_file_count === "number" ? Math.trunc(body.source_file_count) : null,
      preprocess_cache_segments_created:
        typeof body.segments_created === "number" ? Math.trunc(body.segments_created) : null,
      preprocess_cache_segments_accepted:
        typeof body.segments_accepted === "number" ? Math.trunc(body.segments_accepted) : null,
      preprocess_cache_accepted_duration_min:
        typeof body.accepted_duration_min === "number" ? body.accepted_duration_min : null,
      preprocess_cache_train_segments:
        typeof body.train_segments === "number" ? Math.trunc(body.train_segments) : null,
      preprocess_cache_holdout_segments:
        typeof body.holdout_segments === "number" ? Math.trunc(body.holdout_segments) : null,
      preprocess_cache_holdout_ratio:
        typeof body.holdout_ratio === "number" ? body.holdout_ratio : null,
      preprocess_cache_saved_at: now,
    },
    supervisor: {
      ...(auth.job.supervisor ?? {}),
      phase: "preprocessing",
      preprocess_cache_saved_at: now,
    },
    last_heartbeat_at: now,
  });

  if (auth.job.round_id) {
    await updateTrainingRound(c.env.DB, auth.job.round_id, {
      dataset_snapshot_id: auth.job.dataset_snapshot_id ?? null,
      summary: {
        preprocess_cache_saved_at: now,
        last_job_id: auth.job.job_id,
      },
    });
  }

  return c.json({ status: "ok" });
});

export default app;
