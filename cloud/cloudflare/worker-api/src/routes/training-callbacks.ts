import { Hono } from "hono";
import type { Context } from "hono";
import {
  createTrainingLogChunk,
  getTrainingJobByToken,
  updateTrainingJob,
  updateVoice,
} from "../lib/d1";
import type { AppContext, TrainingProgress } from "../types";

const app = new Hono<AppContext>();

type CheckpointRecord = { epoch?: number; r2_prefix?: string };

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

const parseRunNameFromCheckpointPrefix = (prefix: string): string | null => {
  const parts = prefix.split("/");
  if (parts.length < 4 || parts[0] !== "checkpoints") {
    return null;
  }
  return parts[2] || null;
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
    error_message: body.error ?? (body.status === "failed" ? "Training failed" : auth.job.error_message),
    completed_at: body.status === "completed" ? now : auth.job.completed_at,
    last_heartbeat_at: now,
  });

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

export default app;
