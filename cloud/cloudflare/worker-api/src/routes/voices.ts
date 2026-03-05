import { Hono } from "hono";
import { authMiddleware } from "../middleware/auth";
import { createVoice, deleteVoice, getVoice, listTrainingJobs, listVoices } from "../lib/d1";
import type { AppContext, Voice } from "../types";

const app = new Hono<AppContext>();
app.use("*", authMiddleware);

const parseLabels = (raw: string | null): Record<string, string> => {
  if (!raw) {
    return {};
  }
  try {
    const parsed = JSON.parse(raw) as Record<string, unknown>;
    return Object.entries(parsed).reduce<Record<string, string>>((acc, [key, value]) => {
      if (typeof value === "string") {
        acc[key] = value;
      }
      return acc;
    }, {});
  } catch {
    return {};
  }
};

const sanitizeFileName = (name: string) => name.replace(/[^a-zA-Z0-9._-]/g, "_");
const MAX_UPLOAD_FILE_SIZE_BYTES = 50 * 1024 * 1024;
const ALLOWED_AUDIO_CONTENT_TYPES = new Set([
  "audio/wav",
  "audio/x-wav",
  "audio/wave",
  "application/octet-stream",
]);

const MODEL_BY_SIZE: Record<string, { model_size: "1.7B" | "0.6B"; model_id: "qwen3-tts-1.7b" | "qwen3-tts-0.6b" }> = {
  "1.7b": { model_size: "1.7B", model_id: "qwen3-tts-1.7b" },
  "0.6b": { model_size: "0.6B", model_id: "qwen3-tts-0.6b" },
};

const toSpeakerName = (name: string): string => {
  const normalized = name.trim().toLowerCase().replace(/[^a-z0-9]+/g, "_");
  return normalized || `speaker_${crypto.randomUUID().slice(0, 8)}`;
};

type UploadFile = {
  name: string;
  type: string;
  size: number;
  arrayBuffer(): Promise<ArrayBuffer>;
};

const isUploadFile = (value: unknown): value is UploadFile => {
  if (!value || typeof value !== "object") {
    return false;
  }
  const maybe = value as Record<string, unknown>;
  return (
    typeof maybe.name === "string" &&
    typeof maybe.type === "string" &&
    typeof maybe.size === "number" &&
    typeof maybe.arrayBuffer === "function"
  );
};

app.get("/", async (c) => {
  const search = c.req.query("search");
  const category = c.req.query("category");
  const status = c.req.query("status");

  const voices = await listVoices(c.env.DB, { search, category, status });

  return c.json({
    voices,
    has_more: false,
    total_count: voices.length,
  });
});

app.get("/:voice_id", async (c) => {
  const voiceId = c.req.param("voice_id");
  const voice = await getVoice(c.env.DB, voiceId);

  if (!voice) {
    return c.json({ detail: { message: "Voice not found" } }, 404);
  }

  return c.json(voice);
});

app.post("/add", async (c) => {
  const formData = await c.req.formData();
  const name = String(formData.get("name") ?? "").trim();
  const description = String(formData.get("description") ?? "").trim();
  const labels = parseLabels(formData.get("labels")?.toString() ?? null);
  const requestedModelSize = String(formData.get("model_size") ?? "1.7B").trim().toLowerCase();
  const modelConfig = MODEL_BY_SIZE[requestedModelSize];

  if (!name) {
    return c.json({ detail: { message: "name is required" } }, 400);
  }
  if (!modelConfig) {
    return c.json({ detail: { message: "model_size must be either 1.7B or 0.6B" } }, 400);
  }

  const fileEntries: UploadFile[] = [];
  for (const [key, value] of (formData as unknown as { entries(): Iterable<[string, unknown]> }).entries()) {
    if ((key === "files" || key.startsWith("files")) && isUploadFile(value) && value.size > 0) {
      if (value.size > MAX_UPLOAD_FILE_SIZE_BYTES) {
        return c.json({ detail: { message: `File ${value.name} exceeds 50MB limit` } }, 400);
      }
      if (!ALLOWED_AUDIO_CONTENT_TYPES.has(value.type)) {
        return c.json({ detail: { message: `Only WAV format is supported for ${value.name}. Please convert your audio to 24kHz mono WAV before uploading.` } }, 400);
      }
      fileEntries.push(value);
    }
  }

  const voiceId = crypto.randomUUID();
  const now = Date.now();

  const voice: Voice = {
    voice_id: voiceId,
    name,
    description,
    speaker_name: toSpeakerName(name),
    model_size: modelConfig.model_size,
    model_id: modelConfig.model_id,
    category: "cloned",
    status: "created",
    checkpoint_r2_prefix: null,
    run_name: null,
    epoch: null,
    sample_audio_r2_key: null,
    ref_audio_r2_key: null,
    labels,
    settings: {},
    preview_url: null,
    created_at: now,
    updated_at: now,
  };

  await createVoice(c.env.DB, voice);

  await Promise.all(
    fileEntries.map(async (file, index) => {
      const key = `datasets/${voiceId}/${Date.now()}_${index}_${sanitizeFileName(file.name)}`;
      await c.env.R2.put(key, await file.arrayBuffer(), {
        httpMetadata: {
          contentType: file.type || "application/octet-stream",
        },
      });
    })
  );

  return c.json({ voice_id: voiceId });
});

app.delete("/:voice_id", async (c) => {
  const voiceId = c.req.param("voice_id");

  const activeJobs = await listTrainingJobs(c.env.DB, { voice_id: voiceId, limit: 5 });
  const hasActiveJob = activeJobs.some((j) =>
    ["pending", "provisioning", "downloading", "preprocessing", "preparing", "training", "uploading"].includes(j.status)
  );
  if (hasActiveJob) {
    return c.json(
      { detail: { message: "Cannot delete voice while training is active. Cancel the training job first." } },
      409
    );
  }

  await deleteVoice(c.env.DB, voiceId);

  const cleanupPrefixes = [`datasets/${voiceId}/`, `audio/${voiceId}/`, `checkpoints/${voiceId}/`];
  for (const prefix of cleanupPrefixes) {
    let cursor: string | undefined;
    do {
      const listing = await c.env.R2.list({ prefix, cursor });
      if (listing.objects.length > 0) {
        await c.env.R2.delete(listing.objects.map((obj) => obj.key));
      }
      cursor = listing.truncated ? listing.cursor : undefined;
    } while (cursor);
  }

  return c.json({ status: "ok" });
});

export default app;
