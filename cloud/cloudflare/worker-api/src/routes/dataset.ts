import { Hono } from "hono";
import { authMiddleware } from "../middleware/auth";
import { getVoice, updateVoice } from "../lib/d1";
import { asrSimilarity, transcribeAudioWithReviewAsr } from "../lib/review-asr";
import { reviewTranscriptEntries } from "../lib/transcript-review";
import type { AppContext, CreateDatasetRequest, DatasetInfo } from "../types";

const app = new Hono<AppContext>();
app.use("*", authMiddleware);

const SAFE_DATASET_NAME_RE = /^[a-zA-Z0-9._-]{1,128}$/;

const extractFilename = (r2Key: string): string => {
  const parts = r2Key.split("/");
  return parts[parts.length - 1] ?? r2Key;
};

const arrayBufferToBase64 = (buffer: ArrayBuffer): string => {
  const bytes = new Uint8Array(buffer);
  let binary = "";
  for (let index = 0; index < bytes.length; index += 1) {
    binary += String.fromCharCode(bytes[index]);
  }
  return btoa(binary);
};

const getVoiceDatasetRootPrefix = (voiceId: string): string => `datasets/${voiceId}/`;

const isRawDatasetUploadKey = (voiceId: string, key: string): boolean => {
  const prefix = getVoiceDatasetRootPrefix(voiceId);
  if (!key.startsWith(prefix)) {
    return false;
  }
  const remainder = key.slice(prefix.length);
  return Boolean(remainder) && !remainder.includes("/");
};

const isAudioDatasetFile = (key: string): boolean => {
  const lower = key.toLowerCase();
  return (
    lower.endsWith(".wav") ||
    lower.endsWith(".wave") ||
    lower.endsWith(".mp3") ||
    lower.endsWith(".mp4") ||
    lower.endsWith(".m4a") ||
    lower.endsWith(".flac")
  );
};

// POST /v1/dataset/:voice_id — Create a finalized dataset (train_raw.jsonl + copies files)
app.post("/:voice_id", async (c) => {
  const voiceId = c.req.param("voice_id");

  const voice = await getVoice(c.env.DB, voiceId);
  if (!voice) {
    return c.json({ detail: { message: "Voice not found" } }, 404);
  }

  const body = (await c.req.json()) as CreateDatasetRequest;

  if (!body.dataset_name || !SAFE_DATASET_NAME_RE.test(body.dataset_name)) {
    return c.json(
      { detail: { message: "dataset_name is required and must be alphanumeric/dash/dot/underscore (1-128 chars)" } },
      400,
    );
  }

  if (!Array.isArray(body.items) || body.items.length === 0) {
    return c.json({ detail: { message: "items array is required and must not be empty" } }, 400);
  }

  if (!body.ref_audio_r2_key) {
    return c.json({ detail: { message: "ref_audio_r2_key is required" } }, 400);
  }

  // Validate each item has required fields
  for (let i = 0; i < body.items.length; i++) {
    const item = body.items[i];
    if (!item.audio_r2_key || typeof item.audio_r2_key !== "string") {
      return c.json({ detail: { message: `items[${i}].audio_r2_key is required` } }, 400);
    }
    if (!item.text || typeof item.text !== "string" || item.text.trim().length === 0) {
      return c.json({ detail: { message: `items[${i}].text is required and must not be empty` } }, 400);
    }
  }

  // Validate ref_audio exists in R2
  const refHead = await c.env.R2.head(body.ref_audio_r2_key);
  if (!refHead) {
    return c.json(
      { detail: { message: `ref_audio not found in R2: ${body.ref_audio_r2_key}` } },
      404,
    );
  }

  // Validate all audio files exist in R2
  const missingAudio: string[] = [];
  for (const item of body.items) {
    const head = await c.env.R2.head(item.audio_r2_key);
    if (!head) {
      missingAudio.push(item.audio_r2_key);
    }
  }
  if (missingAudio.length > 0) {
    return c.json(
      { detail: { message: `Audio files not found in R2: ${missingAudio.join(", ")}` } },
      404,
    );
  }

  const datasetPrefix = `datasets/${voiceId}/${body.dataset_name}`;

  // Copy ref_audio to dataset directory with a fixed name
  const refAudioObj = await c.env.R2.get(body.ref_audio_r2_key);
  if (!refAudioObj) {
    return c.json({ detail: { message: "Failed to read ref_audio from R2" } }, 500);
  }
  await c.env.R2.put(`${datasetPrefix}/ref_audio.wav`, refAudioObj.body, {
    httpMetadata: { contentType: "audio/wav" },
  });

  // Copy all audio files to dataset directory and build JSONL lines
  const jsonlLines: string[] = [];

  for (let i = 0; i < body.items.length; i++) {
    const item = body.items[i];
    const rawFilename = extractFilename(item.audio_r2_key);
    const audioFilename = `${String(i).padStart(6, "0")}_${rawFilename}`;

    // Copy audio file into dataset directory
    const audioObj = await c.env.R2.get(item.audio_r2_key);
    if (!audioObj) {
      return c.json({ detail: { message: `Failed to read audio from R2: ${item.audio_r2_key}` } }, 500);
    }
    await c.env.R2.put(`${datasetPrefix}/${audioFilename}`, audioObj.body, {
      httpMetadata: { contentType: "audio/wav" },
    });

    // Build JSONL line with paths matching where training_handler.py downloads to
    const line = JSON.stringify({
      audio: `/tmp/dataset/${audioFilename}`,
      text: item.text.trim(),
      ref_audio: "/tmp/dataset/ref_audio.wav",
    });
    jsonlLines.push(line);
  }

  // Upload train_raw.jsonl
  const jsonlContent = jsonlLines.join("\n") + "\n";
  await c.env.R2.put(`${datasetPrefix}/train_raw.jsonl`, jsonlContent, {
    httpMetadata: { contentType: "application/jsonl" },
  });

  const refText = typeof body.ref_text === "string" ? body.ref_text.trim() : "";
  if (refText) {
    await c.env.R2.put(
      `${datasetPrefix}/reference_profile.json`,
      JSON.stringify(
        {
          reference_audio_key: `${datasetPrefix}/ref_audio.wav`,
          reference_text: refText,
        },
        null,
        2,
      ),
      {
        httpMetadata: { contentType: "application/json" },
      },
    );
  }

  // Update voice with ref_audio_r2_key
  await updateVoice(c.env.DB, voiceId, {
    ref_audio_r2_key: `${datasetPrefix}/ref_audio.wav`,
  });

  return c.json({
    dataset_name: body.dataset_name,
    dataset_r2_prefix: datasetPrefix,
    items_count: body.items.length,
    ref_audio_r2_key: `${datasetPrefix}/ref_audio.wav`,
  });
});

app.get("/:voice_id/raw-files", async (c) => {
  const voiceId = c.req.param("voice_id");

  const voice = await getVoice(c.env.DB, voiceId);
  if (!voice) {
    return c.json({ detail: { message: "Voice not found" } }, 404);
  }

  const limitRaw = Number(c.req.query("limit") ?? 500);
  const limit = Number.isFinite(limitRaw) ? Math.max(1, Math.min(1000, Math.trunc(limitRaw))) : 500;
  const prefix = getVoiceDatasetRootPrefix(voiceId);
  const listing = await c.env.R2.list({ prefix, limit: 1000 });
  const files = listing.objects
    .filter((obj) => isRawDatasetUploadKey(voiceId, obj.key) && isAudioDatasetFile(obj.key))
    .slice(0, limit)
    .map((obj) => ({
      key: obj.key,
      filename: extractFilename(obj.key),
      size: obj.size,
      uploaded: obj.uploaded.toISOString(),
      content_type: obj.httpMetadata?.contentType ?? null,
    }))
    .sort((a, b) => Date.parse(b.uploaded) - Date.parse(a.uploaded));

  return c.json({
    voice_id: voiceId,
    prefix,
    files,
  });
});

// GET /v1/dataset/:voice_id — List datasets for a voice
app.get("/:voice_id", async (c) => {
  const voiceId = c.req.param("voice_id");

  const voice = await getVoice(c.env.DB, voiceId);
  if (!voice) {
    return c.json({ detail: { message: "Voice not found" } }, 404);
  }

  const prefix = `datasets/${voiceId}/`;
  const listing = await c.env.R2.list({ prefix, delimiter: "/" });

  const datasets: DatasetInfo[] = [];

  // R2 list with delimiter returns "directories" as commonPrefixes
  if (listing.delimitedPrefixes) {
    for (const dirPrefix of listing.delimitedPrefixes) {
      // dirPrefix looks like "datasets/{voice_id}/{dataset_name}/"
      const parts = dirPrefix.replace(/\/$/, "").split("/");
      const datasetName = parts[parts.length - 1] ?? "";
      if (!datasetName) continue;

      // Count files in this dataset
      const filesListing = await c.env.R2.list({ prefix: dirPrefix });
      datasets.push({
        name: datasetName,
        r2_prefix: dirPrefix.replace(/\/$/, ""),
        file_count: filesListing.objects.length,
      });
    }
  }

  return c.json({ datasets });
});

app.post("/:voice_id/select", async (c) => {
  const voiceId = c.req.param("voice_id");
  const voice = await getVoice(c.env.DB, voiceId);
  if (!voice) {
    return c.json({ detail: { message: "Voice not found" } }, 404);
  }

  const body = (await c.req.json().catch(() => ({}))) as {
    dataset_name?: string;
  };
  const datasetName = typeof body.dataset_name === "string" ? body.dataset_name.trim() : "";
  if (!datasetName || !SAFE_DATASET_NAME_RE.test(datasetName)) {
    return c.json({ detail: { message: "dataset_name is required" } }, 400);
  }

  const refAudioKey = `datasets/${voiceId}/${datasetName}/ref_audio.wav`;
  const refHead = await c.env.R2.head(refAudioKey);
  if (!refHead) {
    return c.json({ detail: { message: `Dataset ref_audio not found: ${refAudioKey}` } }, 404);
  }

  await updateVoice(c.env.DB, voiceId, {
    ref_audio_r2_key: refAudioKey,
  });

  return c.json({
    status: "ok",
    dataset_name: datasetName,
    ref_audio_r2_key: refAudioKey,
  });
});

app.post("/:voice_id/retranscribe", async (c) => {
  const voiceId = c.req.param("voice_id");
  const voice = await getVoice(c.env.DB, voiceId);
  if (!voice) {
    return c.json({ detail: { message: "Voice not found" } }, 404);
  }

  const body = (await c.req.json().catch(() => ({}))) as {
    language_code?: string;
    entries?: Array<{ audio_r2_key?: string; text?: string }>;
  };
  const entries = Array.isArray(body.entries) ? body.entries : [];
  if (entries.length === 0) {
    return c.json({ detail: { message: "entries array is required" } }, 400);
  }
  if (entries.length > 50) {
    return c.json({ detail: { message: "entries must contain at most 50 items" } }, 400);
  }

  const prefix = getVoiceDatasetRootPrefix(voiceId);
  const results = await Promise.all(
    entries.map(async (entry) => {
      const audioKey = typeof entry.audio_r2_key === "string" ? entry.audio_r2_key.trim() : "";
      const sourceText = typeof entry.text === "string" ? entry.text.trim() : "";
      if (!audioKey) {
        return {
          audio_r2_key: audioKey,
          error: "audio_r2_key is required",
        };
      }
      if (!audioKey.startsWith(prefix)) {
        return {
          audio_r2_key: audioKey,
          error: `audio_r2_key must belong to ${prefix}`,
        };
      }

      const obj = await c.env.R2.get(audioKey);
      if (!obj) {
        return {
          audio_r2_key: audioKey,
          error: `R2 object not found: ${audioKey}`,
        };
      }

      const transcription = await transcribeAudioWithReviewAsr({
        env: c.env,
        audioBase64: arrayBufferToBase64(await obj.arrayBuffer()),
        languageHint: body.language_code ?? voice.labels?.language ?? "ko",
      });

      return {
        audio_r2_key: audioKey,
        provider: transcription.provider,
        asr_text: transcription.text,
        source_text: sourceText,
        asr_score: sourceText ? asrSimilarity(sourceText, transcription.text) : null,
      };
    })
  );

  return c.json({
    voice_id: voiceId,
    language_code: body.language_code ?? voice.labels?.language ?? "ko",
    results,
  });
});

app.post("/:voice_id/review-texts", async (c) => {
  const voiceId = c.req.param("voice_id");
  const voice = await getVoice(c.env.DB, voiceId);
  if (!voice) {
    return c.json({ detail: { message: "Voice not found" } }, 404);
  }

  const body = (await c.req.json().catch(() => ({}))) as {
    entries?: Array<{ segment?: string; text?: string; duration?: number }>;
  };
  const entries = Array.isArray(body.entries) ? body.entries : [];
  if (entries.length === 0) {
    return c.json({ detail: { message: "entries array is required" } }, 400);
  }
  if (entries.length > 100) {
    return c.json({ detail: { message: "entries must contain at most 100 items" } }, 400);
  }

  const review = await reviewTranscriptEntries({
    env: c.env,
    languageCode: voice.labels?.language,
    entries: entries.map((entry) => ({
      segment: typeof entry.segment === "string" ? entry.segment : undefined,
      text: typeof entry.text === "string" ? entry.text : "",
      duration: typeof entry.duration === "number" ? entry.duration : undefined,
    })),
  });

  return c.json({
    voice_id: voiceId,
    ...review,
  });
});

export default app;
