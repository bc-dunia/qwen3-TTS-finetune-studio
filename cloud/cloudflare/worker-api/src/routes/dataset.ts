import { Hono } from "hono";
import { authMiddleware } from "../middleware/auth";
import { getVoice, updateVoice } from "../lib/d1";
import type { AppContext, CreateDatasetRequest, DatasetInfo } from "../types";

const app = new Hono<AppContext>();
app.use("*", authMiddleware);

const SAFE_DATASET_NAME_RE = /^[a-zA-Z0-9._-]{1,128}$/;

const extractFilename = (r2Key: string): string => {
  const parts = r2Key.split("/");
  return parts[parts.length - 1] ?? r2Key;
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

  for (const item of body.items) {
    const audioFilename = extractFilename(item.audio_r2_key);

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

  // Update voice with ref_audio_r2_key
  await updateVoice(c.env.DB, voiceId, {
    ref_audio_r2_key: body.ref_audio_r2_key,
  });

  return c.json({
    dataset_name: body.dataset_name,
    dataset_r2_prefix: datasetPrefix,
    items_count: body.items.length,
    ref_audio_r2_key: body.ref_audio_r2_key,
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

export default app;
