import { Hono } from "hono";
import type { Context } from "hono";
import { createGeneration, getVoice } from "../lib/d1";
import { invokeServerless } from "../lib/runpod";
import { authMiddleware } from "../middleware/auth";
import type { AppContext, Generation, TTSRequest } from "../types";

const app = new Hono<AppContext>();
app.use("*", authMiddleware);

const OUTPUT_FORMAT_TO_CONTENT_TYPE: Record<string, string> = {
  wav_24000: "audio/wav",
  wav_44100: "audio/wav",
};

const getContentTypeForFormat = (outputFormat: string): string =>
  OUTPUT_FORMAT_TO_CONTENT_TYPE[outputFormat] ?? "audio/wav";

const decodeBase64 = (value: string): Uint8Array => {
  const decoded = atob(value);
  const bytes = new Uint8Array(decoded.length);
  for (let index = 0; index < decoded.length; index += 1) {
    bytes[index] = decoded.charCodeAt(index);
  }
  return bytes;
};

const runTtsRequest = async (c: Context<AppContext>): Promise<Response> => {
  const voiceId = c.req.param("voice_id");
  const outputFormat = c.req.query("output_format") ?? "wav_24000";

  if (!["wav_24000", "wav_44100"].includes(outputFormat)) {
    return c.json(
      {
        detail: {
          message: "Only wav_24000 and wav_44100 are currently supported",
        },
      },
      400
    );
  }

  const body = (await c.req.json()) as TTSRequest;
  if (!body.text || !body.text.trim()) {
    return c.json({ detail: { message: "text is required" } }, 400);
  }
  if (body.text.length > 5000) {
    return c.json({ detail: { message: "text must be 5000 characters or less" } }, 400);
  }

  const voice = await getVoice(c.env.DB, voiceId);
  if (!voice || voice.status !== "ready") {
    return c.json({ detail: { message: "Voice not found or not ready" } }, 404);
  }

  const generationId = crypto.randomUUID();
  const requestId = crypto.randomUUID();
  const startedAt = Date.now();
  const modelId = body.model_id ?? voice.model_id ?? "qwen3-tts-1.7b";

  const inputPayload: Record<string, unknown> = {
    text: body.text,
    voice_id: voiceId,
    speaker_name: voice.speaker_name,
    model_id: modelId,
    voice_settings: body.voice_settings ?? voice.settings,
    seed: body.seed,
    language: body.language_code ?? "auto",
    checkpoint_info: voice.checkpoint_r2_prefix
      ? {
          r2_prefix: voice.checkpoint_r2_prefix,
          type: "full" as const,
        }
      : {
          voice_id: voiceId,
          run_name: voice.run_name,
          epoch: voice.epoch,
        },
  };
  let runpodResponse: Record<string, unknown>;
  try {
    runpodResponse = await invokeServerless(c.env, c.env.RUNPOD_ENDPOINT_ID, inputPayload);
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown RunPod error";
    return c.json({ detail: { message } }, 502);
  }

  if (typeof runpodResponse.error === "string" && runpodResponse.error) {
    return c.json({ detail: { message: runpodResponse.error } }, 502);
  }

  const output = (runpodResponse.output ?? {}) as {
    audio?: string;
    sample_rate?: number;
    duration_ms?: number;
  };

  if (!output.audio) {
    return c.json({ detail: { message: "RunPod response did not include audio output" } }, 502);
  }

  let audioBytes: Uint8Array;
  try {
    audioBytes = decodeBase64(output.audio);
  } catch {
    return c.json({ detail: { message: "Failed to decode audio from RunPod" } }, 502);
  }

  const extension = "wav";
  const audioKey = `audio/${voiceId}/${generationId}.${extension}`;

  await c.env.R2.put(audioKey, audioBytes, {
    httpMetadata: {
      contentType: getContentTypeForFormat(outputFormat),
    },
  });

  const generation: Generation = {
    generation_id: generationId,
    voice_id: voiceId,
    model_id: modelId,
    text: body.text,
    audio_r2_key: audioKey,
    output_format: outputFormat,
    duration_ms: output.duration_ms ?? null,
    latency_ms: Date.now() - startedAt,
    settings: body.voice_settings ?? voice.settings,
    created_at: Date.now(),
  };
  await createGeneration(c.env.DB, generation);

  return new Response(audioBytes, {
    status: 200,
    headers: {
      "Content-Type": getContentTypeForFormat(outputFormat),
      "x-request-id": requestId,
      "x-generation-id": generationId,
    },
  });
};

app.post("/:voice_id", async (c) => runTtsRequest(c));

app.post("/:voice_id/stream", async (c) => {
  return runTtsRequest(c);
});

export default app;
