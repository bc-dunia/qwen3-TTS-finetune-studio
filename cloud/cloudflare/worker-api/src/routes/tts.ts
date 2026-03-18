import { Hono } from "hono";
import type { Context } from "hono";
import { createGeneration, getVoice } from "../lib/d1";
import { enrichOutputWithReviewAsr } from "../lib/review-asr";
import { getServerlessStatus, invokeServerless, invokeServerlessAsync, type ServerlessSyncResult } from "../lib/runpod";
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

const MAX_TORCH_SEED = 0xFFFFFFFF;
const DEFAULT_INFERENCE_SEED = 123456;
const OVERALL_SCORE_ERROR_RE = /overall_score=([0-9.]+)/i;
const LANGUAGE_ALIASES: Record<string, string> = {
  auto: "auto",
  zh: "chinese",
  "zh-cn": "chinese",
  "zh-tw": "chinese",
  chinese: "chinese",
  en: "english",
  "en-us": "english",
  "en-gb": "english",
  english: "english",
  fr: "french",
  "fr-fr": "french",
  french: "french",
  de: "german",
  "de-de": "german",
  german: "german",
  it: "italian",
  "it-it": "italian",
  italian: "italian",
  ja: "japanese",
  "ja-jp": "japanese",
  japanese: "japanese",
  ko: "korean",
  "ko-kr": "korean",
  korean: "korean",
  pt: "portuguese",
  "pt-br": "portuguese",
  "pt-pt": "portuguese",
  portuguese: "portuguese",
  ru: "russian",
  "ru-ru": "russian",
  russian: "russian",
  es: "spanish",
  "es-es": "spanish",
  "es-mx": "spanish",
  spanish: "spanish",
};

const normalizeSeed = (seed: number | undefined): number => {
  const raw = Number.isFinite(seed) ? Math.trunc(seed as number) : DEFAULT_INFERENCE_SEED;
  const bounded = ((raw % MAX_TORCH_SEED) + MAX_TORCH_SEED) % MAX_TORCH_SEED;
  return bounded === 0 ? 1 : bounded;
};

export const normalizeLanguageCode = (languageCode: string | undefined): string => {
  if (!languageCode || !languageCode.trim()) {
    return "auto";
  }
  const normalized = languageCode.trim().toLowerCase().replace(/_/g, "-");
  return LANGUAGE_ALIASES[normalized] ?? normalized;
};

const parseOverallScoreFromError = (message: string): number | null => {
  const match = OVERALL_SCORE_ERROR_RE.exec(message);
  if (!match || !match[1]) {
    return null;
  }
  const value = Number(match[1]);
  return Number.isFinite(value) ? value : null;
};

const getDatasetPrefixFromReferenceKey = (refAudioKey: string | null): string | null => {
  if (!refAudioKey) {
    return null;
  }
  const index = refAudioKey.lastIndexOf("/");
  if (index <= 0) {
    return null;
  }
  return refAudioKey.slice(0, index);
};

const getDefaultReviewText = (languageCode: string): string => {
  switch (normalizeLanguageCode(languageCode)) {
    case "korean":
      return "안녕하세요. 현재 배포된 음성 품질과 화자 유사도를 점검하는 검증 샘플입니다.";
    case "japanese":
      return "こんにちは。現在デプロイされている音声品質と話者類似度を確認する検証サンプルです。";
    case "chinese":
      return "你好。这是一段用于检查当前部署音色质量和说话人相似度的验证样例。";
    case "english":
    default:
      return "Hello. This is a verification sample for checking deployed voice quality and speaker similarity.";
  }
};

const decodeBase64 = (value: string): Uint8Array => {
  const decoded = atob(value);
  const bytes = new Uint8Array(decoded.length);
  for (let index = 0; index < decoded.length; index += 1) {
    bytes[index] = decoded.charCodeAt(index);
  }
  return bytes;
};

const loadQualityReviewReference = async (
  c: Context<AppContext>,
  voice: NonNullable<Awaited<ReturnType<typeof getVoice>>>
): Promise<{ referenceAudioKey: string | null; referenceText: string }> => {
  const datasetPrefix = getDatasetPrefixFromReferenceKey(voice.ref_audio_r2_key);
  let referenceAudioKey = voice.ref_audio_r2_key;
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
      // Best-effort only.
    }
  }

  return { referenceAudioKey, referenceText };
};

const isLowQualityOutput = (quality: unknown): { low: boolean; reason?: string } => {
  if (!quality || typeof quality !== "object") {
    return { low: false };
  }
  const q = quality as Record<string, unknown>;
  const overall = Number(q.overall_score ?? NaN);
  const duration = Number(q.duration_score ?? NaN);
  const health = Number(q.health_score ?? NaN);

  if (Number.isFinite(overall) && overall < 0.75) {
    return { low: true, reason: `Low overall quality score (${overall.toFixed(2)})` };
  }
  if (Number.isFinite(duration) && duration < 0.35) {
    return { low: true, reason: `Duration mismatch detected (duration_score=${duration.toFixed(2)})` };
  }
  if (Number.isFinite(health) && health < 0.6) {
    return { low: true, reason: `Audio health score too low (${health.toFixed(2)})` };
  }
  return { low: false };
};

const isQualityThresholdError = (message: string): boolean => {
  const m = message.toLowerCase();
  return (
    m.includes("quality threshold") ||
    m.includes("quality gate") ||
    m.includes("below quality") ||
    m.includes("overall_score")
  );
};

const resolveCheckpointInfo = (
  voiceId: string,
  voice: NonNullable<Awaited<ReturnType<typeof getVoice>>>,
  body: TTSRequest
):
  | { r2_prefix: string; type: "full" }
  | { voice_id: string; run_name: string | null; epoch: number | null } => {
  const overridePrefix =
    typeof body.checkpoint_prefix === "string" && body.checkpoint_prefix.trim()
      ? body.checkpoint_prefix.trim()
      : "";

  if (overridePrefix) {
    return {
      r2_prefix: overridePrefix,
      type: "full",
    };
  }

  if (voice.checkpoint_r2_prefix) {
    return {
      r2_prefix: voice.checkpoint_r2_prefix,
      type: "full",
    };
  }

  return {
    voice_id: voiceId,
    run_name: voice.run_name,
    epoch:
      typeof body.checkpoint_epoch === "number" && Number.isFinite(body.checkpoint_epoch)
        ? body.checkpoint_epoch
        : voice.epoch,
  };
};

const buildInputPayload = (
  voiceId: string,
  modelId: string,
  voice: NonNullable<Awaited<ReturnType<typeof getVoice>>>,
  body: TTSRequest,
  seed: number
): Record<string, unknown> => {
  const stylePrompt = typeof body.style_prompt === "string" ? body.style_prompt.trim() : "";
  const instruct = typeof body.instruct === "string" ? body.instruct.trim() : "";
  const combinedInstruct = [stylePrompt, instruct].filter(Boolean).join("\n");

  return {
    text: body.text,
    voice_id: voiceId,
    speaker_name: voice.speaker_name,
    model_id: modelId,
    voice_settings: body.voice_settings ?? voice.settings,
    ...(combinedInstruct ? { instruct: combinedInstruct } : {}),
    seed,
    language: normalizeLanguageCode(body.language_code),
    checkpoint_info: resolveCheckpointInfo(voiceId, voice, body),
  };
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
  const seed = normalizeSeed(body.seed);

  const inputPayload = buildInputPayload(voiceId, modelId, voice, body, seed);
  let syncResult: ServerlessSyncResult;
  try {
    syncResult = await invokeServerless(c.env, c.env.RUNPOD_ENDPOINT_ID, inputPayload);
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown RunPod error";
    return c.json({ detail: { message } }, 502);
  }

  if (syncResult.autoAsync) {
    const jobId = String(syncResult.body.id ?? "");
    if (!jobId) {
      return c.json({ detail: { message: "RunPod returned async status but no job id" } }, 502);
    }
    return c.json(
      { job_id: jobId, status: String(syncResult.body.status ?? "IN_PROGRESS") },
      202
    );
  }

  const runpodResponse = syncResult.body;

  if (typeof runpodResponse.error === "string" && runpodResponse.error) {
    const statusCode = isQualityThresholdError(runpodResponse.error) ? 422 : 502;
    return c.json({ detail: { message: runpodResponse.error } }, statusCode);
  }

  const output = (runpodResponse.output ?? {}) as {
    audio?: string;
    sample_rate?: number;
    duration_ms?: number;
    quality?: Record<string, unknown>;
  };

  if (!output.audio) {
    return c.json({ detail: { message: "RunPod response did not include audio output" } }, 502);
  }

  const qualityCheck = isLowQualityOutput(output.quality);
  if (qualityCheck.low) {
    return c.json(
      {
        detail: {
          message: `Generated audio rejected by quality gate: ${qualityCheck.reason ?? "low quality"}`,
        },
      },
      422
    );
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

app.post("/:voice_id/review", async (c) => {
  const voiceId = c.req.param("voice_id");
  const body = (await c.req.json()) as TTSRequest;
  const voice = await getVoice(c.env.DB, voiceId);
  if (!voice || voice.status !== "ready") {
    return c.json({ detail: { message: "Voice not found or not ready" } }, 404);
  }

  const modelId = body.model_id ?? voice.model_id ?? "qwen3-tts-1.7b";
  const seed = normalizeSeed(body.seed);
  const reviewReference = await loadQualityReviewReference(c, voice);
  const text =
    typeof body.text === "string" && body.text.trim()
      ? body.text.trim()
      : reviewReference.referenceText || getDefaultReviewText(body.language_code ?? voice.labels?.language ?? "ko");

  const inputPayload = {
    ...buildInputPayload(voiceId, modelId, voice, { ...body, text }, seed),
    quality_review: {
      enable_asr: false,
      enable_speaker: Boolean(reviewReference.referenceAudioKey),
      enable_style: Boolean(reviewReference.referenceAudioKey),
      enable_speed: Boolean(reviewReference.referenceAudioKey && reviewReference.referenceText),
      allow_below_threshold: true,
      reference_audio_key: reviewReference.referenceAudioKey,
      reference_text: reviewReference.referenceText,
    },
  };

  let syncResult: ServerlessSyncResult;
  try {
    syncResult = await invokeServerless(c.env, c.env.RUNPOD_ENDPOINT_ID, inputPayload);
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown RunPod error";
    return c.json({ detail: { message } }, 502);
  }

  if (syncResult.autoAsync) {
    return c.json({
      ok: false,
      voice_id: voiceId,
      model_id: modelId,
      text,
      voice_settings: body.voice_settings ?? voice.settings,
      error: "Review generation exceeded sync timeout. Please retry.",
      quality: null,
    });
  }

  const runpodResponse = syncResult.body;

  const topLevelError =
    typeof runpodResponse.error === "string" && runpodResponse.error.trim()
      ? runpodResponse.error.trim()
      : null;
  const output = (runpodResponse.output ?? {}) as {
    audio?: string;
    sample_rate?: number;
    duration_ms?: number;
    error?: string;
    warning?: string;
    quality?: Record<string, unknown>;
  };
  const outputError =
    typeof output.error === "string" && output.error.trim() ? output.error.trim() : null;
  const errorMessage = outputError ?? topLevelError;
  if (errorMessage) {
    return c.json({
      ok: false,
      voice_id: voiceId,
      model_id: modelId,
      text,
      voice_settings: body.voice_settings ?? voice.settings,
      error: errorMessage,
      quality: {
        overall_score: parseOverallScoreFromError(errorMessage),
      },
    });
  }

  if (!output.audio) {
    return c.json({
      ok: false,
      voice_id: voiceId,
      model_id: modelId,
      text,
      voice_settings: body.voice_settings ?? voice.settings,
      error: "RunPod response did not include audio output",
      quality: null,
    });
  }

  let enrichedOutput = output;
  let asrWarning: string | null = null;
  try {
    enrichedOutput = (await enrichOutputWithReviewAsr({
      env: c.env,
      output,
      expectedText: text,
      languageHint: body.language_code ?? voice.labels?.language ?? "auto",
    })) ?? output;
  } catch (error) {
    asrWarning = error instanceof Error ? error.message : "Review ASR enrichment failed";
  }

  return c.json({
    ok: true,
    voice_id: voiceId,
    model_id: modelId,
    text,
    voice_settings: body.voice_settings ?? voice.settings,
    sample_rate: enrichedOutput.sample_rate ?? output.sample_rate ?? 24000,
    duration_ms: enrichedOutput.duration_ms ?? output.duration_ms ?? null,
    quality: enrichedOutput.quality ?? null,
    audio: enrichedOutput.audio ?? output.audio,
    warning: [output.warning, asrWarning].filter((value): value is string => Boolean(value)).join(" | ") || null,
  });
});

app.post("/:voice_id/async", async (c) => {
  const voiceId = c.req.param("voice_id");
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

  const modelId = body.model_id ?? voice.model_id ?? "qwen3-tts-1.7b";
  const seed = normalizeSeed(body.seed);
  const inputPayload = buildInputPayload(voiceId, modelId, voice, body, seed);

  try {
    const runpodResponse = await invokeServerlessAsync(c.env, c.env.RUNPOD_ENDPOINT_ID, inputPayload);
    return c.json({
      job_id: runpodResponse.id,
      status: runpodResponse.status ?? "IN_QUEUE",
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown RunPod error";
    return c.json({ detail: { message } }, 502);
  }
});

app.get("/jobs/:job_id", async (c) => {
  const jobId = c.req.param("job_id");
  try {
    const runpodResponse = await getServerlessStatus(c.env, c.env.RUNPOD_ENDPOINT_ID, jobId);
    const status = String(runpodResponse.status ?? "UNKNOWN");
    const output = (runpodResponse.output ?? {}) as {
      audio?: string;
      duration_ms?: number;
      sample_rate?: number;
      error?: string;
      quality?: Record<string, unknown>;
    };

    if (status === "FAILED") {
      const message = String(output.error ?? runpodResponse.error ?? "Generation failed");
      const statusCode = isQualityThresholdError(message) ? 422 : 502;
      return c.json({ status, error: message }, statusCode);
    }

    if (status !== "COMPLETED") {
      return c.json({ status });
    }

    if (!output.audio) {
      return c.json({ status: "FAILED", error: "RunPod response did not include audio output" }, 502);
    }

    const qualityCheck = isLowQualityOutput(output.quality);
    if (qualityCheck.low) {
      return c.json({ status: "FAILED", error: `Quality gate rejected output: ${qualityCheck.reason ?? "low quality"}` }, 422);
    }

    return c.json({
      status,
      audio: output.audio,
      sample_rate: output.sample_rate ?? 24000,
      duration_ms: output.duration_ms ?? null,
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown RunPod error";
    return c.json({ detail: { message } }, 502);
  }
});

export default app;
