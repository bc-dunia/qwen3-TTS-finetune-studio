import { Hono } from "hono";
import type { Context } from "hono";
import {
  createTemplate,
  getServerlessEndpoint,
  getServerlessStatus,
  getTemplateById,
  invokeServerlessAsync,
  updateServerlessEndpoint,
} from "../lib/runpod";
import { asrSimilarity, transcribeAudioWithReviewAsr } from "../lib/review-asr";
import { reviewTranscriptEntries } from "../lib/transcript-review";
import { getVoice } from "../lib/d1";
import type { AppContext } from "../types";

const app = new Hono<AppContext>();

const DEFAULT_INFERENCE_IMAGE = "ghcr.io/bc-dunia/qwen3-tts-inference:03e6b487166a708cdf81a0c9af09e81d212ad5a3";
const DEFAULT_PREVIEW_TEXT_KO = "안녕하세요. 현재 체크포인트의 실제 음성 품질과 화자 유사도를 확인하는 검증 샘플입니다.";

const LANGUAGE_ALIASES: Record<string, string> = {
  auto: "auto",
  ko: "korean",
  "ko-kr": "korean",
  korean: "korean",
  en: "english",
  "en-us": "english",
  english: "english",
  ja: "japanese",
  japanese: "japanese",
  zh: "chinese",
  chinese: "chinese",
};

const requireAdminKey = (c: Context<AppContext>) => {
  const expected = String(c.env.API_KEY ?? "").trim();
  const provided = String(c.req.header("xi-api-key") ?? "").trim();
  return Boolean(expected && provided && provided === expected);
};

const normalizeLanguageCode = (languageCode: string | undefined): string => {
  if (!languageCode || !languageCode.trim()) {
    return "auto";
  }
  const normalized = languageCode.trim().toLowerCase().replace(/_/g, "-");
  return LANGUAGE_ALIASES[normalized] ?? normalized;
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

const toEnvPairs = (envValue: unknown): Array<{ key: string; value: string }> => {
  if (Array.isArray(envValue)) {
    return envValue.flatMap((entry) => {
      if (
        !entry ||
        typeof entry !== "object" ||
        typeof (entry as { key?: unknown }).key !== "string" ||
        typeof (entry as { value?: unknown }).value !== "string"
      ) {
        return [];
      }
      const typed = entry as { key: string; value: string };
      return [{ key: typed.key, value: typed.value }];
    });
  }

  if (!envValue || typeof envValue !== "object") {
    return [];
  }

  return Object.entries(envValue).flatMap(([key, value]) =>
    typeof value === "string" ? [{ key, value }] : []
  );
};

const toEnvObject = (envValue: unknown): Record<string, string> =>
  Object.fromEntries(toEnvPairs(envValue).map((entry) => [entry.key, entry.value]));

const sanitizeTemplate = (template: Awaited<ReturnType<typeof getTemplateById>>) =>
  template
    ? {
        id: template.id,
        name: template.name ?? null,
        imageName: template.imageName ?? null,
        isServerless: template.isServerless ?? null,
        containerDiskInGb: template.containerDiskInGb ?? null,
        volumeMountPath: template.volumeMountPath ?? null,
        ports: template.ports ?? null,
        dockerEntrypoint: template.dockerEntrypoint ?? null,
        dockerStartCmd: template.dockerStartCmd ?? null,
        envKeys: Array.from(new Set(toEnvPairs(template.env).map((entry) => entry.key.trim()).filter(Boolean))),
      }
    : null;

const sanitizeEndpoint = (endpoint: Awaited<ReturnType<typeof getServerlessEndpoint>>) =>
  endpoint
    ? {
        id: endpoint.id,
        name: endpoint.name ?? null,
        templateId: endpoint.templateId ?? null,
        workersMin: endpoint.workersMin ?? null,
        workersMax: endpoint.workersMax ?? null,
        idleTimeout: endpoint.idleTimeout ?? null,
        executionTimeoutMs: endpoint.executionTimeoutMs ?? null,
        gpuIds: endpoint.gpuIds ?? null,
        scalerType: endpoint.scalerType ?? null,
      }
    : null;

const arrayBufferToBase64 = (buffer: ArrayBuffer): string => {
  const bytes = new Uint8Array(buffer);
  let binary = "";
  for (let index = 0; index < bytes.length; index += 1) {
    binary += String.fromCharCode(bytes[index]);
  }
  return btoa(binary);
};

app.use("*", async (c, next) => {
  if (!requireAdminKey(c)) {
    return c.json({ detail: { message: "Admin API key required" } }, 401);
  }
  return next();
});

app.get("/runpod/endpoint", async (c) => {
  const endpointId = c.req.query("endpoint_id")?.trim() || c.env.RUNPOD_ENDPOINT_ID;
  const endpoint = await getServerlessEndpoint(c.env, endpointId);
  if (!endpoint) {
    return c.json({ detail: { message: `RunPod endpoint not found: ${endpointId}` } }, 404);
  }

  const template = endpoint.templateId ? await getTemplateById(c.env, endpoint.templateId) : null;
  return c.json({
    endpoint_id: endpointId,
    endpoint: sanitizeEndpoint(endpoint),
    template: sanitizeTemplate(template),
  });
});

app.get("/r2/list", async (c) => {
  const prefix = String(c.req.query("prefix") ?? "").trim();
  const limitRaw = Number(c.req.query("limit") ?? 100);
  const limit = Number.isFinite(limitRaw) ? Math.max(1, Math.min(1000, Math.trunc(limitRaw))) : 100;
  if (!prefix) {
    return c.json({ detail: { message: "prefix query parameter is required" } }, 400);
  }

  const listing = await c.env.R2.list({ prefix, limit });
  return c.json({
    prefix,
    truncated: listing.truncated,
    cursor: listing.truncated ? listing.cursor : null,
    objects: listing.objects.map((obj) => ({
      key: obj.key,
      size: obj.size,
      uploaded: obj.uploaded.toISOString(),
      etag: obj.etag,
      httpEtag: obj.httpEtag,
      contentType: obj.httpMetadata?.contentType ?? null,
    })),
  });
});

app.get("/r2/get", async (c) => {
  const key = String(c.req.query("key") ?? "").trim();
  const rawLimit = Number(c.req.query("limit") ?? 32768);
  const limit = Number.isFinite(rawLimit) ? Math.max(1, Math.min(1_000_000, Math.trunc(rawLimit))) : 32768;
  if (!key) {
    return c.json({ detail: { message: "key query parameter is required" } }, 400);
  }

  const obj = await c.env.R2.get(key);
  if (!obj) {
    return c.json({ detail: { message: `R2 object not found: ${key}` } }, 404);
  }

  const contentType = obj.httpMetadata?.contentType ?? null;
  const text = await obj.text();
  const truncated = text.length > limit;

  return c.json({
    key,
    size: obj.size,
    etag: obj.etag,
    uploaded: obj.uploaded.toISOString(),
    content_type: contentType,
    truncated,
    text: truncated ? text.slice(0, limit) : text,
  });
});

app.post("/dataset/review-texts", async (c) => {
  const body = (await c.req.json().catch(() => ({}))) as {
    language_code?: string;
    entries?: Array<{ segment?: string; text?: string; duration?: number }>;
  };
  const entries = Array.isArray(body.entries) ? body.entries : [];
  if (entries.length === 0) {
    return c.json({ detail: { message: "entries array is required" } }, 400);
  }
  if (entries.length > 50) {
    return c.json({ detail: { message: "entries must contain at most 50 items" } }, 400);
  }

  const review = await reviewTranscriptEntries({
    env: c.env,
    languageCode: body.language_code,
    entries: entries.map((entry) => ({
      segment: typeof entry.segment === "string" ? entry.segment : undefined,
      text: typeof entry.text === "string" ? entry.text : "",
      duration: typeof entry.duration === "number" ? entry.duration : undefined,
    })),
  });

  return c.json(review);
});

app.post("/dataset/retranscribe", async (c) => {
  const body = (await c.req.json().catch(() => ({}))) as {
    language_code?: string;
    entries?: Array<{ audio_r2_key?: string; text?: string }>;
  };
  const entries = Array.isArray(body.entries) ? body.entries : [];
  if (entries.length === 0) {
    return c.json({ detail: { message: "entries array is required" } }, 400);
  }
  if (entries.length > 20) {
    return c.json({ detail: { message: "entries must contain at most 20 items" } }, 400);
  }

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
        languageHint: body.language_code ?? "ko",
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
    language_code: body.language_code ?? "ko",
    results,
  });
});

app.post("/runpod/endpoint/roll-inference", async (c) => {
  const body = (await c.req.json().catch(() => ({}))) as {
    endpoint_id?: string;
    image_name?: string;
    template_name?: string;
    container_disk_gb?: number;
  };
  const endpointId = body.endpoint_id?.trim() || c.env.RUNPOD_ENDPOINT_ID;
  const targetImage = body.image_name?.trim() || DEFAULT_INFERENCE_IMAGE;

  const endpoint = await getServerlessEndpoint(c.env, endpointId);
  if (!endpoint) {
    return c.json({ detail: { message: `RunPod endpoint not found: ${endpointId}` } }, 404);
  }
  if (!endpoint.templateId) {
    return c.json({ detail: { message: `RunPod endpoint has no template: ${endpointId}` } }, 409);
  }

  const currentTemplate = await getTemplateById(c.env, endpoint.templateId);
  if (!currentTemplate) {
    return c.json({ detail: { message: `RunPod template not found: ${endpoint.templateId}` } }, 404);
  }
  const requestedContainerDisk = Number(body.container_disk_gb);
  const containerDiskInGb = Number.isFinite(requestedContainerDisk)
    ? Math.max(20, Math.trunc(requestedContainerDisk))
    : (currentTemplate.containerDiskInGb ?? null);

  const currentEnv = toEnvObject(currentTemplate.env);

  const shortTag = targetImage.split(":").pop() ?? "latest";
  const nextTemplate = await createTemplate(c.env, {
    name:
      body.template_name?.trim() ||
      `${currentTemplate.name ?? "qwen3-tts-inference"}-${shortTag.slice(0, 12)}-${Date.now()}`,
    imageName: targetImage,
    env: currentEnv,
    containerRegistryAuthId: currentTemplate.containerRegistryAuthId ?? null,
    ports: currentTemplate.ports ?? null,
    volumeMountPath: currentTemplate.volumeMountPath ?? null,
    dockerEntrypoint: currentTemplate.dockerEntrypoint ?? null,
    dockerStartCmd: currentTemplate.dockerStartCmd ?? null,
    containerDiskInGb,
    isServerless: currentTemplate.isServerless ?? true,
  });

  const updatedEndpoint = await updateServerlessEndpoint(c.env, endpointId, {
    templateId: nextTemplate.id,
  });
  const updatedTemplate = await getTemplateById(c.env, nextTemplate.id);

  return c.json({
    endpoint_id: endpointId,
    target_image: targetImage,
    previous_template: sanitizeTemplate(currentTemplate),
    created_template: sanitizeTemplate(updatedTemplate),
    endpoint: sanitizeEndpoint(updatedEndpoint),
  });
});

app.post("/runpod/endpoint/scale", async (c) => {
  const body = (await c.req.json().catch(() => ({}))) as {
    endpoint_id?: string;
    workers_min?: number;
    workers_max?: number;
    idle_timeout?: number;
    execution_timeout_ms?: number;
  };

  const endpointId = body.endpoint_id?.trim() || c.env.RUNPOD_ENDPOINT_ID;
  const endpoint = await getServerlessEndpoint(c.env, endpointId);
  if (!endpoint) {
    return c.json({ detail: { message: `RunPod endpoint not found: ${endpointId}` } }, 404);
  }

  const updates: Record<string, unknown> = {};
  const workersMin = Number(body.workers_min);
  const workersMax = Number(body.workers_max);
  const idleTimeout = Number(body.idle_timeout);
  const executionTimeoutMs = Number(body.execution_timeout_ms);

  if (Number.isFinite(workersMin)) {
    updates.workersMin = Math.max(0, Math.trunc(workersMin));
  }
  if (Number.isFinite(workersMax)) {
    updates.workersMax = Math.max(1, Math.trunc(workersMax));
  }
  if (Number.isFinite(idleTimeout)) {
    updates.idleTimeout = Math.max(0, Math.trunc(idleTimeout));
  }
  if (Number.isFinite(executionTimeoutMs)) {
    updates.executionTimeoutMs = Math.max(1000, Math.trunc(executionTimeoutMs));
  }

  if (
    typeof updates.workersMin === "number" &&
    typeof updates.workersMax === "number" &&
    updates.workersMin > updates.workersMax
  ) {
    return c.json({ detail: { message: "workers_min cannot exceed workers_max" } }, 400);
  }

  if (Object.keys(updates).length === 0) {
    return c.json({ detail: { message: "At least one scaling field is required" } }, 400);
  }

  await updateServerlessEndpoint(c.env, endpointId, updates);
  const updatedEndpoint = await getServerlessEndpoint(c.env, endpointId);

  return c.json({
    endpoint_id: endpointId,
    updates,
    endpoint: sanitizeEndpoint(updatedEndpoint),
  });
});

app.post("/runpod/checkpoint-preview", async (c) => {
  const body = (await c.req.json().catch(() => ({}))) as {
    voice_id?: string;
    text?: string;
    language_code?: string;
    checkpoint_r2_prefix?: string;
    model_id?: string;
    seed?: number;
    voice_settings?: Record<string, unknown>;
  };

  const voiceId = body.voice_id?.trim();
  if (!voiceId) {
    return c.json({ detail: { message: "voice_id is required" } }, 400);
  }

  const checkpointPrefix = body.checkpoint_r2_prefix?.trim();
  if (!checkpointPrefix) {
    return c.json({ detail: { message: "checkpoint_r2_prefix is required" } }, 400);
  }

  const voice = await getVoice(c.env.DB, voiceId);
  if (!voice) {
    return c.json({ detail: { message: "Voice not found" } }, 404);
  }

  const { referenceAudioKey, referenceText } = await loadQualityReviewReference(c, voice);
  const inputPayload = {
    text: body.text?.trim() || DEFAULT_PREVIEW_TEXT_KO,
    voice_id: voiceId,
    speaker_name: voice.speaker_name,
    model_id: body.model_id?.trim() || voice.model_id || "qwen3-tts-1.7b",
    language: normalizeLanguageCode(body.language_code),
    seed: Number.isFinite(body.seed) ? Math.trunc(body.seed as number) : 123456,
    voice_settings: body.voice_settings ?? voice.settings,
    checkpoint_info: {
      r2_prefix: checkpointPrefix,
      type: "full" as const,
    },
    quality_review: {
      enable_asr: false,
      enable_speaker: true,
      enable_style: true,
      enable_speed: true,
      allow_below_threshold: true,
      reference_audio_key: referenceAudioKey,
      reference_text: referenceText,
    },
  };

  try {
    const runpodResponse = await invokeServerlessAsync(c.env, c.env.RUNPOD_ENDPOINT_ID, inputPayload);
    return c.json({
      status: "accepted",
      run_id: String(runpodResponse.id ?? ""),
      request: inputPayload,
    });
  } catch (error) {
    return c.json(
      {
        detail: {
          message: error instanceof Error ? error.message : "Preview invocation failed",
        },
      },
      502
    );
  }
});

app.get("/runpod/serverless-status/:run_id", async (c) => {
  const runId = c.req.param("run_id")?.trim();
  if (!runId) {
    return c.json({ detail: { message: "run_id is required" } }, 400);
  }
  const endpointId = c.req.query("endpoint_id")?.trim() || c.env.RUNPOD_ENDPOINT_ID;
  try {
    const payload = await getServerlessStatus(c.env, endpointId, runId);
    return c.json(payload);
  } catch (error) {
    return c.json(
      {
        detail: {
          message: error instanceof Error ? error.message : "Failed to fetch serverless status",
        },
      },
      502
    );
  }
});

export default app;
