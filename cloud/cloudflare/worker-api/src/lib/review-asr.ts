import type { Env } from "../types";

const OPENAI_TRANSCRIPTION_URL = "https://api.openai.com/v1/audio/transcriptions";
const DEFAULT_OPENAI_TRANSCRIBE_MODEL = "gpt-4o-mini-transcribe";
const CLOUDFLARE_TRANSCRIBE_MODEL = "@cf/openai/whisper-large-v3-turbo";
const NORMALIZED_TEXT_RE = /[^\p{Letter}\p{Number}]+/gu;

const LANGUAGE_HINT_TO_CODE: Record<string, string | null> = {
  auto: null,
  en: "en",
  "en-us": "en",
  "en-gb": "en",
  english: "en",
  ko: "ko",
  "ko-kr": "ko",
  korean: "ko",
  ja: "ja",
  "ja-jp": "ja",
  japanese: "ja",
  zh: "zh",
  "zh-cn": "zh",
  "zh-tw": "zh",
  chinese: "zh",
};

type ReviewOutput = {
  audio?: string;
  quality?: Record<string, unknown>;
  [key: string]: unknown;
};

type WorkersAiBinding = {
  run: (model: string, input: Record<string, unknown>) => Promise<unknown>;
};

const decodeBase64 = (value: string): Uint8Array => {
  const decoded = atob(value);
  const bytes = new Uint8Array(decoded.length);
  for (let index = 0; index < decoded.length; index += 1) {
    bytes[index] = decoded.charCodeAt(index);
  }
  return bytes;
};

const normalizeText = (text: string): string =>
  (text || "").normalize("NFKC").trim().toLowerCase().replace(NORMALIZED_TEXT_RE, "");

const levenshteinRatio = (a: string, b: string): number => {
  if (a === b) {
    return 1;
  }
  if (!a || !b) {
    return 0;
  }
  const prev = Array.from({ length: b.length + 1 }, (_, index) => index);
  for (let i = 1; i <= a.length; i += 1) {
    const curr = [i, ...Array.from({ length: b.length }, () => 0)];
    for (let j = 1; j <= b.length; j += 1) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      curr[j] = Math.min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost);
    }
    prev.splice(0, prev.length, ...curr);
  }
  return 1 - (prev[b.length] / Math.max(a.length, b.length, 1));
};

export const asrSimilarity = (target: string, prediction: string): number =>
  levenshteinRatio(normalizeText(target), normalizeText(prediction));

export const resolveAsrLanguageCode = (languageHint: string | null | undefined): string | null => {
  if (!languageHint || !languageHint.trim()) {
    return null;
  }
  const normalized = languageHint.trim().toLowerCase().replace(/_/g, "-");
  return LANGUAGE_HINT_TO_CODE[normalized] ?? normalized;
};

const extractTextFromOpenAiResponse = (payload: unknown): string => {
  if (!payload || typeof payload !== "object") {
    return "";
  }
  const text = (payload as Record<string, unknown>).text;
  return typeof text === "string" ? text.trim() : "";
};

const transcribeWithWorkersAi = async (
  ai: WorkersAiBinding,
  audioBase64: string,
  languageHint: string | null
): Promise<{ text: string; provider: string }> => {
  const response = await ai.run(CLOUDFLARE_TRANSCRIBE_MODEL, {
    audio: audioBase64,
    ...(languageHint ? { language: languageHint } : {}),
    condition_on_previous_text: false,
    vad_filter: true,
  });
  return {
    text: extractTextFromOpenAiResponse(response),
    provider: "cloudflare-workers-ai",
  };
};

const transcribeWithOpenAi = async (
  env: Pick<Env, "OPENAI_API_KEY" | "OPENAI_TRANSCRIBE_MODEL">,
  audioBase64: string,
  languageHint: string | null
): Promise<{ text: string; provider: string }> => {
  const apiKey = String(env.OPENAI_API_KEY ?? "").trim();
  if (!apiKey) {
    throw new Error("OPENAI_API_KEY is not configured");
  }

  const audioBytes = decodeBase64(audioBase64);
  const formData = new FormData();
  formData.append(
    "file",
    new File([audioBytes], "review.wav", { type: "audio/wav" })
  );
  formData.append(
    "model",
    String(env.OPENAI_TRANSCRIBE_MODEL ?? DEFAULT_OPENAI_TRANSCRIBE_MODEL).trim() ||
      DEFAULT_OPENAI_TRANSCRIBE_MODEL
  );
  if (languageHint) {
    formData.append("language", languageHint);
  }
  formData.append("response_format", "json");

  const response = await fetch(OPENAI_TRANSCRIPTION_URL, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
    },
    body: formData,
  });

  if (!response.ok) {
    const detail = (await response.text()).slice(0, 300);
    throw new Error(`OpenAI transcription failed (${response.status}): ${detail}`);
  }

  return {
    text: extractTextFromOpenAiResponse(await response.json()),
    provider: "openai",
  };
};

export const transcribeAudioWithReviewAsr = async ({
  env,
  audioBase64,
  languageHint,
}: {
  env: Pick<Env, "AI" | "OPENAI_API_KEY" | "OPENAI_TRANSCRIBE_MODEL">;
  audioBase64: string;
  languageHint?: string | null;
}): Promise<{ text: string; provider: string }> => {
  const resolvedLanguage = resolveAsrLanguageCode(languageHint);
  const workersAi = env.AI as WorkersAiBinding | undefined;
  if (workersAi && typeof workersAi.run === "function") {
    try {
      return await transcribeWithWorkersAi(workersAi, audioBase64, resolvedLanguage);
    } catch (error) {
      const detail = error instanceof Error ? error.message : "Unknown Workers AI error";
      if (!String(env.OPENAI_API_KEY ?? "").trim()) {
        throw new Error(`Workers AI transcription failed: ${detail}`);
      }
    }
  }
  return transcribeWithOpenAi(env, audioBase64, resolvedLanguage);
};

export const enrichOutputWithReviewAsr = async ({
  env,
  output,
  expectedText,
  languageHint,
}: {
  env: Pick<Env, "AI" | "OPENAI_API_KEY" | "OPENAI_TRANSCRIBE_MODEL">;
  output: ReviewOutput | null;
  expectedText: string;
  languageHint?: string | null;
}): Promise<ReviewOutput | null> => {
  if (!output?.audio || !expectedText.trim()) {
    return output;
  }

  const result = await transcribeAudioWithReviewAsr({
    env,
    audioBase64: output.audio,
    languageHint,
  });
  const transcription = result.text;
  const provider = result.provider;
  const quality = output.quality && typeof output.quality === "object" ? { ...output.quality } : {};
  const score = asrSimilarity(expectedText, transcription);
  quality.asr_provider = provider;
  quality.asr_text = transcription;
  quality.asr_similarity = score;
  quality.asr_score = score;

  return {
    ...output,
    quality,
  };
};
