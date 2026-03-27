import type { Env } from "../types";

const CLOUDFLARE_REVIEW_MODEL = "@cf/meta/llama-3.1-8b-instruct";
const OPENAI_REVIEW_URL = "https://api.openai.com/v1/chat/completions";
const DEFAULT_OPENAI_REVIEW_MODEL = "gpt-5-mini";

const normalizeLanguageCode = (value: string | undefined): string => {
  const normalized = (value ?? "").trim().toLowerCase();
  if (!normalized) {
    return "en";
  }
  if (normalized === "kr" || normalized === "ko-kr") {
    return "ko";
  }
  if (normalized === "jp" || normalized === "ja-jp") {
    return "ja";
  }
  if (
    normalized === "cn" ||
    normalized === "zh-cn" ||
    normalized === "zh-tw" ||
    normalized === "zh-hans" ||
    normalized === "zh-hant"
  ) {
    return "zh";
  }
  if (normalized === "en-us" || normalized === "en-gb") {
    return "en";
  }
  return normalized;
};

const getReviewLanguageLabel = (languageCode: string | undefined): string => {
  switch (normalizeLanguageCode(languageCode)) {
    case "en":
      return "English";
    case "ja":
      return "Japanese";
    case "zh":
      return "Chinese";
    case "ko":
      return "Korean";
    default:
      return "English";
  }
};

const getReviewSystemPrompt = (languageCode: string | undefined): string => {
  const languageLabel = getReviewLanguageLabel(languageCode);
  return `You are an expert reviewer for ${languageLabel} speech transcripts.
Review speech-to-text output and correct only clear transcription mistakes.

Respond with JSON only using this exact shape:
- top level object: {"reviews":[...]}
- each review: {"score":1-5,"corrected":"...","issues":["..."]}
- score meanings: 5=perfect, 4=minor issue, 3=needs correction, 2=severe issue, 1=unusable

Review criteria:
- repetition or hallucination: penalize meaningless repeated phrases
- truncation: flag text that is obviously cut off
- typos or homophones: correct only when the evidence is clear
- unintelligible text: flag parts that do not make sense
- numbers and proper nouns: fix only obvious mistakes and do not guess

Important:
- this is for TTS training, so preserve the spoken phrasing exactly
- do not rewrite the sentence to sound smoother or more literary
- if the correction is uncertain, keep the original text
- keep the transcript in ${languageLabel}; do not translate
- output no text outside the JSON`;
};

type TranscriptReviewEntry = {
  segment?: string;
  text: string;
  duration?: number;
};

export type TranscriptReviewResult = {
  segment?: string;
  original_text: string;
  corrected: string;
  score: number;
  issues: string[];
};

type WorkersAiBinding = {
  run: (model: string, input: Record<string, unknown>) => Promise<unknown>;
};

const buildSkippedReview = (
  fallback: TranscriptReviewEntry,
  issue: string = "empty_transcript"
): TranscriptReviewResult => {
  const hasText = fallback.text.trim().length > 0;
  return {
    segment: fallback.segment,
    original_text: fallback.text,
    corrected: fallback.text,
    score: hasText ? 3 : 1,
    issues: hasText ? (issue === "missing_review_result" ? [issue] : []) : [issue],
  };
};

const clampScore = (value: unknown): number => {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return 3;
  }
  return Math.max(1, Math.min(5, Math.round(numeric)));
};

const normalizeIssues = (value: unknown): string[] => {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.flatMap((issue) => (typeof issue === "string" && issue.trim() ? [issue.trim()] : []));
};

const normalizeSingleReview = (
  value: unknown,
  fallback: TranscriptReviewEntry
): TranscriptReviewResult => {
  const candidate = value && typeof value === "object" ? (value as Record<string, unknown>) : {};
  const corrected =
    typeof candidate.corrected === "string" && candidate.corrected.trim()
      ? candidate.corrected.trim()
      : fallback.text;

  return {
    segment: fallback.segment,
    original_text: fallback.text,
    corrected,
    score: clampScore(candidate.score),
    issues: normalizeIssues(candidate.issues),
  };
};

const normalizeReviews = (
  raw: unknown,
  entries: TranscriptReviewEntry[]
): TranscriptReviewResult[] | null => {
  const container = raw && typeof raw === "object" ? (raw as Record<string, unknown>) : null;
  const directReviews = Array.isArray(container?.reviews)
    ? container?.reviews
    : Array.isArray(raw)
      ? raw
      : null;

  if (!directReviews) {
    return null;
  }

  return entries.map((entry, index) => normalizeSingleReview(directReviews[index], entry));
};

const parseJsonLike = (value: string): unknown => {
  const cleaned = value.trim().replace(/^```(?:json)?\s*/i, "").replace(/\s*```$/, "");
  return JSON.parse(cleaned);
};

const extractWorkersAiPayload = (raw: unknown): unknown => {
  if (!raw || typeof raw !== "object") {
    return raw;
  }
  const candidate = raw as Record<string, unknown>;
  if (candidate.response !== undefined) {
    return candidate.response;
  }
  if (candidate.result !== undefined) {
    return candidate.result;
  }
  return raw;
};

const workersAiReview = async (
  ai: WorkersAiBinding,
  entries: TranscriptReviewEntry[],
  languageCode: string | undefined
): Promise<TranscriptReviewResult[]> => {
  const languageLabel = getReviewLanguageLabel(languageCode);
  const userPrompt = entries
    .map((entry, index) => {
      const seg = entry.segment ? `seg=${entry.segment}` : "seg=unknown";
      const duration =
        typeof entry.duration === "number" && Number.isFinite(entry.duration)
          ? `, ${entry.duration.toFixed(2)}s`
          : "";
      return `[${index + 1}] (${seg}${duration}): "${entry.text}"`;
    })
    .join("\n");

  const schema = {
    type: "object",
    properties: {
      reviews: {
        type: "array",
        items: {
          type: "object",
          properties: {
            score: { type: "integer" },
            corrected: { type: "string" },
            issues: {
              type: "array",
              items: { type: "string" },
            },
          },
          required: ["score", "corrected", "issues"],
          additionalProperties: false,
        },
      },
    },
    required: ["reviews"],
    additionalProperties: false,
  };

  const raw = await ai.run(CLOUDFLARE_REVIEW_MODEL, {
    messages: [
      { role: "system", content: getReviewSystemPrompt(languageCode) },
      {
        role: "user",
        content: `Review ${entries.length} ${languageLabel} transcript(s).\n${userPrompt}`,
      },
    ],
    temperature: 0.1,
    max_tokens: 2048,
    response_format: {
      type: "json_schema",
      json_schema: schema,
    },
  });

  const payload = extractWorkersAiPayload(raw);
  const parsed =
    typeof payload === "string"
      ? parseJsonLike(payload)
      : payload && typeof payload === "object"
        ? payload
        : null;
  const reviews = normalizeReviews(parsed, entries);
  if (!reviews) {
    throw new Error("Workers AI review response was not valid JSON");
  }
  return reviews;
};

const openAiReview = async (
  env: Pick<Env, "OPENAI_API_KEY" | "OPENAI_REVIEW_MODEL">,
  entries: TranscriptReviewEntry[],
  languageCode: string | undefined
): Promise<TranscriptReviewResult[]> => {
  const apiKey = String(env.OPENAI_API_KEY ?? "").trim();
  if (!apiKey) {
    throw new Error("OPENAI_API_KEY is not configured");
  }

  const languageLabel = getReviewLanguageLabel(languageCode);
  const userPrompt = entries
    .map((entry, index) => {
      const seg = entry.segment ? `seg=${entry.segment}` : "seg=unknown";
      const duration =
        typeof entry.duration === "number" && Number.isFinite(entry.duration)
          ? `, ${entry.duration.toFixed(2)}s`
          : "";
      return `[${index + 1}] (${seg}${duration}): "${entry.text}"`;
    })
    .join("\n");

  const response = await fetch(OPENAI_REVIEW_URL, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model:
        String(env.OPENAI_REVIEW_MODEL ?? DEFAULT_OPENAI_REVIEW_MODEL).trim() ||
        DEFAULT_OPENAI_REVIEW_MODEL,
      messages: [
        { role: "system", content: getReviewSystemPrompt(languageCode) },
        {
          role: "user",
          content: `Review ${entries.length} ${languageLabel} transcript(s).\n${userPrompt}`,
        },
      ],
      temperature: 0.1,
      response_format: { type: "json_object" },
    }),
  });

  if (!response.ok) {
    const detail = (await response.text()).slice(0, 400);
    throw new Error(`OpenAI review failed (${response.status}): ${detail}`);
  }

  const payload = (await response.json()) as Record<string, unknown>;
  const choices = Array.isArray(payload.choices) ? payload.choices : [];
  const message = choices[0] && typeof choices[0] === "object" ? (choices[0] as Record<string, unknown>).message : null;
  const content =
    message && typeof message === "object" && typeof (message as Record<string, unknown>).content === "string"
      ? ((message as Record<string, unknown>).content as string)
      : "";

  const reviews = normalizeReviews(parseJsonLike(content), entries);
  if (!reviews) {
    throw new Error("OpenAI review response was not valid JSON");
  }
  return reviews;
};

export const reviewTranscriptEntries = async ({
  env,
  entries,
  languageCode,
}: {
  env: Pick<Env, "AI" | "OPENAI_API_KEY" | "OPENAI_REVIEW_MODEL">;
  entries: TranscriptReviewEntry[];
  languageCode?: string;
}): Promise<{ provider: string; results: TranscriptReviewResult[] }> => {
  const normalizedLanguageCode = normalizeLanguageCode(languageCode);
  const normalizedEntries = entries.map((entry) => ({
    segment: typeof entry.segment === "string" ? entry.segment.trim() || undefined : undefined,
    text: typeof entry.text === "string" ? entry.text.trim() : "",
    duration: typeof entry.duration === "number" && Number.isFinite(entry.duration) ? entry.duration : undefined,
  }));
  const reviewableEntries = normalizedEntries.flatMap((entry, index) =>
    entry.text.length > 0 ? [{ entry, index }] : []
  );

  if (reviewableEntries.length === 0) {
    return {
      provider: "none",
      results: normalizedEntries.map((entry) => buildSkippedReview(entry)),
    };
  }

  const mergeReviewedResults = (reviewed: TranscriptReviewResult[]): TranscriptReviewResult[] => {
    const merged = normalizedEntries.map((entry) => buildSkippedReview(entry));
    reviewableEntries.forEach(({ entry, index }, reviewIndex) => {
      merged[index] = reviewed[reviewIndex] ?? buildSkippedReview(entry, "missing_review_result");
    });
    return merged;
  };

  const workersAi = env.AI as WorkersAiBinding | undefined;
  let workersAiError: Error | null = null;
  if (workersAi && typeof workersAi.run === "function") {
    try {
      const reviewed = await workersAiReview(
        workersAi,
        reviewableEntries.map(({ entry }) => entry),
        normalizedLanguageCode
      );
      return {
        provider: "cloudflare-workers-ai",
        results: mergeReviewedResults(reviewed),
      };
    } catch (error) {
      workersAiError = error instanceof Error ? error : new Error("Workers AI review failed");
      if (!String(env.OPENAI_API_KEY ?? "").trim()) {
        throw workersAiError;
      }
    }
  }

  try {
    const reviewed = await openAiReview(
      env,
      reviewableEntries.map(({ entry }) => entry),
      normalizedLanguageCode
    );
    return {
      provider: "openai",
      results: mergeReviewedResults(reviewed),
    };
  } catch (error) {
    const openAiError = error instanceof Error ? error : new Error("OpenAI review failed");
    if (workersAiError) {
      throw new Error(`${workersAiError.message}; ${openAiError.message}`);
    }
    throw openAiError;
  }
};
