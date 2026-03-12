import type { Env, TrainingAdvice, TrainingAdviceMode, TrainingConfig, TrainingJob, Voice } from "../types";
import { buildTrainingCheckoutSearch } from "./training-checkout";
import { sanitizeConfig, diversifyAdviceForActiveRuns, isAdviceActiveJob } from "./training-advisor";

const OPENAI_API_URL = "https://api.openai.com/v1/chat/completions";
const DEFAULT_ADVISOR_MODEL = "gpt-4.1";

const VALID_MODES: Set<string> = new Set([
  "compare-first",
  "dataset-first",
  "tone-explore",
  "stability-reset",
  "checkpoint-window",
  "hold-current",
]);

const SYSTEM_PROMPT = `You are a Qwen3-TTS fine-tuning advisor. You analyze past training runs and recommend the next training configuration.

## Domain

Qwen3-TTS is a text-to-speech model fine-tuned per speaker. Each training job runs supervised fine-tuning (SFT) with these tunable hyperparameters:

| Parameter | Range | Effect |
|-----------|-------|--------|
| model_size | "0.6B" or "1.7B" | Larger model = richer tone but slower, needs A100 |
| batch_size | 1-4 | Higher = smoother gradients but more VRAM |
| num_epochs | 3-20 | More epochs = longer training, risk of overfitting |
| learning_rate | 1e-6 to 5e-5 | Higher = faster convergence but less stable |
| gradient_accumulation_steps | 1-8 | Effective batch multiplier |
| subtalker_loss_weight | 0.1-0.5 | Higher = more emphasis on prosody/rhythm submodel |
| save_every_n_epochs | 1-5 | 1 = checkpoint every epoch (best for sweeps) |
| seed | any integer | Different seeds explore different local optima |
| gpu_type_id | "NVIDIA L40S" (0.6B) or "NVIDIA A100-SXM4-80GB" (1.7B) | Must match model_size |

## Validation

After training, checkpoints are validated automatically:
- **overall_score** (0-1): Composite of ASR, tone, speed similarity to reference speaker
- **asr_score**: How accurately the model reproduces the target text
- **tone_score**: Speaker timbre/voice quality match
- **speed_score**: Speaking rate match to reference

A checkpoint passes when overall_score exceeds a threshold (~0.75). Common failure patterns:
- ASR failures → transcript/dataset quality issues, not hyperparameter issues
- Tone failures → learning rate too high, subtalker weight wrong, or need more epochs
- Speed failures → overfitting past the sweet spot epoch, or subtalker weight too low
- Infra failures (no audio, stalled) → provisioning issues, not training issues

## Strategy Modes

Pick one mode that best fits the situation:
- "tone-explore": Lower LR + seed rotation to preserve speaker phrasing
- "stability-reset": Conservative config to recover from repeated failures
- "checkpoint-window": Short sweep with save_every_n_epochs=1 to find the sweet spot epoch
- "compare-first": Scores are too close; user should listen before spending another run
- "dataset-first": ASR failures dominate; fix dataset before retraining
- "hold-current": Current champion is stable; no urgent need for another run

## Response Format

Respond with JSON only, matching this exact shape:
{
  "mode": "<one of the 6 modes above>",
  "title": "<concise 3-8 word recommendation title>",
  "summary": "<2-3 sentence explanation of why this strategy and what it changes>",
  "confidence": "high" or "medium",
  "reasons": ["<reason 1>", "<reason 2>", ...],
  "suggestedConfig": { "model_size": "...", "batch_size": N, "num_epochs": N, "learning_rate": N, "gradient_accumulation_steps": N, "subtalker_loss_weight": N, "save_every_n_epochs": N, "seed": N, "whisper_language": "...", "gpu_type_id": "..." },
  "compareFirst": true/false,
  "reviewDatasetFirst": true/false,
  "primaryActionLabel": "<button label, 2-4 words>"
}

Rules:
- suggestedConfig MUST include ALL fields listed above
- gpu_type_id must be "NVIDIA L40S" for 0.6B or "NVIDIA A100-SXM4-80GB" for 1.7B
- whisper_language should match the voice's language
- compareFirst=true when 2+ validated runs exist and the user should listen/compare before another run (any mode)
- reviewDatasetFirst=true when dataset quality issues need fixing first (any mode)
- The suggested config MUST differ from all currently active runs
- Prefer strategies not yet explored in history`;

function getCheckout(job: TrainingJob) {
  if (!job.checkout_search) {
    job.checkout_search = buildTrainingCheckoutSearch(job);
  }
  return job.checkout_search;
}

function summarizeJob(job: TrainingJob): string {
  const checkout = getCheckout(job);
  const score =
    checkout.selected?.score ??
    checkout.manual_promoted?.score ??
    checkout.champion?.score ??
    null;
  const epoch =
    checkout.selected?.epoch ??
    checkout.manual_promoted?.epoch ??
    checkout.champion?.epoch ??
    null;
  const msg = checkout.message ?? job.error_message ?? null;
  const config = job.config;

  const parts: string[] = [
    `status=${job.status}`,
    score !== null ? `score=${score.toFixed(3)}` : null,
    epoch !== null ? `best_epoch=${epoch}` : null,
    `lr=${config.learning_rate}`,
    `epochs=${config.num_epochs}`,
    `batch=${config.batch_size}`,
    `seed=${config.seed ?? 0}`,
    `subtalker=${config.subtalker_loss_weight ?? 0.3}`,
    `save_every=${config.save_every_n_epochs ?? 1}`,
    checkout.validation_passed === true ? "validated=pass" : null,
    checkout.status === "rejected" ? "validated=rejected" : null,
    checkout.manual_promoted ? "manual_promoted=true" : null,
    msg ? `msg="${msg.slice(0, 120)}"` : null,
  ].filter((part): part is string => part !== null);

  const evaluatedEpochs = checkout.evaluated;
  if (evaluatedEpochs.length > 0) {
    const epochScores = evaluatedEpochs
      .map((ep) => `e${ep.epoch}=${ep.score.toFixed(3)}${ep.ok ? "✓" : "✗"}`)
      .join(" ");
    parts.push(`checkpoints=[${epochScores}]`);
  }

  return parts.join(", ");
}

function buildAnalytics(voice: Voice, completedJobs: TrainingJob[]): string {
  const lines: string[] = ["## Analytics"];

  // Score trend
  const validatedJobs = completedJobs
    .filter((job) => getCheckout(job).validation_passed === true)
    .sort((a, b) => (Number(a.created_at) || 0) - (Number(b.created_at) || 0));
  if (validatedJobs.length >= 2) {
    const scores = validatedJobs.map((job) => {
      const checkout = getCheckout(job);
      return checkout.selected?.score ?? checkout.manual_promoted?.score ?? checkout.champion?.score ?? 0;
    });
    const firstHalf = scores.slice(0, Math.ceil(scores.length / 2));
    const secondHalf = scores.slice(Math.ceil(scores.length / 2));
    const avgFirst = firstHalf.reduce((sum, s) => sum + s, 0) / firstHalf.length;
    const avgSecond = secondHalf.reduce((sum, s) => sum + s, 0) / secondHalf.length;
    const diff = avgSecond - avgFirst;
    const trend = diff > 0.005 ? "improving" : diff < -0.005 ? "declining" : "flat";
    lines.push(`Score trend: ${trend} (early avg=${avgFirst.toFixed(3)}, recent avg=${avgSecond.toFixed(3)})`);
  }

  // Tried parameter ranges
  const numSort = (a: number, b: number) => a - b;
  const triedLRs = [...new Set(completedJobs.map((j) => Number(j.config.learning_rate)).filter(Number.isFinite))].sort(numSort);
  const triedSeeds = [...new Set(completedJobs.map((j) => Number(j.config.seed)).filter(Number.isFinite))].sort(numSort);
  const triedEpochs = [...new Set(completedJobs.map((j) => Number(j.config.num_epochs)).filter(Number.isFinite))].sort(numSort);
  const triedSubtalker = [...new Set(completedJobs.map((j) => Number(j.config.subtalker_loss_weight)).filter(Number.isFinite))].sort(numSort);
  lines.push(`Tried LRs: [${triedLRs.join(", ")}]`);
  lines.push(`Tried seeds: [${triedSeeds.join(", ")}]`);
  lines.push(`Tried epoch counts: [${triedEpochs.join(", ")}]`);
  lines.push(`Tried subtalker weights: [${triedSubtalker.join(", ")}]`);

  // Failure breakdown (recent 6 rejected)
  const rejectedJobs = completedJobs.filter((job) => getCheckout(job).status === "rejected");
  if (rejectedJobs.length > 0) {
    const recentMessages = rejectedJobs
      .slice(0, 6)
      .map((job) => (getCheckout(job).message ?? job.error_message ?? "").toLowerCase())
      .filter(Boolean);
    const asrFailures = recentMessages.filter((m) => m.includes("asr_score") || m.includes("missing asr")).length;
    const toneFailures = recentMessages.filter((m) => m.includes("tone_score")).length;
    const speedFailures = recentMessages.filter((m) => m.includes("speed_score")).length;
    const overallFailures = recentMessages.filter((m) => m.includes("overall_score") || m.includes("quality threshold")).length;
    const infraFailures = recentMessages.filter((m) => m.includes("no audio") || m.includes("stalled") || m.includes("recovery")).length;
    lines.push(`Recent failures (last ${recentMessages.length}): ASR=${asrFailures} tone=${toneFailures} speed=${speedFailures} overall=${overallFailures} infra=${infraFailures}`);
  }

  // Heuristic signals
  const signals: string[] = [];
  const bestValidated = validatedJobs.length > 0 ? validatedJobs[validatedJobs.length - 1] : null;
  if (bestValidated) {
    const bestEpoch = getCheckout(bestValidated).selected?.epoch ?? getCheckout(bestValidated).manual_promoted?.epoch ?? getCheckout(bestValidated).champion?.epoch;
    const maxEvaluated = Math.max(...getCheckout(bestValidated).evaluated.map((ep) => ep.epoch), 0);
    if (bestEpoch !== null && bestEpoch !== undefined && maxEvaluated > 0 && bestEpoch <= maxEvaluated - 2) {
      signals.push(`early_peak_suspected=true (best_epoch=${bestEpoch}, max_evaluated=${maxEvaluated})`);
    }
  }
  if (validatedJobs.length >= 2) {
    const topScores = validatedJobs
      .map((job) => getCheckout(job).selected?.score ?? getCheckout(job).manual_promoted?.score ?? getCheckout(job).champion?.score ?? 0)
      .sort((a, b) => b - a);
    if (topScores.length >= 2 && topScores[0] - topScores[1] <= 0.008) {
      signals.push(`score_cluster=true (gap=${(topScores[0] - topScores[1]).toFixed(4)})`);
    }
  }
  if (signals.length > 0) {
    lines.push(`Heuristic signals: ${signals.join(", ")}`);
  }

  return lines.join("\n");
}

function buildUserPrompt(voice: Voice, jobs: TrainingJob[]): string {
  const voiceJobs = [...jobs]
    .filter((job) => job.voice_id === voice.voice_id)
    .sort((a, b) => (Number(b.created_at) || 0) - (Number(a.created_at) || 0));

  const activeJobs = voiceJobs.filter((job) => isAdviceActiveJob(job));
  const completedJobs = voiceJobs.filter((job) => !isAdviceActiveJob(job));

  const language = voice.labels?.language ?? "ko";
  const lines: string[] = [
    `## Voice`,
    `Name: ${voice.name}, Model: ${voice.model_size}, Language: ${language}`,
    voice.checkpoint_score !== null
      ? `Current champion score: ${voice.checkpoint_score.toFixed(3)}`
      : "No current champion checkpoint",
    "",
  ];

  if (activeJobs.length > 0) {
    lines.push(`## Active Runs (${activeJobs.length})`);
    activeJobs.forEach((job, i) => {
      lines.push(`[Active ${i + 1}] ${summarizeJob(job)}`);
    });
    lines.push("");
  }

  const historyLimit = 20;
  const historyJobs = completedJobs.slice(0, historyLimit);
  if (historyJobs.length > 0) {
    lines.push(`## Completed History (${historyJobs.length} of ${completedJobs.length}, most recent first)`);
    historyJobs.forEach((job, i) => {
      lines.push(`[${i + 1}] ${summarizeJob(job)}`);
    });
    lines.push("");
  }

  if (completedJobs.length > 0) {
    lines.push(buildAnalytics(voice, completedJobs));
    lines.push("");
  }

  lines.push("Recommend one new training configuration.");
  return lines.join("\n");
}

function parseAdvisorResponse(raw: unknown): TrainingAdvice | null {
  if (!raw || typeof raw !== "object") return null;
  const data = raw as Record<string, unknown>;

  const mode = typeof data.mode === "string" && VALID_MODES.has(data.mode)
    ? (data.mode as TrainingAdviceMode)
    : null;
  if (!mode) return null;

  const title = typeof data.title === "string" ? data.title.trim() : "";
  const summary = typeof data.summary === "string" ? data.summary.trim() : "";
  if (!title || !summary) return null;

  const confidence =
    data.confidence === "high" || data.confidence === "medium" ? data.confidence : "medium";

  const reasons = Array.isArray(data.reasons)
    ? data.reasons.flatMap((r) => (typeof r === "string" && r.trim() ? [r.trim()] : []))
    : [];

  const rawConfig = data.suggestedConfig;
  if (!rawConfig || typeof rawConfig !== "object") return null;
  const cfg = rawConfig as Record<string, unknown>;
  const suggestedConfig: TrainingConfig = {
    model_size: typeof cfg.model_size === "string" ? cfg.model_size : undefined,
    batch_size: typeof cfg.batch_size === "number" ? cfg.batch_size : undefined,
    num_epochs: typeof cfg.num_epochs === "number" ? cfg.num_epochs : undefined,
    learning_rate: typeof cfg.learning_rate === "number" ? cfg.learning_rate : undefined,
    gradient_accumulation_steps:
      typeof cfg.gradient_accumulation_steps === "number" ? cfg.gradient_accumulation_steps : undefined,
    subtalker_loss_weight:
      typeof cfg.subtalker_loss_weight === "number" ? cfg.subtalker_loss_weight : undefined,
    save_every_n_epochs:
      typeof cfg.save_every_n_epochs === "number" ? cfg.save_every_n_epochs : undefined,
    seed: typeof cfg.seed === "number" ? cfg.seed : undefined,
    whisper_language: typeof cfg.whisper_language === "string" ? cfg.whisper_language : undefined,
    gpu_type_id: typeof cfg.gpu_type_id === "string" ? cfg.gpu_type_id : undefined,
  };

  const compareFirst = data.compareFirst === true;
  const reviewDatasetFirst = data.reviewDatasetFirst === true;
  const primaryActionLabel =
    typeof data.primaryActionLabel === "string" && data.primaryActionLabel.trim()
      ? data.primaryActionLabel.trim()
      : undefined;

  return {
    mode,
    title,
    summary,
    confidence,
    reasons,
    suggestedConfig,
    compareFirst,
    reviewDatasetFirst,
    primaryActionLabel,
    analysisProvider: "llm",
  };
}

export async function buildLLMTrainingAdvice(
  env: Pick<Env, "OPENAI_API_KEY" | "OPENAI_ADVISOR_MODEL">,
  voice: Voice,
  jobs: TrainingJob[],
): Promise<TrainingAdvice | null> {
  const apiKey = String(env.OPENAI_API_KEY ?? "").trim();
  if (!apiKey) return null;

  const userPrompt = buildUserPrompt(voice, jobs);

  const response = await fetch(OPENAI_API_URL, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model:
        String(env.OPENAI_ADVISOR_MODEL ?? DEFAULT_ADVISOR_MODEL).trim() ||
        DEFAULT_ADVISOR_MODEL,
      messages: [
        { role: "system", content: SYSTEM_PROMPT },
        { role: "user", content: userPrompt },
      ],
      temperature: 0.4,
      response_format: { type: "json_object" },
    }),
  });

  if (!response.ok) {
    const detail = (await response.text()).slice(0, 400);
    console.warn(`LLM advisor failed (${response.status}): ${detail}`);
    return null;
  }

  const payload = (await response.json()) as Record<string, unknown>;
  const choices = Array.isArray(payload.choices) ? payload.choices : [];
  const message =
    choices[0] && typeof choices[0] === "object"
      ? (choices[0] as Record<string, unknown>).message
      : null;
  const content =
    message && typeof message === "object" && typeof (message as Record<string, unknown>).content === "string"
      ? ((message as Record<string, unknown>).content as string)
      : "";

  if (!content) {
    console.warn("LLM advisor returned empty content");
    return null;
  }

  try {
    const parsed = JSON.parse(content.trim().replace(/^```(?:json)?\s*/i, "").replace(/\s*```$/, ""));
    const advice = parseAdvisorResponse(parsed);
    if (!advice) return null;

    // Post-process: sanitize config for model/GPU/language consistency
    if (advice.suggestedConfig) {
      advice.suggestedConfig = sanitizeConfig(
        advice.suggestedConfig,
        voice.model_size,
        voice.labels?.language,
      );
    }

    // Post-process: deduplicate against active runs
    const activeVoiceJobs = [...jobs]
      .filter((job) => job.voice_id === voice.voice_id && isAdviceActiveJob(job));
    return diversifyAdviceForActiveRuns(advice, voice, activeVoiceJobs);
  } catch (error) {
    console.warn("LLM advisor response was not valid JSON:", content.slice(0, 200));
    return null;
  }
}
