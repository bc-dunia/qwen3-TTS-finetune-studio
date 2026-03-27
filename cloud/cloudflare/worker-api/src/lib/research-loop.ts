import type {
  Env, Voice, VoiceResearchState, VoiceResearchJournal,
  ResearchSnapshot, Retrospective, ResearchAction,
  ResearchTrigger, CalibrationSummary, StrategyBrief,
  StableLesson, ResearchBottleneck,
  TrainingJob,
} from "../types";
import { buildTrainingCheckoutSearch } from "./training-checkout";
import { loadEffectiveWeights, calibrateFromArenaData } from "./arena-calibration";
import { VALIDATION_RANKING_WEIGHTS, getTrainingDefaults } from "./training-domain";
import {
  appendVoiceResearchJournal,
  casUpsertVoiceResearchState,
  createTrainingCampaign,
  getVoice,
  getVoiceResearchState,
  listVoiceResearchJournal,
  updateJournalOutcome,
} from "./d1";
import { bootstrapArenaSession } from "./arena";

const OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses";
const DEFAULT_ADVISOR_MODEL = "gpt-5.4";
const PASSING_SCORE_THRESHOLD = 0.8;
const RESEARCH_COOLDOWN_MS = 5 * 60 * 1000;
const SCORING_CHANGE_MAX_SHIFT = 0.10;
const ALLOWED_SCORING_METRICS = new Set(["asr", "speaker", "style", "tone", "speed", "overall", "duration"]);

type DbTrainingSnapshotRow = {
  job_id: string;
  voice_id: string;
  campaign_id: string | null;
  attempt_index: number | null;
  round_id: string | null;
  dataset_snapshot_id: string | null;
  runpod_pod_id: string | null;
  job_token: string | null;
  status: string;
  config_json: string;
  progress_json: string | null;
  summary_json: string | null;
  metrics_json: string | null;
  supervisor_json: string | null;
  dataset_r2_prefix: string;
  log_r2_prefix: string | null;
  error_message: string | null;
  last_heartbeat_at: number | null;
  started_at: number | null;
  completed_at: number | null;
  created_at: number;
  updated_at: number;
};

type DbArenaSnapshotRow = {
  session_id: string;
  status: string;
  ranking_json: string | null;
  completed_at: number | null;
  winner_run_name: string | null;
};

type DbCalibrationSummaryRow = {
  alpha: number | null;
  state: string | null;
  weight_shifts_json: string | null;
  gate_diagnostics_json: string | null;
  matchup_count: number | null;
  accuracy: number | null;
};

const parseJson = <T>(value: string | null | undefined, fallback: T): T => {
  if (!value) {
    return fallback;
  }
  try {
    return JSON.parse(value) as T;
  } catch {
    return fallback;
  }
};

const asRecord = (value: unknown): Record<string, unknown> | null => {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return null;
  }
  return value as Record<string, unknown>;
};

const safeString = (value: unknown): string | null => {
  return typeof value === "string" && value.trim() ? value.trim() : null;
};

const safeNumber = (value: unknown): number | null => {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
};

const validBottleneck = (value: unknown): ResearchBottleneck => {
  const allowed = new Set<string>([
    "tone",
    "asr",
    "speed",
    "speaker",
    "style",
    "overall",
    "dataset_quality",
    "balanced",
  ]);
  if (typeof value === "string" && allowed.has(value)) {
    return value as ResearchBottleneck;
  }
  return null;
};

const validActionType = (value: unknown): ResearchAction["type"] | null => {
  const allowed = new Set<string>([
    "train",
    "start_arena",
    "request_dataset_review",
    "propose_scoring_change",
    "hold",
  ]);
  if (typeof value === "string" && allowed.has(value)) {
    return value as ResearchAction["type"];
  }
  return null;
};

const validHypothesisUpdate = (
  value: unknown,
): Retrospective["hypothesis_update"] => {
  if (value === "maintain" || value === "revise" || value === "discard" || value === "new") {
    return value;
  }
  return "maintain";
};

const validConfidence = (value: unknown): "low" | "medium" | "high" => {
  return value === "high" || value === "medium" ? value : "low";
};

const sanitizeScoringChangeParams = (params: Record<string, unknown>): Record<string, unknown> => {
  const sanitized: Record<string, unknown> = {};

  const adjustments = asRecord(params.weight_adjustments);
  if (adjustments) {
    const clampedAdjustments: Record<string, number> = {};
    for (const [metric, delta] of Object.entries(adjustments)) {
      if (!ALLOWED_SCORING_METRICS.has(metric)) continue;
      const numDelta = typeof delta === "number" && Number.isFinite(delta) ? delta : 0;
      clampedAdjustments[metric] = Math.max(-SCORING_CHANGE_MAX_SHIFT, Math.min(SCORING_CHANGE_MAX_SHIFT, numDelta));
    }
    if (Object.keys(clampedAdjustments).length > 0) {
      sanitized.weight_adjustments = clampedAdjustments;
    }
  }

  const shadowMetrics = Array.isArray(params.shadow_metrics)
    ? params.shadow_metrics.filter((m): m is string => typeof m === "string" && ALLOWED_SCORING_METRICS.has(m))
    : [];
  if (shadowMetrics.length > 0) {
    sanitized.shadow_metrics = shadowMetrics;
  }

  sanitized.advisory_only = true;
  return sanitized;
};

const toTrainingJob = (row: DbTrainingSnapshotRow): TrainingJob => ({
  job_id: row.job_id,
  voice_id: row.voice_id,
  campaign_id: row.campaign_id,
  attempt_index: row.attempt_index,
  round_id: row.round_id,
  dataset_snapshot_id: row.dataset_snapshot_id,
  runpod_pod_id: row.runpod_pod_id,
  job_token: row.job_token,
  status: row.status,
  config: parseJson<Record<string, unknown>>(row.config_json, {}),
  progress: parseJson<Record<string, unknown>>(row.progress_json, {}),
  summary: parseJson<Record<string, unknown>>(row.summary_json, {}),
  metrics: parseJson<Record<string, unknown>>(row.metrics_json, {}),
  supervisor: parseJson<Record<string, unknown>>(row.supervisor_json, {}),
  dataset_r2_prefix: row.dataset_r2_prefix,
  log_r2_prefix: row.log_r2_prefix,
  error_message: row.error_message,
  last_heartbeat_at: row.last_heartbeat_at,
  started_at: row.started_at,
  completed_at: row.completed_at,
  created_at: row.created_at,
  updated_at: row.updated_at,
});

const buildHeuristicRetrospective = (snapshot: ResearchSnapshot): Retrospective => {
  const scored = snapshot.recentJobs.filter((job) => typeof job.score === "number");
  const latestScore = scored.length > 0 ? (scored[0].score as number) : null;
  const olderScore = scored.length > 1 ? (scored[scored.length - 1].score as number) : null;
  const trend = latestScore !== null && olderScore !== null ? latestScore - olderScore : 0;
  const failures = snapshot.recentJobs
    .map((job) => (job.failure_reason ?? "").toLowerCase())
    .filter((v) => v.length > 0)
    .join(" ");

  let bottleneck: ResearchBottleneck = "overall";
  if (failures.includes("asr") || failures.includes("dataset")) bottleneck = "dataset_quality";
  else if (failures.includes("tone")) bottleneck = "tone";
  else if (failures.includes("speed")) bottleneck = "speed";
  else if (failures.includes("speaker")) bottleneck = "speaker";
  else if (failures.includes("style")) bottleneck = "style";
  else if (trend >= 0.015) bottleneck = "balanced";

  const observations = [
    `Recent jobs reviewed: ${snapshot.recentJobs.length} (scored: ${scored.length})`,
    latestScore !== null ? `Latest score: ${latestScore.toFixed(3)}` : "Latest score unavailable",
    olderScore !== null ? `Earlier score: ${olderScore.toFixed(3)}` : "Historical score unavailable",
    `Arena sessions reviewed: ${snapshot.arenaHistory.length}`,
  ];

  return {
    observations,
    hypothesis_update: trend < -0.01 ? "revise" : "maintain",
    new_hypothesis:
      trend < -0.01
        ? "Recent changes are regressing quality; prioritize stability on the detected bottleneck."
        : null,
    lesson: trend > 0.015 ? "Incremental improvements are compounding; keep controlled exploration." : null,
    bottleneck_diagnosis: bottleneck,
    confidence: scored.length >= 3 ? "medium" : "low",
  };
};

const parseRetrospective = (content: string): Retrospective | null => {
  try {
    const parsed = JSON.parse(content.trim().replace(/^```(?:json)?\s*/i, "").replace(/\s*```$/, ""));
    const record = asRecord(parsed);
    if (!record) return null;

    const observationsRaw = Array.isArray(record.observations)
      ? record.observations.filter((item): item is string => typeof item === "string" && item.trim().length > 0)
      : [];

    return {
      observations: observationsRaw,
      hypothesis_update: validHypothesisUpdate(record.hypothesis_update),
      new_hypothesis: safeString(record.new_hypothesis),
      lesson: safeString(record.lesson),
      bottleneck_diagnosis: validBottleneck(record.bottleneck_diagnosis),
      confidence: validConfidence(record.confidence),
    };
  } catch {
    return null;
  }
};

const parseAction = (content: string): ResearchAction | null => {
  try {
    const parsed = JSON.parse(content.trim().replace(/^```(?:json)?\s*/i, "").replace(/\s*```$/, ""));
    const record = asRecord(parsed);
    if (!record) return null;

    const type = validActionType(record.type);
    if (!type) return null;

    const params = asRecord(record.params) ?? {};
    const reasoning = safeString(record.reasoning) ?? "No reasoning provided.";

    return { type, params, reasoning };
  } catch {
    return null;
  }
};

const nowMs = (): number => Date.now();

export async function buildCalibrationSummary(
  db: D1Database,
  voiceId: string,
): Promise<CalibrationSummary | null> {
  const row = await db
    .prepare(
      `SELECT alpha, state, weight_shifts_json, gate_diagnostics_json, matchup_count, accuracy
       FROM arena_calibration_overrides
       WHERE voice_id = ?
       LIMIT 1`
    )
    .bind(voiceId)
    .first<DbCalibrationSummaryRow>();

  if (!row) {
    const calibrated = await calibrateFromArenaData(db, voiceId);
    if (calibrated.matchup_count === 0) {
      return null;
    }
    return {
      alpha: calibrated.alpha,
      state: calibrated.state,
      biggest_shifts: calibrated.weight_shifts,
      both_bad_rate: calibrated.gate_diagnostics.both_bad_rate,
      matchup_count: calibrated.matchup_count,
      accuracy: calibrated.accuracy,
    };
  }

  const shifts = parseJson<Record<string, number>>(row.weight_shifts_json, {});
  const gate = parseJson<Record<string, unknown>>(row.gate_diagnostics_json, {});

  return {
    alpha: row.alpha ?? 0,
    state: (row.state ?? "shadow") as CalibrationSummary["state"],
    biggest_shifts: shifts,
    both_bad_rate: safeNumber(gate.both_bad_rate) ?? 0,
    matchup_count: row.matchup_count ?? 0,
    accuracy: row.accuracy ?? 0,
  };
}

export async function buildResearchSnapshot(
  db: D1Database,
  voiceId: string,
): Promise<ResearchSnapshot> {
  const voice = await getVoice(db, voiceId);
  if (!voice) {
    throw new Error(`voice_not_found:${voiceId}`);
  }

  const state = await getVoiceResearchState(db, voiceId);
  const recentJournal = await listVoiceResearchJournal(db, voiceId, 10);
  const calibrationSummary = await buildCalibrationSummary(db, voiceId);

  const jobsResult = await db
    .prepare(
      `SELECT *
       FROM training_jobs
       WHERE voice_id = ? AND status IN ('completed', 'failed', 'cancelled')
       ORDER BY completed_at DESC, created_at DESC
       LIMIT 20`
    )
    .bind(voiceId)
    .all<DbTrainingSnapshotRow>();

  const recentJobs = (jobsResult.results ?? []).map((row) => {
    const job = toTrainingJob(row);
    const checkout = buildTrainingCheckoutSearch(job);
    const score =
      checkout.selected?.score ??
      checkout.manual_promoted?.score ??
      checkout.champion?.score ??
      null;
    return {
      job_id: row.job_id,
      status: row.status,
      config: job.config,
      score,
      failure_reason: checkout.message ?? row.error_message ?? null,
      created_at: row.created_at,
    };
  });

  const arenaResult = await db
    .prepare(
      `SELECT
         s.session_id,
         s.status,
         s.ranking_json,
         s.completed_at,
         (
           SELECT c.run_name
           FROM arena_candidates c
           WHERE c.candidate_id = s.winner_candidate_id
           LIMIT 1
         ) AS winner_run_name
       FROM arena_sessions s
       WHERE s.voice_id = ?
       ORDER BY s.created_at DESC
       LIMIT 5`
    )
    .bind(voiceId)
    .all<DbArenaSnapshotRow>();

  const arenaHistory = (arenaResult.results ?? []).map((row) => {
    const rankingRaw = parseJson<Array<Record<string, unknown>>>(row.ranking_json, []);
    const ranking = rankingRaw
      .map((item) => ({
        rank: safeNumber(item.rank) ?? 0,
        candidate_id: safeString(item.candidate_id) ?? "",
        wins: safeNumber(item.wins) ?? 0,
      }))
      .filter((item) => item.rank > 0 && item.candidate_id.length > 0);

    return {
      session_id: row.session_id,
      status: row.status,
      winner_run_name: row.winner_run_name,
      ranking,
      completed_at: row.completed_at,
    };
  });

  return {
    voice,
    state,
    recentJournal,
    calibrationSummary,
    recentJobs,
    arenaHistory,
    championScore: voice.checkpoint_score,
  };
}

export async function writeRetrospective(
  env: Pick<Env, "OPENAI_API_KEY" | "OPENAI_ADVISOR_MODEL">,
  snapshot: ResearchSnapshot,
  trigger?: ResearchTrigger,
  triggerContext?: Record<string, unknown>,
): Promise<Retrospective> {
  const apiKey = String(env.OPENAI_API_KEY ?? "").trim();
  if (!apiKey) {
    return buildHeuristicRetrospective(snapshot);
  }

  const systemPrompt = `You are a TTS research strategist analyzing training outcomes. Produce concise JSON retrospective output only.`;
  const jobLines = snapshot.recentJobs
    .map((job, index) => {
      const lr = safeNumber(job.config.learning_rate);
      const epochs = safeNumber(job.config.num_epochs);
      return `[${index + 1}] job=${job.job_id} score=${job.score ?? "n/a"} lr=${lr ?? "n/a"} epochs=${epochs ?? "n/a"} fail=${job.failure_reason ?? "none"}`;
    })
    .join("\n");
  const arenaLines = snapshot.arenaHistory
    .map((session, index) => {
      const top = session.ranking.slice(0, 3).map((row) => `r${row.rank}:${row.candidate_id}:${row.wins}`).join(" ");
      return `[${index + 1}] session=${session.session_id} status=${session.status} winner=${session.winner_run_name ?? "n/a"} top=${top || "none"}`;
    })
    .join("\n");
  const journalLines = snapshot.recentJournal
    .map((entry, index) => `[${index + 1}] trigger=${entry.trigger} decision=${entry.decision} outcome=${entry.outcome ?? "pending"}`)
    .join("\n");

  const triggerLine = trigger ? `Trigger: ${trigger}` : "Trigger: unknown";
  const linkedLine = triggerContext?.linked_ids
    ? `Linked context: ${JSON.stringify(triggerContext.linked_ids)}`
    : "";

  const userPrompt = [
    `Voice: ${snapshot.voice.voice_id} (${snapshot.voice.name})`,
    triggerLine,
    linkedLine,
    `Champion score: ${snapshot.championScore ?? "n/a"}`,
    `Calibration: ${snapshot.calibrationSummary ? JSON.stringify(snapshot.calibrationSummary) : "none"}`,
    "Recent jobs:",
    jobLines || "none",
    "Arena history:",
    arenaLines || "none",
    "Journal history:",
    journalLines || "none",
    "Return JSON with keys: observations (string[]), hypothesis_update, new_hypothesis, lesson, bottleneck_diagnosis, confidence.",
  ].filter(Boolean).join("\n");

  try {
    const response = await fetch(OPENAI_RESPONSES_URL, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: String(env.OPENAI_ADVISOR_MODEL ?? DEFAULT_ADVISOR_MODEL).trim() || DEFAULT_ADVISOR_MODEL,
        input: [
          { role: "system", content: systemPrompt },
          { role: "user", content: userPrompt },
        ],
        reasoning: { effort: "high" },
        text: { format: { type: "json_object" } },
      }),
    });

    if (!response.ok) {
      return buildHeuristicRetrospective(snapshot);
    }

    const payload = (await response.json()) as Record<string, unknown>;
    const output = typeof payload.output_text === "string" ? payload.output_text : "";
    if (!output) {
      return buildHeuristicRetrospective(snapshot);
    }

    const parsed = parseRetrospective(output);
    if (!parsed) {
      return buildHeuristicRetrospective(snapshot);
    }
    return parsed;
  } catch {
    return buildHeuristicRetrospective(snapshot);
  }
}

export async function decideNextAction(
  env: Pick<Env, "OPENAI_API_KEY" | "OPENAI_ADVISOR_MODEL">,
  snapshot: ResearchSnapshot,
  retrospective: Retrospective,
  trigger?: ResearchTrigger,
  triggerContext?: Record<string, unknown>,
): Promise<ResearchAction> {
  const apiKey = String(env.OPENAI_API_KEY ?? "").trim();

  if (apiKey) {
    const systemPrompt = [
      "You are a TTS research controller.",
      "Allowed actions: train, start_arena, request_dataset_review, propose_scoring_change, hold.",
      "Return JSON only with keys: type, params, reasoning.",
      "Never output any action outside the allowed set.",
      "Use request_dataset_review for strong ASR/dataset signals.",
      "Use propose_scoring_change ONLY for ranking weight adjustments (±0.10 per metric max) or shadow metric additions. Never suggest gate threshold changes.",
      "Use start_arena when multiple viable checkpoints exist and human preference comparison is needed.",
      "Use train when there is a clear bottleneck-driven plan.",
      "Use hold when uncertainty is high.",
    ].join("\n");

    const linkedLine = triggerContext?.linked_ids
      ? `Linked context: ${JSON.stringify(triggerContext.linked_ids)}`
      : "";

    const userPrompt = [
      `Trigger: ${trigger ?? "unknown"}`,
      linkedLine,
      `Autonomy mode: ${snapshot.state?.autonomy_mode ?? "supervised"}`,
      `Bottleneck: ${retrospective.bottleneck_diagnosis ?? "none"}`,
      `Retrospective confidence: ${retrospective.confidence}`,
      `Recent job scores: ${snapshot.recentJobs.map((j) => j.score ?? "n/a").join(", ") || "none"}`,
      `Recent arena statuses: ${snapshot.arenaHistory.map((s) => s.status).join(", ") || "none"}`,
      `Calibration summary: ${snapshot.calibrationSummary ? JSON.stringify(snapshot.calibrationSummary) : "none"}`,
    ].filter(Boolean).join("\n");

    try {
      const response = await fetch(OPENAI_RESPONSES_URL, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${apiKey}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model: String(env.OPENAI_ADVISOR_MODEL ?? DEFAULT_ADVISOR_MODEL).trim() || DEFAULT_ADVISOR_MODEL,
          input: [
            { role: "system", content: systemPrompt },
            { role: "user", content: userPrompt },
          ],
          reasoning: { effort: "high" },
          text: { format: { type: "json_object" } },
        }),
      });

      if (response.ok) {
        const payload = (await response.json()) as Record<string, unknown>;
        const output = typeof payload.output_text === "string" ? payload.output_text : "";
        const parsed = output ? parseAction(output) : null;
        if (parsed) {
          return parsed;
        }
      }
    } catch {
      void 0;
    }
  }

  const passingCount = snapshot.recentJobs.filter(
    (job) => typeof job.score === "number" && (job.score as number) >= PASSING_SCORE_THRESHOLD,
  ).length;
  const latestArena = snapshot.arenaHistory[0] ?? null;
  const arenaJustCompleted =
    latestArena?.status === "completed" &&
    typeof latestArena.completed_at === "number" &&
    nowMs() - latestArena.completed_at < 6 * 60 * 60 * 1000;
  const hasActiveOrRecentArena = snapshot.arenaHistory.some((s) =>
    (s.status !== "completed" && s.status !== "cancelled") ||
    (typeof s.completed_at === "number" && nowMs() - s.completed_at < 6 * 60 * 60 * 1000),
  );

  if (passingCount >= 2 && !hasActiveOrRecentArena) {
    return {
      type: "start_arena",
      params: { reason: "multiple_passing_checkpoints", passing_count: passingCount },
      reasoning: "Multiple passing checkpoints exist and no recent arena comparison is available.",
    };
  }

  if (arenaJustCompleted) {
    const brief: Record<string, unknown> = {
      bottleneck: retrospective.bottleneck_diagnosis,
      calibration_alpha: snapshot.calibrationSummary?.alpha ?? 0,
      ranking_weights: VALIDATION_RANKING_WEIGHTS,
    };
    return {
      type: "train",
      params: brief,
      reasoning: "Arena completed recently; launch bottleneck-aware training iteration.",
    };
  }

  return {
    type: "hold",
    params: { reason: "insufficient_signal" },
    reasoning: "Insufficient confidence to trigger a new autonomous action.",
  };
}

type ResearchLoopResult = {
  entry: VoiceResearchJournal;
  action: ResearchAction;
  executed: boolean;
  arena_session_id: string | null;
  campaign_id: string | null;
  state: VoiceResearchState;
};

const buildDefaultState = (voiceId: string, createdAt: number): VoiceResearchState => ({
  voice_id: voiceId,
  cycle_count: 0,
  current_bottleneck: null,
  active_hypothesis: null,
  stable_lessons: [],
  pending_action: null,
  pending_action_params: null,
  dataset_snapshot_id: null,
  calibration_summary: null,
  scoring_policy_version: 1,
  autonomy_mode: "supervised",
  last_retrospective: null,
  created_at: createdAt,
  updated_at: createdAt,
});

const nextHypothesis = (
  current: string | null,
  retrospective: Retrospective,
): string | null => {
  if (retrospective.hypothesis_update === "discard") return null;
  if (retrospective.hypothesis_update === "new" || retrospective.hypothesis_update === "revise") {
    return retrospective.new_hypothesis;
  }
  return current;
};

const addLesson = (lessons: StableLesson[], retrospective: Retrospective, now: number): StableLesson[] => {
  if (!retrospective.lesson) {
    return lessons;
  }
  const existing = lessons.find((item) => item.lesson === retrospective.lesson);
  if (existing) {
    return lessons.map((item) =>
      item.lesson === retrospective.lesson
        ? {
            ...item,
            evidence_count: item.evidence_count + 1,
            confidence: retrospective.confidence,
            expires_at: item.expires_at,
          }
        : item,
    );
  }
  return [
    {
      lesson: retrospective.lesson,
      confidence: retrospective.confidence,
      evidence_count: 1,
      created_at: now,
      expires_at: null,
    },
    ...lessons,
  ].slice(0, 25);
};

const getDefaultArenaTexts = (): string[] => [
  "Hello. This is a test sentence for checking voice quality.",
  "The quick brown fox jumps over the lazy dog while keeping natural rhythm.",
  "Next, we will check intonation, pronunciation, and speaking-rate consistency.",
];

const normalizeArenaTexts = (context?: Record<string, unknown>): string[] => {
  const raw = context?.test_texts;
  if (!Array.isArray(raw)) return getDefaultArenaTexts();
  const texts = raw.filter((item): item is string => typeof item === "string" && item.trim().length > 0);
  return texts.length > 0 ? texts : getDefaultArenaTexts();
};

const maybeCreateArenaSession = async (
  db: D1Database,
  voice: Voice,
  context?: Record<string, unknown>,
): Promise<string | null> => {
  const testTexts = normalizeArenaTexts(context);
  const result = await bootstrapArenaSession(db, {
    voiceId: voice.voice_id,
    testTexts,
    seed: 42,
    settings: voice.settings,
  });
  return result?.session.session_id ?? null;
};

const createResearchCampaign = async (
  db: D1Database,
  voice: Voice,
  actionParams: Record<string, unknown>,
): Promise<string | null> => {
  const modelSize = voice.model_size || "1.7B";
  const defaults = getTrainingDefaults(modelSize);
  const now = nowMs();
  const campaignId = crypto.randomUUID();

  const campaign: import("../types").TrainingCampaign = {
    campaign_id: campaignId,
    voice_id: voice.voice_id,
    dataset_name: null,
    dataset_r2_prefix: null,
    dataset_snapshot_id: null,
    attempt_count: 3,
    parallelism: 1,
    status: "planning",
    base_config: {
      model_size: modelSize,
      batch_size: defaults.batch_size,
      learning_rate: defaults.learning_rate,
      num_epochs: defaults.num_epochs,
      gradient_accumulation_steps: defaults.gradient_accumulation_steps,
      subtalker_loss_weight: defaults.subtalker_loss_weight,
      save_every_n_epochs: defaults.save_every_n_epochs,
      seed: defaults.seed,
      gpu_type_id: defaults.gpu_type_id,
      whisper_language: voice.labels?.language ?? "ko",
    },
    stop_rules: {
      max_asr_failures: 2,
      max_infra_failures: 2,
      min_score_improvement: 0.005,
      stagnation_window: 2,
    },
    planner_state: {
      direction: actionParams.bottleneck === "dataset_quality" ? "exploratory" : "balanced",
      source: "research_controller",
    },
    summary: {},
    created_at: now,
    updated_at: now,
    completed_at: null,
  };

  try {
    await createTrainingCampaign(db, campaign);
    return campaignId;
  } catch {
    return null;
  }
};

export async function maybeAdvanceResearchLoop(
  db: D1Database,
  env: Pick<Env, "OPENAI_API_KEY" | "OPENAI_ADVISOR_MODEL">,
  voiceId: string,
  trigger: ResearchTrigger,
  context?: Record<string, unknown>,
): Promise<ResearchLoopResult> {
  const now = nowMs();

  const existingState = await getVoiceResearchState(db, voiceId);
  if (existingState && (now - existingState.updated_at) < RESEARCH_COOLDOWN_MS) {
    const cooldownEntry: VoiceResearchJournal = {
      entry_id: crypto.randomUUID(),
      voice_id: voiceId,
      cycle_id: existingState.cycle_count,
      trigger,
      linked_ids: null,
      observations: "Cooldown active — skipped.",
      hypothesis: existingState.active_hypothesis,
      decision: "hold",
      decision_params: { reason: "cooldown", cooldown_remaining_ms: RESEARCH_COOLDOWN_MS - (now - existingState.updated_at) },
      expected_signal: null,
      outcome: "skipped_cooldown",
      confidence: "low",
      created_at: now,
    };
    await appendVoiceResearchJournal(db, cooldownEntry);
    return {
      entry: cooldownEntry,
      action: { type: "hold", params: { reason: "cooldown" }, reasoning: "Cooldown period active." },
      executed: false,
      arena_session_id: null,
      campaign_id: null,
      state: existingState,
    };
  }

  const snapshot = await buildResearchSnapshot(db, voiceId);
  const retrospective = await writeRetrospective(env, snapshot, trigger, context);
  const rawAction = await decideNextAction(env, snapshot, retrospective, trigger, context);

  const action: ResearchAction = rawAction.type === "propose_scoring_change"
    ? { ...rawAction, params: sanitizeScoringChangeParams(rawAction.params) }
    : rawAction;

  const stateBase = snapshot.state ?? buildDefaultState(voiceId, now);
  const expectedUpdatedAt = snapshot.state?.updated_at ?? null;
  const cycleId = stateBase.cycle_count + 1;

  const autonomyMode = stateBase.autonomy_mode;
  let pendingAction: VoiceResearchState["pending_action"] = action.type;
  let pendingParams: VoiceResearchState["pending_action_params"] = action.params;

  if (action.type === "hold") {
    pendingAction = null;
    pendingParams = null;
  }

  const claimState: VoiceResearchState = {
    ...stateBase,
    cycle_count: cycleId,
    current_bottleneck: retrospective.bottleneck_diagnosis,
    active_hypothesis: nextHypothesis(stateBase.active_hypothesis, retrospective),
    stable_lessons: addLesson(stateBase.stable_lessons, retrospective, now),
    pending_action: pendingAction,
    pending_action_params: pendingParams,
    calibration_summary: snapshot.calibrationSummary,
    last_retrospective: {
      observations: retrospective.observations,
      hypothesis_update: retrospective.hypothesis_update,
      new_hypothesis: retrospective.new_hypothesis,
      lesson: retrospective.lesson,
      bottleneck_diagnosis: retrospective.bottleneck_diagnosis,
      confidence: retrospective.confidence,
    },
    updated_at: now,
  };

  const claimed = await casUpsertVoiceResearchState(db, claimState, expectedUpdatedAt);

  const entry: VoiceResearchJournal = {
    entry_id: crypto.randomUUID(),
    voice_id: voiceId,
    cycle_id: cycleId,
    trigger,
    linked_ids: asRecord(context?.linked_ids) as VoiceResearchJournal["linked_ids"] | null,
    observations: retrospective.observations.join(" | ") || "No observations recorded.",
    hypothesis: retrospective.new_hypothesis ?? stateBase.active_hypothesis,
    decision: action.type,
    decision_params: action.params,
    expected_signal: action.reasoning,
    outcome: null,
    confidence: retrospective.confidence,
    created_at: now,
  };

  if (!claimed) {
    entry.outcome = "cas_conflict_aborted";
    await appendVoiceResearchJournal(db, entry);
    return {
      entry,
      action,
      executed: false,
      arena_session_id: null,
      campaign_id: null,
      state: (await getVoiceResearchState(db, voiceId)) ?? claimState,
    };
  }

  await appendVoiceResearchJournal(db, entry);

  let executed = false;
  let arenaSessionId: string | null = null;
  let campaignId: string | null = null;
  let outcomeMsg: string;

  if (action.type === "start_arena" && (autonomyMode === "semi_auto" || autonomyMode === "auto")) {
    arenaSessionId = await maybeCreateArenaSession(db, snapshot.voice, context);
    executed = arenaSessionId !== null;
    outcomeMsg = executed
      ? `arena_session_created:${arenaSessionId}`
      : "arena_insufficient_candidates";
  } else if (action.type === "train" && autonomyMode === "auto") {
    const hasActiveCampaign = await db
      .prepare("SELECT campaign_id FROM training_campaigns WHERE voice_id = ? AND status IN ('planning', 'running') LIMIT 1")
      .bind(voiceId)
      .first<{ campaign_id: string }>();
    if (hasActiveCampaign) {
      outcomeMsg = `train_skipped:active_campaign_exists:${hasActiveCampaign.campaign_id}`;
    } else {
      campaignId = await createResearchCampaign(db, snapshot.voice, action.params);
      executed = campaignId !== null;
      outcomeMsg = executed
        ? `campaign_created:${campaignId}`
        : "campaign_creation_failed";
    }
  } else if (action.type === "hold") {
    outcomeMsg = "held";
  } else {
    outcomeMsg = autonomyMode === "supervised"
      ? `pending_manual_review:${action.type}`
      : `pending:${action.type}`;
  }

  await updateJournalOutcome(db, entry.entry_id, outcomeMsg);
  entry.outcome = outcomeMsg;

  if (executed) {
    await casUpsertVoiceResearchState(db, {
      ...claimState,
      pending_action: null,
      pending_action_params: null,
      updated_at: nowMs(),
    }, claimState.updated_at);
  }

  return {
    entry,
    action,
    executed,
    arena_session_id: arenaSessionId,
    campaign_id: campaignId,
    state: claimState,
  };
}

export async function buildStrategyBrief(
  db: D1Database,
  voiceId: string,
): Promise<StrategyBrief> {
  const state = await getVoiceResearchState(db, voiceId);
  const calibration = await buildCalibrationSummary(db, voiceId);
  const recentJournal = await listVoiceResearchJournal(db, voiceId, 20);
  const { state: weightState } = await loadEffectiveWeights(db, voiceId);

  const lessons = (state?.stable_lessons ?? []).filter(
    (lesson) => lesson.expires_at === null || lesson.expires_at > nowMs(),
  );

  const flags = new Set<string>();
  const recentDecisions = recentJournal.slice(0, 8).map((entry) => entry.decision);
  if (recentDecisions.includes("request_dataset_review")) flags.add("dataset_review_recently_requested");
  if (recentDecisions.includes("propose_scoring_change")) flags.add("scoring_policy_under_review");
  if ((calibration?.both_bad_rate ?? 0) > 0.3) flags.add("high_both_bad_rate");
  if ((calibration?.accuracy ?? 0) > 0 && (calibration?.accuracy ?? 0) < 0.55) flags.add("low_calibration_accuracy");
  if (weightState === "rolled_back") flags.add("calibration_rolled_back");

  const latestCompletedArena = await db
    .prepare(
      `SELECT
         s.session_id,
         (
           SELECT c.run_name
           FROM arena_candidates c
           WHERE c.candidate_id = s.winner_candidate_id
           LIMIT 1
         ) AS winner_run_name
       FROM arena_sessions s
       WHERE s.voice_id = ? AND s.status = 'completed'
       ORDER BY s.completed_at DESC
       LIMIT 3`
    )
    .bind(voiceId)
    .all<{ session_id: string; winner_run_name: string | null }>();

  const winnerRuns = (latestCompletedArena.results ?? [])
    .map((row) => row.winner_run_name)
    .filter((value): value is string => typeof value === "string" && value.trim().length > 0);

  return {
    bottleneck: state?.current_bottleneck ?? null,
    calibration_insights: calibration,
    dataset_flags: [...flags],
    lessons,
    arena_winner_patterns: winnerRuns.length > 0 ? winnerRuns.join(" -> ") : null,
  };
}
