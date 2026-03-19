import type { CalibrationResult, ArenaCalibrationConfidence, CalibrationState } from "../types";
import {
  VALIDATION_RANKING_WEIGHTS,
  VALIDATION_GATE_THRESHOLDS,
  type CheckpointScores,
  type ValidationWeights,
  type ValidationThresholds,
  readNumber,
  computeCalibrationAlpha,
  getEffectiveWeights,
} from "./training-domain";

const FEATURE_KEYS = ["asr", "speaker", "style", "tone", "speed", "overall", "duration"] as const;

// ── Helpers ────────────────────────────────────────────────────────────────────

function getScoreValue(scores: Record<string, number | null>, key: string): number {
  const mapping: Record<string, string> = {
    asr: "asr_score",
    speaker: "speaker_score",
    style: "style_score",
    tone: "tone_score",
    speed: "speed_score",
    overall: "overall_score",
    duration: "duration_score",
  };
  const val = scores[mapping[key] ?? key];
  return typeof val === "number" && Number.isFinite(val) ? val : 0;
}

function priorFromWeights(): number[] {
  return FEATURE_KEYS.map(k => VALIDATION_RANKING_WEIGHTS[k as keyof ValidationWeights] ?? 0);
}

// ── Logistic regression (unchanged core) ──────────────────────────────────────

export function fitWeights(
  deltas: number[][],
  outcomes: number[],
  sampleWeights: number[],
  priorWeights: number[],
  lambda: number,
  iterations = 500,
  lr = 0.01,
): number[] {
  const n = deltas.length;
  const d = priorWeights.length;
  if (n === 0) return [...priorWeights];

  const w = [...priorWeights];

  for (let iter = 0; iter < iterations; iter++) {
    const grad = new Array<number>(d).fill(0);
    for (let i = 0; i < n; i++) {
      const dot = deltas[i].reduce((s, v, j) => s + w[j] * v, 0);
      const p = 1 / (1 + Math.exp(-dot));
      const err = (p - outcomes[i]) * sampleWeights[i];
      for (let j = 0; j < d; j++) {
        grad[j] += err * deltas[i][j] + lambda * (w[j] - priorWeights[j]);
      }
    }
    for (let j = 0; j < d; j++) {
      w[j] -= lr * grad[j] / n;
    }
  }

  const sum = w.reduce((s, v) => s + Math.abs(v), 0);
  return sum > 0 ? w.map(v => Math.abs(v) / sum) : [...priorWeights];
}

// ── Lambda + confidence schedule (lowered minimums) ───────────────────────────

function determineLambda(effectiveCount: number): { lambda: number; confidence: ArenaCalibrationConfidence | "insufficient" } {
  if (effectiveCount < 10) return { lambda: 1, confidence: "insufficient" };
  if (effectiveCount < 20) return { lambda: 0.8, confidence: "preliminary" };
  if (effectiveCount < 40) return { lambda: 0.5, confidence: "preliminary" };
  if (effectiveCount < 80) return { lambda: 0.2, confidence: "calibrated" };
  return { lambda: 0.05, confidence: "high" };
}

// ── Ranking pseudo-pair extraction ────────────────────────────────────────────

const PSEUDO_PAIR_WEIGHT = 0.25;

interface RankingEntry {
  candidate_id: string;
  rank: number;
}

/**
 * Extract adjacent-rank pseudo-pairs from completed arena sessions.
 * Only adjacent ranks (1>2, 2>3, 3>4, 4>5) — NOT all combinations.
 */
async function extractRankingPseudoPairs(
  db: D1Database,
  voiceId: string | undefined,
  scoreMap: Map<string, Record<string, number | null>>,
): Promise<{ deltas: number[][]; outcomes: number[]; weights: number[] }> {
  const deltas: number[][] = [];
  const outcomes: number[] = [];
  const weights: number[] = [];

  const rankQuery = voiceId
    ? `SELECT s.session_id, s.ranking_json
       FROM arena_sessions s
       WHERE s.voice_id = ? AND s.status = 'completed' AND s.ranking_json IS NOT NULL AND s.ranking_json != '{}'`
    : `SELECT s.session_id, s.ranking_json
       FROM arena_sessions s
       WHERE s.status = 'completed' AND s.ranking_json IS NOT NULL AND s.ranking_json != '{}'`;

  const rankResult = voiceId
    ? await db.prepare(rankQuery).bind(voiceId).all<{ session_id: string; ranking_json: string }>()
    : await db.prepare(rankQuery).all<{ session_id: string; ranking_json: string }>();

  for (const row of rankResult.results ?? []) {
    let ranking: RankingEntry[] = [];
    try {
      ranking = JSON.parse(row.ranking_json);
    } catch {
      continue;
    }
    if (!Array.isArray(ranking) || ranking.length < 2) continue;

    ranking.sort((a, b) => a.rank - b.rank);

    for (let i = 0; i < ranking.length - 1; i++) {
      const winnerId = ranking[i].candidate_id;
      const loserId = ranking[i + 1].candidate_id;
      const scoresW = scoreMap.get(winnerId);
      const scoresL = scoreMap.get(loserId);
      if (!scoresW || !scoresL) continue;

      const delta = FEATURE_KEYS.map(k => getScoreValue(scoresW, k) - getScoreValue(scoresL, k));
      const allZero = delta.every(v => Math.abs(v) < 1e-8);
      if (allZero) continue;

      deltas.push(delta);
      outcomes.push(1);
      weights.push(PSEUDO_PAIR_WEIGHT);
    }
  }

  return { deltas, outcomes, weights };
}

// ── State machine logic ───────────────────────────────────────────────────────

/**
 * Determine the next calibration state based on matchup count and current state.
 *
 * Progression: shadow -> canary -> active
 * - shadow: <20 effective matches. Compute calibration but don't use it.
 * - canary: 20-39 effective matches. Only affects gray-zone rescues.
 * - active: 40+ effective matches AND accuracy >= 0.60. Affects ranking weights too.
 * - rolled_back: auto-disabled due to poor performance.
 */
function determineCalibrationState(
  effectiveCount: number,
  accuracy: number,
  currentState: CalibrationState | null,
): CalibrationState {
  if (currentState === "rolled_back") return "rolled_back";

  if (effectiveCount < 20) return "shadow";
  if (effectiveCount < 40) return "canary";
  if (accuracy >= 0.60) return "active";
  return "canary";
}

/**
 * Check if a calibration should be rolled back.
 * Triggers:
 * - Accuracy dropped below 0.50 (worse than coin flip)
 * - both_bad rate exceeds 0.35 (calibration letting garbage through)
 */
export function shouldRollback(
  accuracy: number,
  bothBadRate: number,
  currentState: CalibrationState,
): { rollback: boolean; reason: string | null } {
  if (currentState === "shadow") return { rollback: false, reason: null };
  if (currentState === "rolled_back") return { rollback: false, reason: null };

  if (accuracy < 0.50) {
    return { rollback: true, reason: `Accuracy ${accuracy.toFixed(3)} dropped below 0.50 (coin flip)` };
  }
  if (bothBadRate > 0.35) {
    return { rollback: true, reason: `both_bad rate ${bothBadRate.toFixed(3)} exceeds 0.35 safety limit` };
  }
  return { rollback: false, reason: null };
}

// ── Main calibration function ─────────────────────────────────────────────────

export async function calibrateFromArenaData(
  db: D1Database,
  voiceId?: string,
): Promise<CalibrationResult> {
  const prior = priorFromWeights();
  const defaultWeightsObj = Object.fromEntries(FEATURE_KEYS.map((k, i) => [k, prior[i]]));

  const emptyResult: CalibrationResult = {
    learned_weights: defaultWeightsObj,
    effective_weights: defaultWeightsObj,
    confidence: "insufficient",
    state: "shadow",
    matchup_count: 0,
    effective_matchup_count: 0,
    accuracy: 0,
    alpha: 0,
    ranking_pseudo_pairs_count: 0,
    weight_shifts: Object.fromEntries(FEATURE_KEYS.map(k => [k, 0])),
    gate_diagnostics: { both_bad_rate: 0, gate_pass_loss_rate: 0, rescued_checkpoint_loss_rate: 0, suggested_gate_changes: [] },
  };

  // ── Fetch direct pairwise votes ───────────────────────────────────────────

  const matchQuery = voiceId
    ? `SELECT m.candidate_a_id, m.candidate_b_id, m.winner, m.confidence
       FROM arena_matches m
       JOIN arena_sessions s ON s.session_id = m.session_id
       WHERE s.voice_id = ? AND m.winner IS NOT NULL`
    : `SELECT m.candidate_a_id, m.candidate_b_id, m.winner, m.confidence
       FROM arena_matches m
       WHERE m.winner IS NOT NULL`;

  const matchResult = voiceId
    ? await db.prepare(matchQuery).bind(voiceId).all<{ candidate_a_id: string; candidate_b_id: string; winner: string; confidence: string | null }>()
    : await db.prepare(matchQuery).all<{ candidate_a_id: string; candidate_b_id: string; winner: string; confidence: string | null }>();

  const allMatches = matchResult.results ?? [];
  const usable = allMatches.filter((m: { winner: string }) => m.winner === "a" || m.winner === "b");

  // ── Fetch candidate scores ────────────────────────────────────────────────

  const candidateIds = new Set<string>();
  for (const m of allMatches) {
    candidateIds.add(m.candidate_a_id);
    candidateIds.add(m.candidate_b_id);
  }

  const scoreMap = new Map<string, Record<string, number | null>>();
  if (candidateIds.size > 0) {
    const ids = [...candidateIds];
    const placeholders = ids.map(() => "?").join(", ");
    const scoreRows = await db
      .prepare(`SELECT candidate_id, auto_scores_json FROM arena_candidates WHERE candidate_id IN (${placeholders})`)
      .bind(...ids)
      .all<{ candidate_id: string; auto_scores_json: string | null }>();
    for (const row of scoreRows.results ?? []) {
      try {
        scoreMap.set(row.candidate_id, JSON.parse(row.auto_scores_json ?? "{}"));
      } catch {
        scoreMap.set(row.candidate_id, {});
      }
    }
  }

  // ── Build training data: direct votes ─────────────────────────────────────

  const deltas: number[][] = [];
  const outcomes: number[] = [];
  const weights: number[] = [];

  for (const m of usable) {
    const scoresA = scoreMap.get(m.candidate_a_id) ?? {};
    const scoresB = scoreMap.get(m.candidate_b_id) ?? {};
    const delta = FEATURE_KEYS.map(k => getScoreValue(scoresA, k) - getScoreValue(scoresB, k));

    const allZero = delta.every(v => Math.abs(v) < 1e-8);
    if (allZero) continue;

    deltas.push(delta);
    outcomes.push(m.winner === "a" ? 1 : 0);
    weights.push(m.confidence === "clear" ? 1.0 : 0.5);
  }

  // ── Add ranking pseudo-pairs ──────────────────────────────────────────────

  const pseudoPairs = await extractRankingPseudoPairs(db, voiceId, scoreMap);
  const directVoteCount = deltas.length;
  deltas.push(...pseudoPairs.deltas);
  outcomes.push(...pseudoPairs.outcomes);
  weights.push(...pseudoPairs.weights);

  const effectiveMatchupCount = directVoteCount + Math.round(pseudoPairs.deltas.length * PSEUDO_PAIR_WEIGHT);

  // ── Check if enough data ──────────────────────────────────────────────────

  const { lambda, confidence } = determineLambda(effectiveMatchupCount);

  if (confidence === "insufficient") {
    emptyResult.matchup_count = allMatches.length;
    emptyResult.effective_matchup_count = effectiveMatchupCount;
    emptyResult.ranking_pseudo_pairs_count = pseudoPairs.deltas.length;
    return emptyResult;
  }

  // ── Fit model ─────────────────────────────────────────────────────────────

  if (deltas.length < 5) {
    emptyResult.matchup_count = allMatches.length;
    emptyResult.effective_matchup_count = effectiveMatchupCount;
    emptyResult.ranking_pseudo_pairs_count = pseudoPairs.deltas.length;
    return emptyResult;
  }

  const learned = fitWeights(deltas, outcomes, weights, prior, lambda);

  // ── Compute accuracy ──────────────────────────────────────────────────────

  let correct = 0;
  for (let i = 0; i < deltas.length; i++) {
    const dot = deltas[i].reduce((s, v, j) => s + learned[j] * v, 0);
    if ((dot > 0 && outcomes[i] === 1) || (dot < 0 && outcomes[i] === 0)) correct++;
  }
  const accuracy = correct / deltas.length;

  // ── Compute weight shifts ─────────────────────────────────────────────────

  const weightShifts: Record<string, number> = {};
  for (let i = 0; i < FEATURE_KEYS.length; i++) {
    weightShifts[FEATURE_KEYS[i]] = Math.round((learned[i] - prior[i]) * 1000) / 1000;
  }

  // ── Compute alpha and effective weights ───────────────────────────────────

  const alpha = computeCalibrationAlpha(effectiveMatchupCount);
  const learnedObj = Object.fromEntries(FEATURE_KEYS.map((k, i) => [k, Math.round(learned[i] * 1000) / 1000]));
  const effectiveWeightsResult = getEffectiveWeights(VALIDATION_RANKING_WEIGHTS, learnedObj, alpha);
  const effectiveWeightsRounded = Object.fromEntries(
    Object.entries(effectiveWeightsResult).map(([k, v]) => [k, Math.round((v as number) * 1000) / 1000])
  );

  // ── Determine calibration state ───────────────────────────────────────────

  let currentState: CalibrationState | null = null;
  if (voiceId) {
    const existing = await db
      .prepare("SELECT state FROM arena_calibration_overrides WHERE voice_id = ? LIMIT 1")
      .bind(voiceId)
      .first<{ state: string | null }>();
    currentState = (existing?.state as CalibrationState) ?? null;
  }

  let state = determineCalibrationState(effectiveMatchupCount, accuracy, currentState);

  // ── Gate diagnostics ──────────────────────────────────────────────────────

  const bothBadCount = allMatches.filter((m: { winner: string }) => m.winner === "both_bad").length;
  const bothBadRate = allMatches.length > 0 ? bothBadCount / allMatches.length : 0;

  const rollbackCheck = shouldRollback(accuracy, bothBadRate, state);
  if (rollbackCheck.rollback) {
    state = "rolled_back";
  }

  let gatePassLosses = 0;
  let gatePassTotal = 0;
  for (const m of usable) {
    const winnerId = m.winner === "a" ? m.candidate_a_id : m.candidate_b_id;
    const loserId = m.winner === "a" ? m.candidate_b_id : m.candidate_a_id;
    const winnerScores = scoreMap.get(winnerId);
    const loserScores = scoreMap.get(loserId);
    if (winnerScores && loserScores) {
      const winnerRank = FEATURE_KEYS.reduce((s, k, i) => s + getScoreValue(winnerScores, k) * prior[i], 0);
      const loserRank = FEATURE_KEYS.reduce((s, k, i) => s + getScoreValue(loserScores, k) * prior[i], 0);
      if (loserRank > winnerRank) {
        gatePassLosses++;
      }
      gatePassTotal++;
    }
  }
  const gatePassLossRate = gatePassTotal > 0 ? gatePassLosses / gatePassTotal : 0;

  const gateDiagnostics = computeGateDiagnostics(bothBadRate, gatePassLossRate);

  return {
    learned_weights: learnedObj,
    effective_weights: effectiveWeightsRounded,
    confidence,
    state,
    matchup_count: allMatches.length,
    effective_matchup_count: effectiveMatchupCount,
    accuracy: Math.round(accuracy * 1000) / 1000,
    alpha,
    ranking_pseudo_pairs_count: pseudoPairs.deltas.length,
    weight_shifts: weightShifts,
    gate_diagnostics: gateDiagnostics,
  };
}

// ── Gate diagnostics ──────────────────────────────────────────────────────────

function computeGateDiagnostics(
  bothBadRate: number,
  gatePassLossRate: number,
): CalibrationResult["gate_diagnostics"] {
  const suggestions: CalibrationResult["gate_diagnostics"]["suggested_gate_changes"] = [];

  if (bothBadRate > 0.25) {
    suggestions.push(
      { metric: "overall", direction: "tighten", current: VALIDATION_GATE_THRESHOLDS.overall_min, suggested: Math.min(0.90, VALIDATION_GATE_THRESHOLDS.overall_min + 0.05) },
      { metric: "health", direction: "tighten", current: VALIDATION_GATE_THRESHOLDS.health_min, suggested: Math.min(0.85, VALIDATION_GATE_THRESHOLDS.health_min + 0.05) },
    );
  }

  if (gatePassLossRate > 0.40) {
    suggestions.push(
      { metric: "tone", direction: "loosen", current: VALIDATION_GATE_THRESHOLDS.tone_min, suggested: Math.max(0.40, VALIDATION_GATE_THRESHOLDS.tone_min - 0.05) },
      { metric: "style", direction: "loosen", current: VALIDATION_GATE_THRESHOLDS.style_min, suggested: Math.max(0.45, VALIDATION_GATE_THRESHOLDS.style_min - 0.05) },
    );
  }

  return {
    both_bad_rate: Math.round(bothBadRate * 1000) / 1000,
    gate_pass_loss_rate: Math.round(gatePassLossRate * 1000) / 1000,
    rescued_checkpoint_loss_rate: 0,
    suggested_gate_changes: suggestions,
  };
}

// ── Publish calibration to DB ─────────────────────────────────────────────────

export async function publishCalibration(
  db: D1Database,
  voiceId: string,
  result: CalibrationResult,
): Promise<void> {
  if (result.confidence === "insufficient") return;

  const now = Date.now();

  const existing = await db
    .prepare("SELECT override_id, version FROM arena_calibration_overrides WHERE voice_id = ? LIMIT 1")
    .bind(voiceId)
    .first<{ override_id: string; version: number }>();

  const overrideId = existing?.override_id ?? crypto.randomUUID();
  const version = (existing?.version ?? 0) + 1;

  const rollbackReason = result.state === "rolled_back"
    ? `Auto-rollback at version ${version}: accuracy=${result.accuracy}, both_bad=${result.gate_diagnostics.both_bad_rate}`
    : null;

  if (existing) {
    await db.prepare(
      `UPDATE arena_calibration_overrides SET
        weights_json = ?, effective_weights_json = ?, matchup_count = ?,
        effective_matchup_count = ?, ranking_pseudo_pairs_count = ?,
        accuracy = ?, confidence = ?, state = ?, version = ?, alpha = ?,
        weight_shifts_json = ?, gate_diagnostics_json = ?,
        rollback_reason = ?, shadow_accuracy = ?, updated_at = ?
       WHERE override_id = ?`
    ).bind(
      JSON.stringify(result.learned_weights),
      JSON.stringify(result.effective_weights),
      result.matchup_count,
      result.effective_matchup_count,
      result.ranking_pseudo_pairs_count,
      result.accuracy,
      result.confidence,
      result.state,
      version,
      result.alpha,
      JSON.stringify(result.weight_shifts),
      JSON.stringify(result.gate_diagnostics),
      rollbackReason,
      result.state === "shadow" ? result.accuracy : null,
      now,
      overrideId,
    ).run();
  } else {
    await db.prepare(
      `INSERT INTO arena_calibration_overrides (
        override_id, voice_id, weights_json, effective_weights_json,
        matchup_count, effective_matchup_count, ranking_pseudo_pairs_count,
        accuracy, confidence, state, version, alpha,
        weight_shifts_json, gate_diagnostics_json,
        rollback_reason, shadow_accuracy, created_at, updated_at
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
    ).bind(
      overrideId, voiceId,
      JSON.stringify(result.learned_weights),
      JSON.stringify(result.effective_weights),
      result.matchup_count,
      result.effective_matchup_count,
      result.ranking_pseudo_pairs_count,
      result.accuracy,
      result.confidence,
      result.state,
      version,
      result.alpha,
      JSON.stringify(result.weight_shifts),
      JSON.stringify(result.gate_diagnostics),
      rollbackReason,
      result.state === "shadow" ? result.accuracy : null,
      now, now,
    ).run();
  }
}

// ── Load effective weights from DB ────────────────────────────────────────────

/**
 * Load calibrated effective weights for a voice from the DB.
 * Returns default weights if no calibration exists or calibration is in shadow/rolled_back state.
 *
 * canary state: returns effective weights (used only for gray-zone rescue by caller)
 * active state: returns effective weights (used for ranking + gray-zone rescue)
 */
export async function loadEffectiveWeights(
  db: D1Database,
  voiceId: string,
): Promise<{ weights: ValidationWeights; state: CalibrationState; alpha: number }> {
  const defaults = { weights: { ...VALIDATION_RANKING_WEIGHTS }, state: "shadow" as CalibrationState, alpha: 0 };

  const row = await db
    .prepare("SELECT effective_weights_json, state, alpha FROM arena_calibration_overrides WHERE voice_id = ? LIMIT 1")
    .bind(voiceId)
    .first<{ effective_weights_json: string | null; state: string; alpha: number }>();

  const needsGlobalFallback = !row || !row.effective_weights_json || row.state === "shadow" || row.state === "rolled_back";

  if (needsGlobalFallback) {
    const global = await db
      .prepare("SELECT effective_weights_json, state, alpha FROM arena_calibration_overrides WHERE voice_id = '__global__' LIMIT 1")
      .first<{ effective_weights_json: string | null; state: string; alpha: number }>();
    if (global?.effective_weights_json && global.state !== "shadow" && global.state !== "rolled_back") {
      try {
        return { weights: JSON.parse(global.effective_weights_json) as ValidationWeights, state: global.state as CalibrationState, alpha: global.alpha ?? 0 };
      } catch { /* fall through to defaults */ }
    }
    return { ...defaults, state: (row?.state as CalibrationState) ?? "shadow" };
  }

  try {
    return { weights: JSON.parse(row!.effective_weights_json!) as ValidationWeights, state: row!.state as CalibrationState, alpha: row!.alpha ?? 0 };
  } catch { return defaults; }
}
