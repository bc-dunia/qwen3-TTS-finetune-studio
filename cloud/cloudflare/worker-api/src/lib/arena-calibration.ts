import type { CalibrationResult, ArenaCalibrationConfidence } from "../types";
import {
  VALIDATION_RANKING_WEIGHTS,
  VALIDATION_GATE_THRESHOLDS,
  type CheckpointScores,
  type ValidationWeights,
  type ValidationThresholds,
  readNumber,
} from "./training-domain";

const FEATURE_KEYS = ["asr", "speaker", "style", "tone", "speed", "overall", "duration"] as const;

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

function determineLambda(matchupCount: number): { lambda: number; confidence: ArenaCalibrationConfidence | "insufficient" } {
  if (matchupCount < 30) return { lambda: 1, confidence: "insufficient" };
  if (matchupCount < 50) return { lambda: 0.5, confidence: "preliminary" };
  if (matchupCount < 100) return { lambda: 0.1, confidence: "calibrated" };
  return { lambda: 0.01, confidence: "high" };
}

export async function calibrateFromArenaData(
  db: D1Database,
  voiceId?: string,
): Promise<CalibrationResult> {
  const prior = priorFromWeights();
  const emptyResult: CalibrationResult = {
    learned_weights: Object.fromEntries(FEATURE_KEYS.map((k, i) => [k, prior[i]])),
    confidence: "insufficient",
    matchup_count: 0,
    accuracy: 0,
    weight_shifts: Object.fromEntries(FEATURE_KEYS.map(k => [k, 0])),
    gate_diagnostics: { both_bad_rate: 0, gate_pass_loss_rate: 0, suggested_gate_changes: [] },
  };

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
  const { lambda, confidence } = determineLambda(usable.length);

  if (confidence === "insufficient") {
    emptyResult.matchup_count = allMatches.length;
    return emptyResult;
  }

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

  if (deltas.length < 10) {
    emptyResult.matchup_count = allMatches.length;
    return emptyResult;
  }

  const learned = fitWeights(deltas, outcomes, weights, prior, lambda);

  let correct = 0;
  for (let i = 0; i < deltas.length; i++) {
    const dot = deltas[i].reduce((s, v, j) => s + learned[j] * v, 0);
    if ((dot > 0 && outcomes[i] === 1) || (dot < 0 && outcomes[i] === 0)) correct++;
  }
  const accuracy = correct / deltas.length;

  const weightShifts: Record<string, number> = {};
  for (let i = 0; i < FEATURE_KEYS.length; i++) {
    weightShifts[FEATURE_KEYS[i]] = Math.round((learned[i] - prior[i]) * 1000) / 1000;
  }

  const bothBadCount = allMatches.filter((m: { winner: string }) => m.winner === "both_bad").length;
  const bothBadRate = allMatches.length > 0 ? bothBadCount / allMatches.length : 0;

  const gateDiagnostics = computeGateDiagnostics(bothBadRate);

  return {
    learned_weights: Object.fromEntries(FEATURE_KEYS.map((k, i) => [k, Math.round(learned[i] * 1000) / 1000])),
    confidence,
    matchup_count: allMatches.length,
    accuracy: Math.round(accuracy * 1000) / 1000,
    weight_shifts: weightShifts,
    gate_diagnostics: gateDiagnostics,
  };
}

function computeGateDiagnostics(bothBadRate: number): CalibrationResult["gate_diagnostics"] {
  const suggestions: CalibrationResult["gate_diagnostics"]["suggested_gate_changes"] = [];

  if (bothBadRate > 0.25) {
    suggestions.push(
      { metric: "overall", direction: "tighten", current: VALIDATION_GATE_THRESHOLDS.overall_min, suggested: Math.min(0.90, VALIDATION_GATE_THRESHOLDS.overall_min + 0.05) },
      { metric: "health", direction: "tighten", current: VALIDATION_GATE_THRESHOLDS.health_min, suggested: Math.min(0.85, VALIDATION_GATE_THRESHOLDS.health_min + 0.05) },
    );
  }

  return {
    both_bad_rate: Math.round(bothBadRate * 1000) / 1000,
    gate_pass_loss_rate: 0,
    suggested_gate_changes: suggestions,
  };
}
