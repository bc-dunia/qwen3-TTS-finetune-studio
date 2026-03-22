import type {
  ArenaCandidate,
  ArenaMatch,
  ArenaSession,
  ArenaVoteConfidence,
  ArenaVoteWinner,
  ArenaCandidateSource,
  VoiceSettings,
} from "../types";
import {
  createArenaCandidate,
  createArenaMatch,
  createArenaSession,
} from "./d1";
import { computeRankingScore, passesValidationGate, passesHardSafetyGate, grayZoneRescue, type CheckpointScores, type ValidationWeights } from "./training-domain";
import { loadEffectiveWeights } from "./arena-calibration";

export interface AssembledCandidate {
  checkpoint_r2_prefix: string;
  job_id: string | null;
  run_name: string | null;
  epoch: number | null;
  source: ArenaCandidateSource;
  auto_scores: Record<string, number | null>;
  ranking_score: number;
}

export interface AssemblyResult {
  candidates: AssembledCandidate[];
  algorithm: "swiss" | "round_robin";
}

interface Standing {
  candidate_id: string;
  wins: number;
  losses: number;
  ties: number;
  bye_count: number;
  buchholz: number;
}

function readScore(rec: Record<string, unknown>, primary: string, fallback: string): number | null {
  if (typeof rec[primary] === "number") return rec[primary] as number;
  if (typeof rec[fallback] === "number") return rec[fallback] as number;
  return null;
}

function parseScoreFromMessage(msg: string, key: string): number | null {
  const match = new RegExp(`(?:^|\\s)${key}=([\\d.]+)`).exec(msg);
  if (!match) return null;
  const val = parseFloat(match[1]);
  return Number.isFinite(val) ? val : null;
}

export async function assembleArenaCandidates(
  db: D1Database,
  voiceId: string,
): Promise<AssemblyResult> {
  // Load calibrated weights for this voice (respects state machine)
  const { weights: effectiveWeights, state: calState } = await loadEffectiveWeights(db, voiceId);
  const useCalibrated = calState === "active";

  const voice = await db
    .prepare("SELECT checkpoint_r2_prefix, checkpoint_score, run_name, epoch, checkpoint_job_id FROM voices WHERE voice_id = ? LIMIT 1")
    .bind(voiceId)
    .first<{ checkpoint_r2_prefix: string | null; checkpoint_score: number | null; run_name: string | null; epoch: number | null; checkpoint_job_id: string | null }>();

  const carried = await db
    .prepare(
      `SELECT candidate_id, checkpoint_r2_prefix, job_id, run_name, epoch, retention_status, auto_scores_json
       FROM arena_candidates
       WHERE voice_id = ? AND retention_status IN ('champion', 'second', 'third')
       ORDER BY CASE retention_status WHEN 'champion' THEN 1 WHEN 'second' THEN 2 WHEN 'third' THEN 3 END`
    )
    .bind(voiceId)
    .all<{ candidate_id: string; checkpoint_r2_prefix: string; job_id: string | null; run_name: string | null; epoch: number | null; retention_status: string; auto_scores_json: string | null }>();

  const recentJobs = await db
    .prepare(
      `SELECT job_id, summary_json, config_json
       FROM training_jobs
       WHERE voice_id = ? AND status = 'completed'
       ORDER BY completed_at DESC
       LIMIT 30`
    )
    .bind(voiceId)
    .all<{ job_id: string; summary_json: string | null; config_json: string | null }>();

  const result: AssembledCandidate[] = [];
  const seen = new Set<string>();

  if (voice?.checkpoint_r2_prefix) {
    const prefix = voice.checkpoint_r2_prefix;
    if (!seen.has(prefix)) {
      seen.add(prefix);
      const isCarried = (carried.results ?? []).some((c: { checkpoint_r2_prefix: string }) => c.checkpoint_r2_prefix === prefix);
      if (!isCarried) {
        result.push({
          checkpoint_r2_prefix: prefix,
          job_id: voice.checkpoint_job_id ?? null,
          run_name: voice.run_name ?? null,
          epoch: voice.epoch ?? null,
          source: "champion_carry",
          auto_scores: {},
          ranking_score: voice.checkpoint_score ?? 0,
        });
      }
    }
  }

  const sourceMap: Record<string, ArenaCandidateSource> = {
    champion: "champion_carry",
    second: "second_carry",
    third: "third_carry",
  };

  const MAX_NON_CHAMPION_CARRIES = 2;

  for (const row of carried.results ?? []) {
    if (seen.has(row.checkpoint_r2_prefix)) continue;

    if (row.retention_status !== "champion") {
      const history = await db.prepare(
        `SELECT source FROM arena_candidates
         WHERE voice_id = ? AND checkpoint_r2_prefix = ?
         ORDER BY created_at DESC`
      ).bind(voiceId, row.checkpoint_r2_prefix).all<{ source: string }>();
      let consecutiveCarries = 0;
      for (const h of history.results ?? []) {
        if (h.source === "second_carry" || h.source === "third_carry") consecutiveCarries++;
        else break;
      }
      if (consecutiveCarries >= MAX_NON_CHAMPION_CARRIES) continue;
    }

    seen.add(row.checkpoint_r2_prefix);
    let scores: Record<string, number | null> = {};
    try { scores = JSON.parse(row.auto_scores_json ?? "{}"); } catch { /* empty */ }
    const cs = scores as unknown as CheckpointScores;
    result.push({
      checkpoint_r2_prefix: row.checkpoint_r2_prefix,
      job_id: row.job_id,
      run_name: row.run_name,
      epoch: row.epoch,
      source: sourceMap[row.retention_status] ?? "champion_carry",
      auto_scores: scores,
      ranking_score: computeRankingScore(cs, useCalibrated ? effectiveWeights : undefined),
    });
  }

  const newCandidates: AssembledCandidate[] = [];
  for (const job of recentJobs.results ?? []) {
    let summary: Record<string, unknown> = {};
    try { summary = JSON.parse(job.summary_json ?? "{}"); } catch { /* empty */ }
    // Training pipeline writes "evaluated_checkpoints"; fall back to "evaluated" for compat
    const evaluated = Array.isArray(summary.evaluated_checkpoints)
      ? summary.evaluated_checkpoints
      : Array.isArray(summary.evaluated)
        ? summary.evaluated
        : [];
    for (const ev of evaluated) {
      if (!ev || typeof ev !== "object") continue;
      const rec = ev as Record<string, unknown>;
      if (!rec.ok || !rec.prefix || typeof rec.prefix !== "string") continue;
      const prefix = rec.prefix as string;
      if (seen.has(prefix)) continue;

      // Scores may be stored as top-level fields (asr_score / asr) or embedded in message string.
      // Try both naming conventions, then fall back to parsing the message.
      const scores: CheckpointScores = {
        asr_score: readScore(rec, "asr_score", "asr"),
        speaker_score: readScore(rec, "speaker_score", "speaker"),
        health_score: readScore(rec, "health_score", "health"),
        tone_score: readScore(rec, "tone_score", "tone"),
        speed_score: readScore(rec, "speed_score", "speed"),
        style_score: readScore(rec, "style_score", "style"),
        overall_score: readScore(rec, "overall_score", "overall"),
        duration_score: readScore(rec, "duration_score", "duration"),
      };

      // If individual scores are missing, try parsing from the message string
      // Format: "...asr=0.974 tone=0.874 speed=0.907 health=1.000 duration=1.000..."
      if (typeof rec.message === "string") {
        const msg = rec.message as string;
        if (scores.asr_score === null) scores.asr_score = parseScoreFromMessage(msg, "asr");
        if (scores.speaker_score === null) scores.speaker_score = parseScoreFromMessage(msg, "speaker");
        if (scores.health_score === null) scores.health_score = parseScoreFromMessage(msg, "health");
        if (scores.tone_score === null) scores.tone_score = parseScoreFromMessage(msg, "tone");
        if (scores.speed_score === null) scores.speed_score = parseScoreFromMessage(msg, "speed");
        if (scores.style_score === null) scores.style_score = parseScoreFromMessage(msg, "style");
        if (scores.overall_score === null) scores.overall_score = parseScoreFromMessage(msg, "overall");
        if (scores.duration_score === null) scores.duration_score = parseScoreFromMessage(msg, "duration");
      }

      const hasAnyScore = Object.values(scores).some((v) => v !== null);
      if (hasAnyScore && !passesValidationGate(scores)) {
        // Gray-zone rescue: allow if passes hard safety and nearly passes soft thresholds
        if (calState === "canary" || calState === "active") {
          const rescue = grayZoneRescue(scores, effectiveWeights);
          if (!rescue.rescued) continue;
        } else {
          continue;
        }
      }

      seen.add(prefix);
      const rankingScore = hasAnyScore
        ? computeRankingScore(scores, useCalibrated ? effectiveWeights : undefined)
        : typeof rec.score === "number" ? rec.score : 0;
      newCandidates.push({
        checkpoint_r2_prefix: prefix,
        job_id: job.job_id,
        run_name: typeof rec.run_name === "string" ? rec.run_name : null,
        epoch: typeof rec.epoch === "number" ? rec.epoch : null,
        source: "new",
        auto_scores: scores as unknown as Record<string, number | null>,
        ranking_score: rankingScore,
      });
    }
  }

  newCandidates.sort((a, b) => b.ranking_score - a.ranking_score);
  const slotsForNew = 9 - result.length;
  result.push(...newCandidates.slice(0, Math.max(0, slotsForNew)));

  const algorithm = result.length <= 6 ? "round_robin" as const : "swiss" as const;
  return { candidates: result, algorithm };
}

export function computeTotalRounds(candidateCount: number, algorithm: "swiss" | "round_robin"): number {
  if (algorithm === "round_robin") return 1;
  return Math.ceil(Math.log2(Math.max(2, candidateCount))) + 1;
}

export function generateSwissPairings(
  standings: Standing[],
  roundNumber: number,
  previousMatches: Array<{ candidate_a_id: string; candidate_b_id: string }>,
  textCount: number,
): { pairs: Array<{ a: string; b: string; textIndex: number; displayOrder: "ab" | "ba" }>; byeCandidateId: string | null } {
  const sorted = [...standings].sort((a, b) => {
    if (a.wins !== b.wins) return b.wins - a.wins;
    return b.buchholz - a.buchholz;
  });

  const playedPairs = new Set(
    previousMatches.map(m => [m.candidate_a_id, m.candidate_b_id].sort().join("|"))
  );

  let byeCandidateId: string | null = null;

  if (sorted.length % 2 !== 0) {
    for (let i = sorted.length - 1; i >= 0; i--) {
      if (sorted[i].bye_count === 0) {
        byeCandidateId = sorted[i].candidate_id;
        sorted.splice(i, 1);
        break;
      }
    }
    if (byeCandidateId === null && sorted.length % 2 !== 0) {
      byeCandidateId = sorted[sorted.length - 1].candidate_id;
      sorted.pop();
    }
  }

  const pairs: Array<{ a: string; b: string; textIndex: number; displayOrder: "ab" | "ba" }> = [];
  const paired = new Set<string>();

  for (let i = 0; i < sorted.length; i++) {
    if (paired.has(sorted[i].candidate_id)) continue;
    for (let j = i + 1; j < sorted.length; j++) {
      if (paired.has(sorted[j].candidate_id)) continue;
      const key = [sorted[i].candidate_id, sorted[j].candidate_id].sort().join("|");
      if (playedPairs.has(key)) continue;
      paired.add(sorted[i].candidate_id);
      paired.add(sorted[j].candidate_id);
      const pairIndex = pairs.length;
      pairs.push({
        a: sorted[i].candidate_id,
        b: sorted[j].candidate_id,
        textIndex: (roundNumber * 7 + pairIndex) % Math.max(1, textCount),
        displayOrder: Math.random() < 0.5 ? "ab" : "ba",
      });
      break;
    }
  }

  for (let i = 0; i < sorted.length; i++) {
    if (paired.has(sorted[i].candidate_id)) continue;
    for (let j = i + 1; j < sorted.length; j++) {
      if (paired.has(sorted[j].candidate_id)) continue;
      paired.add(sorted[i].candidate_id);
      paired.add(sorted[j].candidate_id);
      const pairIndex = pairs.length;
      pairs.push({
        a: sorted[i].candidate_id,
        b: sorted[j].candidate_id,
        textIndex: (roundNumber * 7 + pairIndex) % Math.max(1, textCount),
        displayOrder: Math.random() < 0.5 ? "ab" : "ba",
      });
      break;
    }
  }

  return { pairs, byeCandidateId };
}

export function generateRoundRobinSchedule(
  candidateIds: string[],
  textCount: number,
): Array<{ a: string; b: string; textIndex: number; displayOrder: "ab" | "ba" }> {
  const pairs: Array<{ a: string; b: string; textIndex: number; displayOrder: "ab" | "ba" }> = [];
  let pairIndex = 0;
  for (let i = 0; i < candidateIds.length; i++) {
    for (let j = i + 1; j < candidateIds.length; j++) {
      pairs.push({
        a: candidateIds[i],
        b: candidateIds[j],
        textIndex: pairIndex % Math.max(1, textCount),
        displayOrder: Math.random() < 0.5 ? "ab" : "ba",
      });
      pairIndex++;
    }
  }
  return pairs;
}

export async function submitVote(
  db: D1Database,
  matchId: string,
  winner: ArenaVoteWinner,
  confidence: ArenaVoteConfidence | null,
): Promise<{ success: boolean; conflict: boolean; roundComplete: boolean; sessionId: string }> {
  const match = await db
    .prepare("SELECT * FROM arena_matches WHERE match_id = ? LIMIT 1")
    .bind(matchId)
    .first<{ match_id: string; session_id: string; candidate_a_id: string; candidate_b_id: string; winner: string | null }>();

  if (!match) throw new Error("Match not found");
  if (match.winner !== null) return { success: false, conflict: true, roundComplete: false, sessionId: match.session_id };

  const now = Date.now();

  const lockResult = await db
    .prepare("UPDATE arena_matches SET winner = ?, confidence = ?, voted_at = ? WHERE match_id = ? AND winner IS NULL")
    .bind(winner, confidence, now, matchId)
    .run();

  if ((lockResult.meta?.changes ?? 0) === 0) {
    return { success: false, conflict: true, roundComplete: false, sessionId: match.session_id };
  }

  const statsStmts: D1PreparedStatement[] = [];
  if (winner === "a") {
    statsStmts.push(db.prepare("UPDATE arena_candidates SET wins = wins + 1 WHERE candidate_id = ?").bind(match.candidate_a_id));
    statsStmts.push(db.prepare("UPDATE arena_candidates SET losses = losses + 1 WHERE candidate_id = ?").bind(match.candidate_b_id));
  } else if (winner === "b") {
    statsStmts.push(db.prepare("UPDATE arena_candidates SET wins = wins + 1 WHERE candidate_id = ?").bind(match.candidate_b_id));
    statsStmts.push(db.prepare("UPDATE arena_candidates SET losses = losses + 1 WHERE candidate_id = ?").bind(match.candidate_a_id));
  } else {
    statsStmts.push(db.prepare("UPDATE arena_candidates SET ties = ties + 1 WHERE candidate_id = ?").bind(match.candidate_a_id));
    statsStmts.push(db.prepare("UPDATE arena_candidates SET ties = ties + 1 WHERE candidate_id = ?").bind(match.candidate_b_id));
  }
  await db.batch(statsStmts);

  const session = await db
    .prepare("SELECT current_round FROM arena_sessions WHERE session_id = ? LIMIT 1")
    .bind(match.session_id)
    .first<{ current_round: number }>();

  const unvoted = await db
    .prepare("SELECT COUNT(*) as cnt FROM arena_matches WHERE session_id = ? AND round_number = ? AND winner IS NULL")
    .bind(match.session_id, session?.current_round ?? 0)
    .first<{ cnt: number }>();

  return {
    success: true,
    conflict: false,
    roundComplete: (unvoted?.cnt ?? 1) === 0,
    sessionId: match.session_id,
  };
}

export async function advanceRound(
  db: D1Database,
  sessionId: string,
): Promise<{ advanced: boolean; finalized: boolean }> {
  const session = await db
    .prepare("SELECT current_round, total_rounds, algorithm FROM arena_sessions WHERE session_id = ? LIMIT 1")
    .bind(sessionId)
    .first<{ current_round: number; total_rounds: number | null; algorithm: string }>();

  if (!session) throw new Error("Session not found");

  const nextRound = session.current_round + 1;
  const isLast = session.total_rounds !== null && nextRound >= session.total_rounds;

  if (isLast) {
    await finalizeSession(db, sessionId);
    return { advanced: false, finalized: true };
  }

  const lockResult = await db
    .prepare("UPDATE arena_sessions SET current_round = ? WHERE session_id = ? AND current_round = ?")
    .bind(nextRound, sessionId, session.current_round)
    .run();

  if ((lockResult.meta?.changes ?? 0) === 0) return { advanced: false, finalized: false };

  if (session.algorithm === "swiss") {
    const candidates = await db
      .prepare("SELECT candidate_id, wins, losses, ties, bye_count, buchholz FROM arena_candidates WHERE session_id = ?")
      .bind(sessionId)
      .all<{ candidate_id: string; wins: number; losses: number; ties: number; bye_count: number; buchholz: number }>();

    const prevMatches = await db
      .prepare("SELECT candidate_a_id, candidate_b_id FROM arena_matches WHERE session_id = ?")
      .bind(sessionId)
      .all<{ candidate_a_id: string; candidate_b_id: string }>();

    const sessionDetail = await db
      .prepare("SELECT test_texts_json FROM arena_sessions WHERE session_id = ? LIMIT 1")
      .bind(sessionId)
      .first<{ test_texts_json: string }>();

    let textCount = 1;
    try { textCount = JSON.parse(sessionDetail?.test_texts_json ?? "[]").length; } catch { /* empty */ }

    const votedMatches = await db
      .prepare("SELECT candidate_a_id, candidate_b_id, winner FROM arena_matches WHERE session_id = ? AND winner IS NOT NULL")
      .bind(sessionId)
      .all<{ candidate_a_id: string; candidate_b_id: string; winner: string }>();

    const winsLookup = new Map<string, number>();
    for (const c of candidates.results ?? []) winsLookup.set(c.candidate_id, c.wins);

    const oppWinsMap = new Map<string, number[]>();
    for (const c of candidates.results ?? []) oppWinsMap.set(c.candidate_id, []);
    for (const m of votedMatches.results ?? []) {
      oppWinsMap.get(m.candidate_a_id)?.push(winsLookup.get(m.candidate_b_id) ?? 0);
      oppWinsMap.get(m.candidate_b_id)?.push(winsLookup.get(m.candidate_a_id) ?? 0);
    }

    const standings: Standing[] = (candidates.results ?? []).map((c: { candidate_id: string; wins: number; losses: number; ties: number; bye_count: number; buchholz: number }) => {
      const oppWins = oppWinsMap.get(c.candidate_id) ?? [];
      const avgOpp = oppWins.length > 0 ? oppWins.reduce((s: number, v: number) => s + v, 0) / oppWins.length : 0;
      const buch = oppWins.reduce((s: number, v: number) => s + v, 0) + (c.bye_count > 0 ? avgOpp * c.bye_count : 0);
      return {
        candidate_id: c.candidate_id,
        wins: c.wins,
        losses: c.losses,
        ties: c.ties,
        bye_count: c.bye_count,
        buchholz: buch,
      };
    });

    const { pairs, byeCandidateId } = generateSwissPairings(
      standings,
      nextRound,
      prevMatches.results ?? [],
      textCount,
    );

    if (byeCandidateId) {
      await db.prepare("UPDATE arena_candidates SET wins = wins + 1, bye_count = bye_count + 1 WHERE candidate_id = ?")
        .bind(byeCandidateId).run();
    }

    const now = Date.now();
    const matchStmts = pairs.map(p =>
      db.prepare(
        `INSERT INTO arena_matches (match_id, session_id, round_number, candidate_a_id, candidate_b_id, display_order, text_index, audio_a_r2_key, audio_b_r2_key, created_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
      ).bind(
        crypto.randomUUID(), sessionId, nextRound, p.a, p.b, p.displayOrder, p.textIndex,
        `arena/${sessionId}/${p.a}/text${p.textIndex}.wav`,
        `arena/${sessionId}/${p.b}/text${p.textIndex}.wav`,
        now,
      )
    );
    if (matchStmts.length > 0) await db.batch(matchStmts);
  }

  return { advanced: true, finalized: false };
}

export async function finalizeSession(
  db: D1Database,
  sessionId: string,
): Promise<void> {
  const candidates = await db
    .prepare("SELECT candidate_id, wins, losses, ties, bye_count FROM arena_candidates WHERE session_id = ?")
    .bind(sessionId)
    .all<{ candidate_id: string; wins: number; losses: number; ties: number; bye_count: number }>();

  const matches = await db
    .prepare("SELECT candidate_a_id, candidate_b_id, winner FROM arena_matches WHERE session_id = ? AND winner IS NOT NULL")
    .bind(sessionId)
    .all<{ candidate_a_id: string; candidate_b_id: string; winner: string }>();

  const winsMap = new Map<string, number>();
  for (const c of candidates.results ?? []) {
    winsMap.set(c.candidate_id, c.wins);
  }

  const opponentWins = new Map<string, number[]>();
  for (const c of candidates.results ?? []) {
    opponentWins.set(c.candidate_id, []);
  }
  for (const m of matches.results ?? []) {
    opponentWins.get(m.candidate_a_id)?.push(winsMap.get(m.candidate_b_id) ?? 0);
    opponentWins.get(m.candidate_b_id)?.push(winsMap.get(m.candidate_a_id) ?? 0);
  }

  type CandidateRow = { candidate_id: string; wins: number; losses: number; ties: number; bye_count: number };
  const ranked = (candidates.results ?? []).map((c: CandidateRow) => {
    const oppWins = opponentWins.get(c.candidate_id) ?? [];
    const avgOppWins = oppWins.length > 0 ? oppWins.reduce((s: number, v: number) => s + v, 0) / oppWins.length : 0;
    const byeBuchholz = c.bye_count > 0 ? avgOppWins * c.bye_count : 0;
    const buchholz = oppWins.reduce((s: number, v: number) => s + v, 0) + byeBuchholz;
    return { candidate_id: c.candidate_id, wins: c.wins, buchholz };
  });

  ranked.sort((a: { wins: number; buchholz: number }, b: { wins: number; buchholz: number }) => {
    if (a.wins !== b.wins) return b.wins - a.wins;
    return b.buchholz - a.buchholz;
  });

  const now = Date.now();
  const stmts: D1PreparedStatement[] = [];

  for (let i = 0; i < ranked.length; i++) {
    const r = ranked[i];
    let retention: string;
    if (i === 0) retention = "champion";
    else if (i === 1) retention = "second";
    else if (i === 2) retention = "third";
    else retention = "eliminated";

    stmts.push(
      db.prepare("UPDATE arena_candidates SET final_rank = ?, buchholz = ?, retention_status = ?, eliminated_at = ? WHERE candidate_id = ?")
        .bind(i + 1, r.buchholz, retention, retention === "eliminated" ? now : null, r.candidate_id)
    );
  }

  const winnerId = ranked.length > 0 ? ranked[0].candidate_id : null;
  const rankingData = ranked.map((r: { candidate_id: string; wins: number; buchholz: number }, i: number) => ({ rank: i + 1, candidate_id: r.candidate_id, wins: r.wins, buchholz: r.buchholz }));

  stmts.push(
    db.prepare("UPDATE arena_sessions SET status = 'completed', winner_candidate_id = ?, ranking_json = ?, completed_at = ? WHERE session_id = ?")
      .bind(winnerId, JSON.stringify(rankingData), now, sessionId)
  );

  const prevCarried = await db
    .prepare("SELECT candidate_id FROM arena_candidates WHERE voice_id = (SELECT voice_id FROM arena_sessions WHERE session_id = ? LIMIT 1) AND session_id != ? AND retention_status IN ('champion', 'second', 'third')")
    .bind(sessionId, sessionId)
    .all<{ candidate_id: string }>();

  for (const old of prevCarried.results ?? []) {
    stmts.push(
      db.prepare("UPDATE arena_candidates SET retention_status = 'eliminated', eliminated_at = ? WHERE candidate_id = ?")
        .bind(now, old.candidate_id)
    );
  }

  await db.batch(stmts);
}

export interface BootstrapArenaInput {
  voiceId: string;
  testTexts: string[];
  seed: number;
  settings: VoiceSettings;
}

export interface BootstrapArenaResult {
  session: ArenaSession;
  candidates: ArenaCandidate[];
  matches: ArenaMatch[];
}

export async function bootstrapArenaSession(
  db: D1Database,
  input: BootstrapArenaInput,
): Promise<BootstrapArenaResult | null> {
  const { candidates: assembled, algorithm } = await assembleArenaCandidates(db, input.voiceId);
  if (assembled.length < 2) {
    return null;
  }

  const now = Date.now();
  const sessionId = crypto.randomUUID();
  const totalRounds = computeTotalRounds(assembled.length, algorithm);

  const session: ArenaSession = {
    session_id: sessionId,
    voice_id: input.voiceId,
    status: "assembling",
    algorithm,
    current_round: 1,
    total_rounds: totalRounds,
    test_texts: input.testTexts,
    seed: input.seed,
    settings: input.settings,
    ranking: {},
    winner_candidate_id: null,
    promoted: false,
    notes: null,
    created_at: now,
    completed_at: null,
  };

  const candidateRecords: ArenaCandidate[] = assembled.map((ac, idx) => ({
    candidate_id: crypto.randomUUID(),
    session_id: sessionId,
    voice_id: input.voiceId,
    checkpoint_r2_prefix: ac.checkpoint_r2_prefix,
    job_id: ac.job_id,
    run_name: ac.run_name,
    epoch: ac.epoch,
    source: ac.source,
    seed_rank: idx + 1,
    final_rank: null,
    wins: 0,
    losses: 0,
    ties: 0,
    bye_count: 0,
    buchholz: 0,
    retention_status: "active",
    auto_scores: ac.auto_scores,
    created_at: now,
    eliminated_at: null,
  }));

  const matchRecords: ArenaMatch[] = [];
  if (algorithm === "round_robin") {
    const pairs = generateRoundRobinSchedule(
      candidateRecords.map((cr) => cr.candidate_id),
      input.testTexts.length,
    );
    for (const p of pairs) {
      matchRecords.push({
        match_id: crypto.randomUUID(),
        session_id: sessionId,
        round_number: 1,
        candidate_a_id: p.a,
        candidate_b_id: p.b,
        display_order: p.displayOrder,
        text_index: p.textIndex,
        audio_a_r2_key: null,
        audio_b_r2_key: null,
        winner: null,
        confidence: null,
        replay_count_a: 0,
        replay_count_b: 0,
        created_at: now,
        voted_at: null,
      });
    }
  } else {
    const standings = candidateRecords.map((cr) => ({
      candidate_id: cr.candidate_id,
      wins: 0,
      losses: 0,
      ties: 0,
      bye_count: 0,
      buchholz: 0,
    }));
    const { pairs, byeCandidateId } = generateSwissPairings(standings, 1, [], input.testTexts.length);

    if (byeCandidateId) {
      const byeCandidate = candidateRecords.find((cr) => cr.candidate_id === byeCandidateId);
      if (byeCandidate) {
        byeCandidate.wins = 1;
        byeCandidate.bye_count = 1;
      }
    }

    for (const p of pairs) {
      matchRecords.push({
        match_id: crypto.randomUUID(),
        session_id: sessionId,
        round_number: 1,
        candidate_a_id: p.a,
        candidate_b_id: p.b,
        display_order: p.displayOrder,
        text_index: p.textIndex,
        audio_a_r2_key: null,
        audio_b_r2_key: null,
        winner: null,
        confidence: null,
        replay_count_a: 0,
        replay_count_b: 0,
        created_at: now,
        voted_at: null,
      });
    }
  }

  await createArenaSession(db, session);
  for (const candidate of candidateRecords) {
    await createArenaCandidate(db, candidate);
  }
  for (const match of matchRecords) {
    await createArenaMatch(db, match);
  }

  return {
    session,
    candidates: candidateRecords,
    matches: matchRecords,
  };
}
