import { Hono } from "hono";
import {
  createArenaCandidate,
  createArenaMatch,
  createArenaSession,
  getArenaSession,
  getVoice,
  listArenaCandidates,
  listArenaMatches,
  updateArenaMatch,
  updateArenaSession,
  updateVoice,
} from "../lib/d1";
import {
  assembleArenaCandidates,
  computeTotalRounds,
  generateSwissPairings,
  generateRoundRobinSchedule,
  submitVote,
  advanceRound,
  finalizeSession,
} from "../lib/arena";
import { invokeServerlessAsync, getServerlessStatus } from "../lib/runpod";
import { normalizeLanguageCode } from "./tts";
import { calibrateFromArenaData, publishCalibration } from "../lib/arena-calibration";
import { authMiddleware } from "../middleware/auth";
import type {
  AppContext,
  ArenaCandidate,
  ArenaMatch,
  ArenaSession,
  ArenaVoteConfidence,
  ArenaVoteWinner,
  VoiceSettings,
} from "../types";

interface GenerationJob {
  job_id: string;
  r2_key: string;
  candidate_id: string;
  text_index: number;
  status: "pending" | "completed" | "failed";
  error?: string;
}

interface GenerationTracking {
  jobs: GenerationJob[];
}

function parseTracking(notes: string | null): GenerationTracking | null {
  if (!notes) return null;
  try {
    const parsed = JSON.parse(notes);
    return Array.isArray(parsed.jobs) ? parsed as GenerationTracking : null;
  } catch { return null; }
}

const decodeBase64ToBytes = (value: string): Uint8Array => {
  const decoded = atob(value);
  const bytes = new Uint8Array(decoded.length);
  for (let i = 0; i < decoded.length; i++) bytes[i] = decoded.charCodeAt(i);
  return bytes;
};

const app = new Hono<AppContext>();
app.use("*", authMiddleware);

app.post("/sessions", async (c) => {
  try {
    const body = await c.req.json<{
      voice_id: string;
      test_texts: string[];
      seed?: number;
      settings?: VoiceSettings;
    }>();

    if (!body.voice_id || !Array.isArray(body.test_texts) || body.test_texts.length === 0) {
      return c.json({ detail: { message: "voice_id and non-empty test_texts are required" } }, 400);
    }

    const voice = await getVoice(c.env.DB, body.voice_id);
    if (!voice) {
      return c.json({ detail: { message: "Voice not found" } }, 404);
    }

    const { candidates: assembled, algorithm } = await assembleArenaCandidates(c.env.DB, body.voice_id);
    if (assembled.length < 2) {
      return c.json({ detail: { message: "Need at least 2 candidates for an arena session" } }, 400);
    }

    const totalRounds = computeTotalRounds(assembled.length, algorithm);
    const seed = body.seed ?? 42;
    const now = Date.now();
    const sessionId = crypto.randomUUID();

    const session: ArenaSession = {
      session_id: sessionId,
      voice_id: body.voice_id,
      status: "assembling",
      algorithm,
      current_round: 1,
      total_rounds: totalRounds,
      test_texts: body.test_texts,
      seed,
      settings: body.settings ?? {},
      ranking: {},
      winner_candidate_id: null,
      promoted: false,
      notes: null,
      created_at: now,
      completed_at: null,
    };

    await createArenaSession(c.env.DB, session);

    const candidateRecords: ArenaCandidate[] = assembled.map((ac, idx) => ({
      candidate_id: crypto.randomUUID(),
      session_id: sessionId,
      voice_id: body.voice_id,
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

    for (const candidate of candidateRecords) {
      await createArenaCandidate(c.env.DB, candidate);
    }

    const matchRecords: ArenaMatch[] = [];
    if (algorithm === "round_robin") {
      const pairs = generateRoundRobinSchedule(
        candidateRecords.map((cr) => cr.candidate_id),
        body.test_texts.length,
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
      const { pairs, byeCandidateId } = generateSwissPairings(standings, 1, [], body.test_texts.length);

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

    for (const match of matchRecords) {
      await createArenaMatch(c.env.DB, match);
    }

    return c.json({
      ...session,
      candidates: candidateRecords,
      matches: matchRecords,
    }, 201);
  } catch (err) {
    console.error("POST /sessions error:", err);
    return c.json({ detail: { message: err instanceof Error ? err.message : "Internal server error" } }, 500);
  }
});

app.get("/sessions/:sessionId", async (c) => {
  try {
    const sessionId = c.req.param("sessionId");
    let session = await getArenaSession(c.env.DB, sessionId);
    if (!session) {
      return c.json({ detail: { message: "Session not found" } }, 404);
    }

    if (session.status === "generating") {
      const tracking = parseTracking(session.notes);
      if (tracking) {
        const pending = tracking.jobs.filter((j) => j.status === "pending");
        const BATCH_SIZE = 5;
        const batch = pending.slice(0, BATCH_SIZE);
        let changed = false;

        for (const job of batch) {
          try {
            const resp = await getServerlessStatus(c.env, c.env.RUNPOD_ENDPOINT_ID, job.job_id);
            const status = String(resp.status ?? "");

            if (status === "COMPLETED") {
              const output = (resp.output ?? {}) as { audio?: string; error?: string };
              if (output.audio) {
                const audioBytes = decodeBase64ToBytes(output.audio);
                await c.env.R2.put(job.r2_key, audioBytes, {
                  httpMetadata: { contentType: "audio/wav" },
                });
                job.status = "completed";
              } else {
                job.status = "failed";
                job.error = output.error ?? "RunPod returned no audio data";
              }
              changed = true;
            } else if (status === "FAILED") {
              job.status = "failed";
              job.error = String((resp.output as Record<string, unknown>)?.error ?? resp.error ?? "Generation failed");
              changed = true;
            }
          } catch {
            // Transient RunPod error, will retry on next poll
          }
        }

        if (changed) {
          const allDone = tracking.jobs.every((j) => j.status !== "pending");
          const updates: Record<string, unknown> = { notes: JSON.stringify(tracking) };

          if (allDone) {
            updates.status = "active";

            const allMatches = await listArenaMatches(c.env.DB, { session_id: sessionId });
            for (const match of allMatches) {
              const audioA = tracking.jobs.find(
                (j) => j.candidate_id === match.candidate_a_id && j.text_index === match.text_index && j.status === "completed",
              );
              const audioB = tracking.jobs.find(
                (j) => j.candidate_id === match.candidate_b_id && j.text_index === match.text_index && j.status === "completed",
              );
              if (audioA || audioB) {
                await updateArenaMatch(c.env.DB, match.match_id, {
                  audio_a_r2_key: audioA?.r2_key ?? null,
                  audio_b_r2_key: audioB?.r2_key ?? null,
                });
              }
            }
          }

          await updateArenaSession(c.env.DB, sessionId, updates);
          session = await getArenaSession(c.env.DB, sessionId);
          if (!session) {
            return c.json({ detail: { message: "Session not found" } }, 404);
          }
        }
      }
    }

    const candidates = await listArenaCandidates(c.env.DB, { session_id: sessionId });
    const matches = await listArenaMatches(c.env.DB, {
      session_id: sessionId,
      round_number: session.current_round,
    });

    const tracking = parseTracking(session.notes);
    const generationProgress = tracking ? {
      total: tracking.jobs.length,
      completed: tracking.jobs.filter((j) => j.status === "completed").length,
      failed: tracking.jobs.filter((j) => j.status === "failed").length,
      pending: tracking.jobs.filter((j) => j.status === "pending").length,
    } : undefined;

    return c.json({ ...session, candidates, matches, generation_progress: generationProgress });
  } catch (err) {
    console.error("GET /sessions/:sessionId error:", err);
    return c.json({ detail: { message: err instanceof Error ? err.message : "Internal server error" } }, 500);
  }
});

app.post("/sessions/:sessionId/generate", async (c) => {
  try {
    const sessionId = c.req.param("sessionId");
    const session = await getArenaSession(c.env.DB, sessionId);
    if (!session) {
      return c.json({ detail: { message: "Session not found" } }, 404);
    }

    if (session.status !== "assembling") {
      return c.json({ detail: { message: `Cannot generate: session status is '${session.status}', expected 'assembling'` } }, 400);
    }

    const voice = await getVoice(c.env.DB, session.voice_id);
    if (!voice) {
      return c.json({ detail: { message: "Voice not found" } }, 404);
    }

    const candidates = await listArenaCandidates(c.env.DB, { session_id: sessionId });
    const speakerName = voice.speaker_name ?? session.voice_id;
    const modelId = voice.model_id ?? "qwen3-tts-1.7b";
    const language = normalizeLanguageCode(voice.labels?.language);
    const seed = session.seed ?? 42;

    const tasks: Array<{ candidate_id: string; text_index: number; r2_key: string; input: Record<string, unknown> }> = [];
    for (const candidate of candidates) {
      for (let textIdx = 0; textIdx < session.test_texts.length; textIdx++) {
        tasks.push({
          candidate_id: candidate.candidate_id,
          text_index: textIdx,
          r2_key: `arena/${sessionId}/${candidate.candidate_id}/text${textIdx}.wav`,
          input: {
            text: session.test_texts[textIdx],
            voice_id: session.voice_id,
            speaker_name: speakerName,
            model_id: modelId,
            voice_settings: session.settings ?? {},
            seed,
            language,
            checkpoint_info: { r2_prefix: candidate.checkpoint_r2_prefix, type: "full" },
          },
        });
      }
    }

    const CONCURRENCY = 5;
    const jobs: GenerationJob[] = [];
    for (let i = 0; i < tasks.length; i += CONCURRENCY) {
      const batch = tasks.slice(i, i + CONCURRENCY);
      const results = await Promise.all(
        batch.map(async (t) => {
          try {
            const resp = await invokeServerlessAsync(c.env, c.env.RUNPOD_ENDPOINT_ID, t.input);
            const jobId = String(resp.id ?? "");
            return { ...t, job_id: jobId, status: (jobId ? "pending" : "failed") as "pending" | "failed", error: jobId ? undefined : "No job ID returned" };
          } catch (err) {
            return { ...t, job_id: "", status: "failed" as const, error: err instanceof Error ? err.message : "RunPod invocation failed" };
          }
        }),
      );
      for (const r of results) {
        jobs.push({ job_id: r.job_id, r2_key: r.r2_key, candidate_id: r.candidate_id, text_index: r.text_index, status: r.status, ...(r.error ? { error: r.error } : {}) });
      }
    }

    const tracking: GenerationTracking = { jobs };

    await updateArenaSession(c.env.DB, sessionId, {
      status: "generating",
      notes: JSON.stringify(tracking),
    });

    return c.json({
      session_id: sessionId,
      status: "generating",
      total_jobs: jobs.length,
      pending: jobs.filter((j) => j.status === "pending").length,
    });
  } catch (err) {
    console.error("POST /sessions/:sessionId/generate error:", err);
    return c.json({ detail: { message: err instanceof Error ? err.message : "Internal server error" } }, 500);
  }
});

app.get("/audio/:r2Key{.+}", async (c) => {
  try {
    const r2Key = decodeURIComponent(c.req.param("r2Key"));
    if (!r2Key.startsWith("arena/")) {
      return c.json({ detail: { message: "Invalid arena audio key" } }, 400);
    }
    const object = await c.env.R2.get(r2Key);
    if (!object) {
      return c.json({ detail: { message: "Audio not found" } }, 404);
    }
    return new Response(object.body, {
      headers: {
        "Content-Type": object.httpMetadata?.contentType ?? "audio/wav",
        "Cache-Control": "public, max-age=86400",
        "Content-Length": String(object.size),
      },
    });
  } catch (err) {
    console.error("GET /audio error:", err);
    return c.json({ detail: { message: err instanceof Error ? err.message : "Internal server error" } }, 500);
  }
});

app.post("/matches/:matchId/vote", async (c) => {
  try {
    const matchId = c.req.param("matchId");
    const body = await c.req.json<{
      winner: ArenaVoteWinner;
      confidence?: ArenaVoteConfidence;
    }>();

    if (!body.winner || !["a", "b", "tie", "both_bad"].includes(body.winner)) {
      return c.json({ detail: { message: "winner must be one of: a, b, tie, both_bad" } }, 400);
    }

    const confidence = body.confidence ?? null;
    if (confidence !== null && !["clear", "slight"].includes(confidence)) {
      return c.json({ detail: { message: "confidence must be one of: clear, slight" } }, 400);
    }

    const result = await submitVote(c.env.DB, matchId, body.winner, confidence);

    if (result.conflict) {
      return c.json({ detail: { message: "Match already voted" } }, 409);
    }

    let advanceResult = { advanced: false, finalized: false };
    if (result.roundComplete) {
      advanceResult = await advanceRound(c.env.DB, result.sessionId);
    }

    if (advanceResult.finalized) {
      const voiceRow = await c.env.DB.prepare("SELECT voice_id FROM arena_sessions WHERE session_id = ? LIMIT 1").bind(result.sessionId).first<{ voice_id: string }>();
      if (voiceRow) {
        c.executionCtx.waitUntil(
          calibrateFromArenaData(c.env.DB, voiceRow.voice_id)
            .then(cal => publishCalibration(c.env.DB, voiceRow.voice_id, cal))
            .catch(() => {}),
        );
      }
    }

    return c.json({
      success: result.success,
      round_complete: result.roundComplete,
      round_advanced: advanceResult.advanced,
      session_finalized: advanceResult.finalized,
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : "Internal server error";
    if (message === "Match not found") {
      return c.json({ detail: { message } }, 404);
    }
    console.error("POST /matches/:matchId/vote error:", err);
    return c.json({ detail: { message } }, 500);
  }
});

app.post("/sessions/:sessionId/complete", async (c) => {
  try {
    const sessionId = c.req.param("sessionId");
    const session = await getArenaSession(c.env.DB, sessionId);
    if (!session) {
      return c.json({ detail: { message: "Session not found" } }, 404);
    }

    if (session.status === "completed") {
      return c.json({ detail: { message: "Session already completed" } }, 400);
    }

    await finalizeSession(c.env.DB, sessionId);

    c.executionCtx.waitUntil(
      calibrateFromArenaData(c.env.DB, session.voice_id)
        .then(cal => publishCalibration(c.env.DB, session.voice_id, cal))
        .catch(() => {}),
    );

    const updated = await getArenaSession(c.env.DB, sessionId);
    const candidates = await listArenaCandidates(c.env.DB, { session_id: sessionId });

    return c.json({ session: updated, candidates });
  } catch (err) {
    console.error("POST /sessions/:sessionId/complete error:", err);
    return c.json({ detail: { message: err instanceof Error ? err.message : "Internal server error" } }, 500);
  }
});

app.post("/sessions/:sessionId/promote", async (c) => {
  try {
    const sessionId = c.req.param("sessionId");
    const session = await getArenaSession(c.env.DB, sessionId);
    if (!session) {
      return c.json({ detail: { message: "Session not found" } }, 404);
    }

    if (session.status !== "completed") {
      return c.json({ detail: { message: "Session must be completed before promotion" } }, 400);
    }

    if (!session.winner_candidate_id) {
      return c.json({ detail: { message: "No winner to promote" } }, 400);
    }

    if (session.promoted) {
      return c.json({ detail: { message: "Winner already promoted" } }, 400);
    }

    const candidates = await listArenaCandidates(c.env.DB, { session_id: sessionId });
    const winner = candidates.find((cr) => cr.candidate_id === session.winner_candidate_id);
    if (!winner) {
      return c.json({ detail: { message: "Winner candidate not found" } }, 404);
    }

    const voice = await getVoice(c.env.DB, session.voice_id);
    if (!voice) {
      return c.json({ detail: { message: "Voice not found" } }, 404);
    }

    await updateVoice(c.env.DB, voice.voice_id, {
      checkpoint_r2_prefix: winner.checkpoint_r2_prefix,
      run_name: winner.run_name,
      epoch: winner.epoch,
      checkpoint_job_id: winner.job_id,
      updated_at: Date.now(),
    });

    await updateArenaSession(c.env.DB, sessionId, { promoted: true });

    return c.json({
      promoted: true,
      voice_id: voice.voice_id,
      checkpoint_r2_prefix: winner.checkpoint_r2_prefix,
      run_name: winner.run_name,
      epoch: winner.epoch,
    });
  } catch (err) {
    console.error("POST /sessions/:sessionId/promote error:", err);
    return c.json({ detail: { message: err instanceof Error ? err.message : "Internal server error" } }, 500);
  }
});

app.get("/calibration", async (c) => {
  try {
    const voiceId = c.req.query("voice_id") || undefined;
    const result = await calibrateFromArenaData(c.env.DB, voiceId);
    return c.json(result);
  } catch (err) {
    console.error("GET /calibration error:", err);
    return c.json({ detail: { message: err instanceof Error ? err.message : "Internal server error" } }, 500);
  }
});

app.post("/calibration/apply", async (c) => {
  try {
    const body = await c.req.json<{ voice_id?: string }>();
    const targetVoiceId = body.voice_id ?? "__global__";

    const result = await calibrateFromArenaData(c.env.DB, targetVoiceId === "__global__" ? undefined : targetVoiceId);
    await publishCalibration(c.env.DB, targetVoiceId, result);

    return c.json({
      applied: true,
      voice_id: targetVoiceId,
      ...result,
    });
  } catch (err) {
    console.error("POST /calibration/apply error:", err);
    return c.json({ detail: { message: err instanceof Error ? err.message : "Internal server error" } }, 500);
  }
});

export default app;
