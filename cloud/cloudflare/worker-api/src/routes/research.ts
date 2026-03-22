import { Hono } from "hono";
import {
  getVoice,
  getVoiceResearchState,
  listVoiceResearchJournal,
  updateResearchStateAutonomy,
  upsertVoiceResearchState,
} from "../lib/d1";
import {
  buildResearchSnapshot,
  buildStrategyBrief,
  maybeAdvanceResearchLoop,
} from "../lib/research-loop";
import { authMiddleware } from "../middleware/auth";
import type { AppContext, AutonomyMode, ResearchTrigger } from "../types";

const app = new Hono<AppContext>();
app.use("*", authMiddleware);

app.get("/:voiceId/state", async (c) => {
  try {
    const voiceId = c.req.param("voiceId");
    const voice = await getVoice(c.env.DB, voiceId);
    if (!voice) {
      return c.json({ detail: { message: "Voice not found" } }, 404);
    }

    const state = await getVoiceResearchState(c.env.DB, voiceId);
    const brief = await buildStrategyBrief(c.env.DB, voiceId);
    const recentJournal = await listVoiceResearchJournal(c.env.DB, voiceId, 5);

    return c.json({
      voice_id: voiceId,
      state,
      strategy_brief: brief,
      recent_journal: recentJournal,
    });
  } catch (err) {
    console.error("GET /research/:voiceId/state error:", err);
    return c.json({ detail: { message: err instanceof Error ? err.message : "Internal server error" } }, 500);
  }
});

app.get("/:voiceId/journal", async (c) => {
  try {
    const voiceId = c.req.param("voiceId");
    const voice = await getVoice(c.env.DB, voiceId);
    if (!voice) {
      return c.json({ detail: { message: "Voice not found" } }, 404);
    }

    const limitParam = c.req.query("limit");
    const limit = limitParam ? Math.max(1, Math.min(parseInt(limitParam, 10) || 20, 200)) : 20;
    const journal = await listVoiceResearchJournal(c.env.DB, voiceId, limit);

    return c.json({ voice_id: voiceId, entries: journal, count: journal.length });
  } catch (err) {
    console.error("GET /research/:voiceId/journal error:", err);
    return c.json({ detail: { message: err instanceof Error ? err.message : "Internal server error" } }, 500);
  }
});

app.post("/:voiceId/trigger", async (c) => {
  try {
    const voiceId = c.req.param("voiceId");
    const voice = await getVoice(c.env.DB, voiceId);
    if (!voice) {
      return c.json({ detail: { message: "Voice not found" } }, 404);
    }

    const body = await c.req.json<{ trigger?: ResearchTrigger; context?: Record<string, unknown> }>().catch(
      () => ({ trigger: undefined as ResearchTrigger | undefined, context: undefined as Record<string, unknown> | undefined }),
    );
    const trigger: ResearchTrigger = body.trigger ?? "manual";

    const result = await maybeAdvanceResearchLoop(
      c.env.DB,
      { OPENAI_API_KEY: c.env.OPENAI_API_KEY, OPENAI_ADVISOR_MODEL: c.env.OPENAI_ADVISOR_MODEL },
      voiceId,
      trigger,
      body.context,
    );

    return c.json({
      voice_id: voiceId,
      cycle_id: result.entry.cycle_id,
      action: result.action,
      executed: result.executed,
      arena_session_id: result.arena_session_id,
      campaign_id: result.campaign_id,
      state: result.state,
      journal_entry: result.entry,
    });
  } catch (err) {
    console.error("POST /research/:voiceId/trigger error:", err);
    return c.json({ detail: { message: err instanceof Error ? err.message : "Internal server error" } }, 500);
  }
});

app.get("/:voiceId/snapshot", async (c) => {
  try {
    const voiceId = c.req.param("voiceId");
    const voice = await getVoice(c.env.DB, voiceId);
    if (!voice) {
      return c.json({ detail: { message: "Voice not found" } }, 404);
    }

    const snapshot = await buildResearchSnapshot(c.env.DB, voiceId);
    return c.json(snapshot);
  } catch (err) {
    console.error("GET /research/:voiceId/snapshot error:", err);
    return c.json({ detail: { message: err instanceof Error ? err.message : "Internal server error" } }, 500);
  }
});

const VALID_AUTONOMY_MODES = new Set<AutonomyMode>(["supervised", "semi_auto", "auto"]);

app.patch("/:voiceId/autonomy", async (c) => {
  try {
    const voiceId = c.req.param("voiceId");
    const voice = await getVoice(c.env.DB, voiceId);
    if (!voice) {
      return c.json({ detail: { message: "Voice not found" } }, 404);
    }

    const body = await c.req.json<{ mode: string }>();
    if (!body.mode || !VALID_AUTONOMY_MODES.has(body.mode as AutonomyMode)) {
      return c.json({ detail: { message: "mode must be one of: supervised, semi_auto, auto" } }, 400);
    }

    const mode = body.mode as AutonomyMode;
    const state = await getVoiceResearchState(c.env.DB, voiceId);
    if (!state) {
      const now = Date.now();
      await upsertVoiceResearchState(c.env.DB, {
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
        autonomy_mode: mode,
        last_retrospective: null,
        created_at: now,
        updated_at: now,
      });
    } else {
      await updateResearchStateAutonomy(c.env.DB, voiceId, mode);
    }

    const updated = await getVoiceResearchState(c.env.DB, voiceId);
    return c.json({ voice_id: voiceId, autonomy_mode: updated?.autonomy_mode ?? mode });
  } catch (err) {
    console.error("PATCH /research/:voiceId/autonomy error:", err);
    return c.json({ detail: { message: err instanceof Error ? err.message : "Internal server error" } }, 500);
  }
});

export default app;
