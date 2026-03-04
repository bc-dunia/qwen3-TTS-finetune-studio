import { Hono } from "hono";
import { generatePresignedPut } from "../lib/r2";
import { authMiddleware } from "../middleware/auth";
import type { AppContext } from "../types";

const app = new Hono<AppContext>();
app.use("*", authMiddleware);

const sanitizeFileName = (name: string): string => name.replace(/[^a-zA-Z0-9._-]/g, "_");

app.post("/presigned", async (c) => {
  const body = (await c.req.json()) as {
    filename?: string;
    content_type?: string;
    voice_id?: string;
  };

  if (!body.filename || !body.content_type) {
    return c.json({ detail: { message: "filename and content_type are required" } }, 400);
  }

  const rawId = sanitizeFileName(body.voice_id ?? "shared");
  const voiceId = rawId && rawId !== "." && rawId !== ".." ? rawId : "shared";
  const safeFilename = sanitizeFileName(body.filename);
  const key = `datasets/${voiceId}/${Date.now()}_${crypto.randomUUID()}_${safeFilename}`;
  const uploadUrl = await generatePresignedPut(c.env, key, body.content_type, 900);

  return c.json({
    upload_url: uploadUrl,
    r2_key: key,
  });
});

export default app;
