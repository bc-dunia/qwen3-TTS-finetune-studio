import { Hono } from "hono";
import { generatePresignedPut } from "../lib/r2";
import { authMiddleware } from "../middleware/auth";
import type { AppContext } from "../types";

const app = new Hono<AppContext>();
app.use("*", authMiddleware);

const sanitizeFileName = (name: string): string => name.replace(/[^a-zA-Z0-9._-]/g, "_");
const MULTIPART_CHUNK_SIZE_BYTES = 8 * 1024 * 1024;
const getContentType = (value: string | null | undefined): string =>
  String(value ?? "").trim() || "application/octet-stream";
const sanitizeVoiceId = (voiceId: string | undefined): string => {
  const rawId = sanitizeFileName(voiceId ?? "shared");
  return rawId && rawId !== "." && rawId !== ".." ? rawId : "shared";
};
const isUploadFile = (
  value: unknown
): value is File & { arrayBuffer(): Promise<ArrayBuffer> } =>
  typeof value === "object" &&
  value !== null &&
  "arrayBuffer" in value &&
  "name" in value &&
  "size" in value;

app.post("/presigned", async (c) => {
  const body = (await c.req.json()) as {
    filename?: string;
    content_type?: string;
    voice_id?: string;
  };

  if (!body.filename || !body.content_type) {
    return c.json({ detail: { message: "filename and content_type are required" } }, 400);
  }

  const voiceId = sanitizeVoiceId(body.voice_id);
  const safeFilename = sanitizeFileName(body.filename);
  const key = `datasets/${voiceId}/${Date.now()}_${crypto.randomUUID()}_${safeFilename}`;
  const uploadUrl = await generatePresignedPut(c.env, key, body.content_type, 900);

  return c.json({
    upload_url: uploadUrl,
    r2_key: key,
  });
});

app.post("/multipart/start", async (c) => {
  const body = (await c.req.json()) as {
    filename?: string;
    content_type?: string;
    voice_id?: string;
  };

  if (!body.filename || !body.content_type) {
    return c.json({ detail: { message: "filename and content_type are required" } }, 400);
  }

  const voiceId = sanitizeVoiceId(body.voice_id);
  const safeFilename = sanitizeFileName(body.filename);
  const key = `datasets/${voiceId}/${Date.now()}_${crypto.randomUUID()}_${safeFilename}`;
  const upload = await c.env.R2.createMultipartUpload(key, {
    httpMetadata: {
      contentType: body.content_type,
    },
  });

  return c.json({
    upload_id: upload.uploadId,
    r2_key: key,
    chunk_size_bytes: MULTIPART_CHUNK_SIZE_BYTES,
  });
});

app.post("/multipart/part", async (c) => {
  const key = String(c.req.query("key") ?? "").trim();
  const uploadId = String(c.req.query("upload_id") ?? "").trim();
  const partNumber = Number(c.req.query("part_number") ?? 0);

  if (!key || !uploadId || !Number.isInteger(partNumber) || partNumber < 1 || partNumber > 10_000) {
    return c.json({ detail: { message: "key, upload_id, and a valid part_number are required" } }, 400);
  }

  const chunk = await c.req.arrayBuffer();
  if (chunk.byteLength <= 0) {
    return c.json({ detail: { message: "multipart chunk body is required" } }, 400);
  }

  const upload = c.env.R2.resumeMultipartUpload(key, uploadId);
  const part = await upload.uploadPart(partNumber, chunk);
  return c.json(part);
});

app.post("/multipart/complete", async (c) => {
  const body = (await c.req.json()) as {
    r2_key?: string;
    upload_id?: string;
    parts?: Array<{ partNumber?: number; etag?: string }>;
  };

  if (!body.r2_key || !body.upload_id || !Array.isArray(body.parts) || body.parts.length === 0) {
    return c.json({ detail: { message: "r2_key, upload_id, and parts are required" } }, 400);
  }

  const parts = body.parts
    .filter(
      (part): part is { partNumber: number; etag: string } =>
        Number.isInteger(part?.partNumber) && Number(part.partNumber) > 0 && typeof part?.etag === "string" && part.etag.length > 0
    )
    .sort((a, b) => a.partNumber - b.partNumber);

  if (parts.length !== body.parts.length) {
    return c.json({ detail: { message: "parts must contain valid partNumber and etag values" } }, 400);
  }

  const upload = c.env.R2.resumeMultipartUpload(body.r2_key, body.upload_id);
  const completed = await upload.complete(parts);
  return c.json({
    r2_key: completed.key,
    size: completed.size,
  });
});

app.post("/multipart/abort", async (c) => {
  const body = (await c.req.json()) as {
    r2_key?: string;
    upload_id?: string;
  };

  if (!body.r2_key || !body.upload_id) {
    return c.json({ detail: { message: "r2_key and upload_id are required" } }, 400);
  }

  const upload = c.env.R2.resumeMultipartUpload(body.r2_key, body.upload_id);
  await upload.abort();
  return c.json({ status: "aborted" });
});

app.post("/raw", async (c) => {
  const directFilename = String(c.req.query("filename") ?? "").trim();
  if (directFilename) {
    const voiceId = sanitizeVoiceId(c.req.query("voice_id") ?? undefined);
    const safeFilename = sanitizeFileName(directFilename);
    const contentType = getContentType(c.req.query("content_type") ?? c.req.header("content-type"));
    const body = c.req.raw.body;

    if (!body) {
      return c.json({ detail: { message: "request body is required" } }, 400);
    }

    const key = `datasets/${voiceId}/${Date.now()}_${crypto.randomUUID()}_${safeFilename}`;
    await c.env.R2.put(key, body, {
      httpMetadata: {
        contentType,
      },
    });

    return c.json({
      r2_key: key,
      content_type: contentType,
    });
  }

  const form = await c.req.formData();
  const fileValue = form.get("file");
  const voiceId = sanitizeVoiceId(
    typeof form.get("voice_id") === "string" ? String(form.get("voice_id")) : undefined
  );

  if (!isUploadFile(fileValue) || fileValue.size <= 0) {
    return c.json({ detail: { message: "file is required" } }, 400);
  }

  const safeFilename = sanitizeFileName(fileValue.name || "upload.bin");
  const contentType = getContentType(fileValue.type);
  const key = `datasets/${voiceId}/${Date.now()}_${crypto.randomUUID()}_${safeFilename}`;

  await c.env.R2.put(key, await fileValue.arrayBuffer(), {
    httpMetadata: {
      contentType,
    },
  });

  return c.json({
    r2_key: key,
    content_type: contentType,
    size: fileValue.size,
  });
});

export default app;
