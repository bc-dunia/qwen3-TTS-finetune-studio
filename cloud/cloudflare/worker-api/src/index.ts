import { Hono } from "hono";
import { cors } from "hono/cors";
import datasetRoutes from "./routes/dataset";
import modelsRoutes from "./routes/models";
import trainingCallbacksRoutes from "./routes/training-callbacks";
import trainingRoutes from "./routes/training";
import ttsRoutes from "./routes/tts";
import uploadRoutes from "./routes/upload";
import voicesRoutes from "./routes/voices";
import type { AppContext } from "./types";

const app = new Hono<AppContext>();

app.use("*", async (c, next) => {
  const origin = c.env.CORS_ORIGIN || "*";
  return cors({
    origin,
    allowMethods: ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allowHeaders: ["Content-Type", "Authorization", "xi-api-key"],
  })(c, next);
});

app.get("/", (c) => c.json({ status: "ok" }));

app.route("/v1/text-to-speech", ttsRoutes);
app.route("/v1/voices", voicesRoutes);
app.route("/v1/models", modelsRoutes);
app.route("/v1/training", trainingRoutes);
app.route("/v1/internal/training", trainingCallbacksRoutes);
app.route("/v1/upload", uploadRoutes);
app.route("/v1/dataset", datasetRoutes);

app.onError((err, c) => {
  if (err instanceof SyntaxError) {
    return c.json({ detail: { status: "error", message: "Invalid JSON in request body" } }, 400);
  }
  console.error("Unhandled error:", err);
  // Pass through useful error messages from upstream services (RunPod, R2, etc.)
  const message = err instanceof Error ? err.message : "Internal server error";
  // Detect RunPod supply constraint errors
  if (message.includes("no longer any instances available") || message.includes("SUPPLY_CONSTRAINT")) {
    return c.json({ detail: { status: "error", message: "No GPU instances currently available. Please try again later or use a different GPU type.", gpu_error: message } }, 503);
  }
  return c.json({ detail: { status: "error", message } }, 500);
});

export default app;
