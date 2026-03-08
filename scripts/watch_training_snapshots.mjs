#!/usr/bin/env node

import { appendFileSync, mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";

const API_BASE = process.env.QWEN_TTS_API_URL || "https://qwen-tts-api.brian-367.workers.dev";
const logPath = process.argv[2]
  ? resolve(process.argv[2])
  : resolve("workspace/monitoring/training_snapshots.log");
const intervalSeconds = Math.max(60, Number(process.argv[3] || process.env.QWEN_TTS_SNAPSHOT_INTERVAL_SEC || 1200));
const intervalMs = intervalSeconds * 1000;

mkdirSync(dirname(logPath), { recursive: true });

const ACTIVE_STATUSES = new Set([
  "pending",
  "running",
  "provisioning",
  "downloading",
  "preprocessing",
  "preparing",
  "training",
  "uploading",
]);

const sleep = (ms) => new Promise((resolveSleep) => setTimeout(resolveSleep, ms));

const log = (payload) => {
  const line = JSON.stringify({
    ts: new Date().toISOString(),
    ...payload,
  });
  console.log(line);
  appendFileSync(logPath, `${line}\n`);
};

const fetchJson = async (path) => {
  const response = await fetch(`${API_BASE}${path}`);
  const text = await response.text();
  let payload = null;
  try {
    payload = text ? JSON.parse(text) : null;
  } catch {
    payload = text;
  }
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${typeof payload === "string" ? payload : JSON.stringify(payload)}`);
  }
  return payload;
};

const summarizeJob = (job) => ({
  job_id: job.job_id,
  voice_id: job.voice_id,
  status: job.status,
  epoch: job.progress?.epoch ?? null,
  total_epochs: job.progress?.total_epochs ?? null,
  loss: job.progress?.loss ?? null,
  validation_in_progress: job.summary?.validation_in_progress === true,
  validation_checked: job.summary?.validation_checked ?? null,
  validation_passed: job.summary?.validation_passed ?? null,
  selected_checkpoint_epoch: job.summary?.selected_checkpoint_epoch ?? null,
  selected_score: job.summary?.selected_score ?? null,
  message: job.summary?.validation_message || job.summary?.last_message || job.error_message || null,
});

log({
  message: "snapshot_monitor_started",
  apiBase: API_BASE,
  intervalSeconds,
  logPath,
});

while (true) {
  try {
    const [{ jobs }, { voices }] = await Promise.all([
      fetchJson("/v1/training/jobs?limit=20"),
      fetchJson("/v1/voices"),
    ]);

    const activeJobs = (Array.isArray(jobs) ? jobs : []).filter(
      (job) => ACTIVE_STATUSES.has(job.status) || job.summary?.validation_in_progress === true
    );

    const readyVoices = (Array.isArray(voices) ? voices : [])
      .filter((voice) => voice.status === "ready")
      .map((voice) => ({
        voice_id: voice.voice_id,
        name: voice.name,
        model_size: voice.model_size,
        run_name: voice.run_name,
        epoch: voice.epoch,
        checkpoint_r2_prefix: voice.checkpoint_r2_prefix,
        updated_at: voice.updated_at,
      }));

    const recentJobs = (Array.isArray(jobs) ? jobs : []).slice(0, 6).map(summarizeJob);

    log({
      message: "snapshot",
      active_jobs: activeJobs.map(summarizeJob),
      ready_voices: readyVoices,
      recent_jobs: recentJobs,
    });
  } catch (error) {
    log({
      message: "snapshot_error",
      detail: error instanceof Error ? error.message : String(error),
    });
  }

  await sleep(intervalMs);
}
