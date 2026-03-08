#!/usr/bin/env node

import { appendFileSync, mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";

const API_BASE = process.env.QWEN_TTS_API_URL || "https://qwen-tts-api.brian-367.workers.dev";
const POLL_MS = Number(process.env.QWEN_TTS_QUEUE_POLL_MS || 30000);
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

const logPath = process.argv[2]
  ? resolve(process.argv[2])
  : resolve("workspace/monitoring/training_queue.log");

mkdirSync(dirname(logPath), { recursive: true });

const queue = [
  {
    voiceId: "1acd69c6-016c-43f7-8ff8-1d571a2402e5",
    label: "0.6B",
    datasetName: "curated_clean_v2_20260306",
    tasks: [
      {
        label: "0.6B-reset-a",
        config: {
          model_size: "0.6B",
          batch_size: 2,
          learning_rate: 0.0000025,
          num_epochs: 12,
          gradient_accumulation_steps: 4,
          subtalker_loss_weight: 0.3,
          save_every_n_epochs: 1,
          seed: 303,
          whisper_language: "ko",
          gpu_type_id: "NVIDIA L40S",
        },
      },
      {
        label: "0.6B-reset-b",
        config: {
          model_size: "0.6B",
          batch_size: 2,
          learning_rate: 0.000002,
          num_epochs: 14,
          gradient_accumulation_steps: 4,
          subtalker_loss_weight: 0.32,
          save_every_n_epochs: 1,
          seed: 202,
          whisper_language: "ko",
          gpu_type_id: "NVIDIA L40S",
        },
      },
    ],
  },
  {
    voiceId: "seo_jaehyung",
    label: "1.7B",
    datasetName: "curated_clean_v2_20260306",
    tasks: [
      {
        label: "1.7B-tone-a",
        config: {
          model_size: "1.7B",
          batch_size: 2,
          learning_rate: 0.000005,
          num_epochs: 16,
          gradient_accumulation_steps: 4,
          subtalker_loss_weight: 0.18,
          save_every_n_epochs: 1,
          seed: 202,
          whisper_language: "ko",
          gpu_type_id: "NVIDIA A100-SXM4-80GB",
        },
      },
      {
        label: "1.7B-tone-b",
        config: {
          model_size: "1.7B",
          batch_size: 2,
          learning_rate: 0.000006,
          num_epochs: 14,
          gradient_accumulation_steps: 4,
          subtalker_loss_weight: 0.22,
          save_every_n_epochs: 1,
          seed: 42,
          whisper_language: "ko",
          gpu_type_id: "NVIDIA A100-SXM4-80GB",
        },
      },
    ],
  },
];

const state = new Map(
  queue.map((item) => [
    item.voiceId,
    {
      nextTaskIndex: 0,
      startedJobIds: [],
      lastBusySummary: "",
    },
  ])
);

const sleep = (ms) => new Promise((resolveSleep) => setTimeout(resolveSleep, ms));

const log = (message, data = null) => {
  const line = JSON.stringify({
    ts: new Date().toISOString(),
    message,
    ...(data ? { data } : {}),
  });
  console.log(line);
  appendFileSync(logPath, `${line}\n`);
};

const fetchJson = async (url, init) => {
  const response = await fetch(url, init);
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

const summarizeBusyJobs = (jobs) =>
  jobs.map((job) => {
    const progress = job.progress ?? {};
    const summary = job.summary ?? {};
    const parts = [`${job.job_id.slice(0, 8)}:${job.status}`];
    if (summary.validation_in_progress) {
      parts.push("validating");
    }
    if (typeof progress.epoch === "number" && typeof progress.total_epochs === "number") {
      parts.push(`epoch=${progress.epoch}/${progress.total_epochs}`);
    }
    if (typeof summary.validation_message === "string" && summary.validation_message.trim()) {
      parts.push(summary.validation_message.trim().slice(0, 120));
    }
    return parts.join(" ");
  }).join(" | ");

const startTraining = async (voiceId, datasetName, task) =>
  fetchJson(`${API_BASE}/v1/training/start`, {
    method: "POST",
    headers: {
      "content-type": "application/json",
    },
    body: JSON.stringify({
      voice_id: voiceId,
      dataset_name: datasetName,
      config: task.config,
    }),
  });

const getJobs = async () => fetchJson(`${API_BASE}/v1/training/jobs?limit=100`);

const isBusyJob = (job) => ACTIVE_STATUSES.has(job.status) || job?.summary?.validation_in_progress === true;

log("queue_started", {
  apiBase: API_BASE,
  pollMs: POLL_MS,
  queue: queue.map(({ voiceId, label, tasks }) => ({
    voiceId,
    label,
    tasks: tasks.map((task) => task.label),
  })),
});

while (true) {
  try {
    const payload = await getJobs();
    const jobs = Array.isArray(payload.jobs) ? payload.jobs : [];

    let pendingTasks = 0;
    let busyVoices = 0;

    for (const item of queue) {
      const voiceState = state.get(item.voiceId);
      if (!voiceState) {
        continue;
      }

      const voiceJobs = jobs.filter((job) => job.voice_id === item.voiceId);
      const busyJobs = voiceJobs.filter(isBusyJob);
      const nextTask = item.tasks[voiceState.nextTaskIndex] ?? null;
      pendingTasks += nextTask ? 1 : 0;

      if (busyJobs.length > 0) {
        busyVoices += 1;
        const summary = summarizeBusyJobs(busyJobs);
        if (summary !== voiceState.lastBusySummary) {
          log("voice_busy", {
            voiceId: item.voiceId,
            label: item.label,
            summary,
          });
          voiceState.lastBusySummary = summary;
        }
        continue;
      }

      if (!nextTask) {
        if (voiceState.lastBusySummary !== "idle_done") {
          log("voice_queue_complete", {
            voiceId: item.voiceId,
            label: item.label,
            startedJobIds: voiceState.startedJobIds,
          });
          voiceState.lastBusySummary = "idle_done";
        }
        continue;
      }

      log("starting_task", {
        voiceId: item.voiceId,
        label: item.label,
        task: nextTask.label,
        config: nextTask.config,
      });
      const started = await startTraining(item.voiceId, item.datasetName, nextTask);
      voiceState.nextTaskIndex += 1;
      voiceState.startedJobIds.push(started.job_id);
      voiceState.lastBusySummary = "";
      log("task_started", {
        voiceId: item.voiceId,
        label: item.label,
        task: nextTask.label,
        jobId: started.job_id,
        status: started.status,
      });
    }

    if (pendingTasks === 0 && busyVoices === 0) {
      log("queue_finished", Object.fromEntries(
        [...state.entries()].map(([voiceId, value]) => [voiceId, value.startedJobIds])
      ));
      break;
    }
  } catch (error) {
    log("queue_error", {
      detail: error instanceof Error ? error.message : String(error),
    });
  }

  await sleep(POLL_MS);
}
