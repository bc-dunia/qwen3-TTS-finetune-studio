# Qwen3-TTS Studio — Cloud Deployment

Cloud stack for training + serving fine-tuned Qwen3-TTS voices on Cloudflare and RunPod.

> This repository focuses on **fine-tuning workflows** and long-running optimization loops.
> If you need instant voice cloning UX, use **https://github.com/bc-dunia/qwen3-TTS-studio**.

## What it includes

- Cloudflare Worker API (ElevenLabs-compatible surface)
- React frontend dashboard
- RunPod training + serverless inference integration
- LLM-assisted campaign planning
- Arena-based checkpoint comparison and promotion

## Architecture (high level)

```text
Frontend (Cloudflare Pages)
  -> Worker API (Hono)
     -> D1 / R2 / KV
     -> RunPod (training + inference)
```

## Endpoints (placeholders)

- Worker API: `https://qwen-tts-api.YOUR_SUBDOMAIN.workers.dev`
- Frontend: `https://YOUR_FRONTEND.pages.dev`

## Typical flow

1. Upload dataset and create a voice
2. Start training campaign (manual or autopilot)
3. Evaluate checkpoints (objective metrics + listening)
4. Run Arena to rank candidates
5. Promote winner and serve through API

## References

- Root project docs: [`../README.md`](../README.md)
- Worker API source: [`cloudflare/worker-api`](./cloudflare/worker-api)
- Frontend source: [`cloudflare/frontend`](./cloudflare/frontend)
