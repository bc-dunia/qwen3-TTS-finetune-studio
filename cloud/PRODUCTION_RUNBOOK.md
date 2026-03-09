# Qwen3-TTS Production Runbook

This runbook describes the product path that is now expected to work end-to-end on the deployed Cloudflare + RunPod stack.

## Goal

- Train and serve both `0.6B` and `1.7B` voices from the frontend.
- Keep the last known-good voice usable even when a new training run fails validation.
- Promote only checkpoints that pass the quality gate.

## Live Endpoints

- Frontend: `https://40fc1973.qwen-tts-studio.pages.dev`
- API: `https://qwen-tts-api.brian-367.workers.dev`

## Current Ready Voices

- `0.6B`: `1acd69c6-016c-43f7-8ff8-1d571a2402e5`
- `1.7B`: `seo_jaehyung`

## Product Flow

1. Create a voice on the `Voices` page.
2. Upload a real training dataset for that voice.
3. Start training on the `Training` page.
4. The Worker launches a RunPod training pod and records the job in D1.
5. After training finishes, the Worker validates candidate checkpoints.
6. If validation passes, the best checkpoint is promoted and the voice remains `ready`.
7. If validation fails, the new run is marked failed and the previous ready checkpoint is preserved.
8. When validation passes, jump straight into `Voice Detail` or `Playground` from the `Training` page.
9. Generate speech from the `Playground` page, the `Voice Detail` page, or `/v1/text-to-speech/:voice_id`.

## Image Build Path

- The normal product path builds RunPod images in GitHub Actions, not on a local laptop.
- Workflow: `.github/workflows/docker-inference.yml`
- Trigger: push to `main` touching `cloud/runpod/**` or `third_party/Qwen3-TTS/finetuning/**`
- Result: GHCR `latest` and commit-SHA tags are pushed automatically for both inference and training images.
- Local Docker cache cleanup is therefore an operator-debug concern only, not part of the expected production loop.

## Model Defaults

These defaults are now applied by the backend even if the frontend omits them.

### 0.6B

- `batch_size=2`
- `learning_rate=4e-6`
- `num_epochs=7`
- `gradient_accumulation_steps=4`
- `subtalker_loss_weight=0.3`
- `save_every_n_epochs=1`
- `seed=77`
- `gpu_type_id=NVIDIA L40S`
- Validation mode: async, latest checkpoints first

### 1.7B

- `batch_size=2`
- `learning_rate=2e-5`
- `num_epochs=15`
- `gradient_accumulation_steps=4`
- `subtalker_loss_weight=0.3`
- `save_every_n_epochs=5`
- `seed=42`
- `gpu_type_id=NVIDIA L40S`

## Validation Rules

- All post-training checkpoint validation runs asynchronously so the API does not hang on completed jobs.
- `0.6B` validates the newest checkpoints first and promotes the first passing latest checkpoint.
- `1.7B` evaluates candidate checkpoints across its full preset/sample plan asynchronously and promotes the best-scoring passing checkpoint.
- The frontend keeps polling jobs whose `status=completed` but `summary.validation_checked !== true`.
- A ready voice already serving traffic is not cleared just because a newer run failed validation.

## Operational Meaning

- `status=completed` and `validation_checked=true`:
  the trained run was fully reconciled and the promoted checkpoint is final.
- `status=completed` and `validation_checked!=true`:
  validation is still in progress and the frontend should continue polling.
- `status=failed` with `evaluated_checkpoints`:
  the run finished training but no candidate checkpoint met the quality bar.

## Recommended Operator Loop

1. Start a training run from the frontend.
2. Watch the `Training` page until validation completes.
3. If the run passes, test the promoted voice in `Playground`.
4. You can also open the exact voice from the completed job card and generate a sample immediately.
5. If the run fails, inspect `summary.validation_message` and `summary.evaluated_checkpoints`.
6. Adjust dataset or training defaults before retrying.

## Known Current Finding

- The recent `0.6B` run `140c1dd3-c6ad-4420-a5d8-cfd0b431dc33` finished training but failed validation:
  - epoch 6: `overall_score=0.809`
  - epoch 5: `overall_score=0.824`
  - required threshold: `0.900`
- The previous ready `0.6B` checkpoint was kept online.
