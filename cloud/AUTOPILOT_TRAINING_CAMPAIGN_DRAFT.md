# Autopilot Training Campaign Draft (Phase 1)

## Goal

Allow users to provide only `attempt_count` (plus optional guardrails), then automatically:

1. Queue training attempts.
2. Reuse preprocess artifacts/snapshots aggressively.
3. Avoid duplicate configs and active collisions.
4. Stop early when quality or budget guardrails trigger.
5. Keep existing promotion safety semantics.

## Existing Behavior We Keep

- `POST /v1/training/start` creates one job and remains unchanged for manual mode.
- `launchQueuedTrainingJobsForVoice` keeps per-voice active limit behavior.
- `training-callbacks` preprocess cache save + dataset snapshot freeze path remains source of truth.
- `training-checkout` status derivation remains source of truth (`promoted`, `candidate_ready`, `kept_current`, `rejected`, `failed`).
- Existing advisor heuristics/LLM path remains optional planner input; no big-bang replacement.
- Campaign Phase 1 does not alter round/adoption semantics and does not auto-promote.

## New API (Campaign Layer)

### Create Campaign

`POST /v1/training/campaigns`

Request:

```json
{
  "voice_id": "uuid",
  "dataset_name": "optional",
  "attempt_count": 6,
  "parallelism": 2,
  "base_config_overrides": {
    "model_size": "1.7B"
  },
  "stop_rules": {
    "max_infra_failures": 2,
    "max_asr_failures": 2,
    "min_score_improvement": 0.005,
    "stagnation_window": 2
  }
}
```

Response:

```json
{
  "campaign_id": "uuid",
  "voice_id": "uuid",
  "status": "planning",
  "attempt_count": 6,
  "attempts_planned": 0,
  "attempts_completed": 0,
  "active_jobs": 0,
  "queued_jobs": 0,
  "best_job_id": null,
  "created_at": 0,
  "updated_at": 0
}
```

### Get Campaign

`GET /v1/training/campaigns/:campaign_id`

Returns campaign summary + ordered attempts (job refs, lane, status, score, taxonomy, config key).

### Cancel Campaign

`POST /v1/training/campaigns/:campaign_id/cancel`

Marks campaign as cancelled, no new attempts planned, existing jobs continue or can be cancelled manually.

## Data Model (Minimal Migration)

### New Table

`training_campaigns`

- `campaign_id` (PK)
- `voice_id`
- `dataset_name`
- `dataset_r2_prefix`
- `dataset_snapshot_id` (nullable until resolved)
- `attempt_count`
- `parallelism`
- `status` (`planning|running|completed|failed|blocked_dataset|blocked_budget|cancelled`)
- `base_config_json`
- `stop_rules_json`
- `planner_state_json`
- `summary_json`
- `created_at`, `updated_at`, `completed_at`

### Existing Table Extensions

`training_jobs`

- add nullable `campaign_id`
- add nullable `attempt_index`

No required changes to `training_rounds` shape for Phase 1.

## State Machine

Campaign:

- `planning` -> `running`
- `running` -> `completed|blocked_dataset|blocked_budget|failed|cancelled`

Attempt lifecycle uses existing job states.

Planner loop trigger points:

1. campaign created
2. preprocess cache callback persisted
3. job reaches terminal state
4. periodic supervisor sweep fallback

## Reuse-First Scheduling Policy

1. Resolve/freeze dataset snapshot once per campaign when first attempt is created.
2. If preprocess cache missing:
   - schedule one builder attempt first,
   - enqueue siblings with `queue_wait_reason=waiting_for_preprocess_artifacts`.
3. On cache-save callback, wake campaign planner and launch queued siblings.
4. Config dedupe key = `dataset_snapshot_id + normalized_config_key`.
5. If exact config already validated/rejected for same snapshot, skip planning that variant.
6. Infra-only failure may retry exact config once; quality failures require lane shift.
7. Phase 1 creates attempts just-in-time. It does not pre-create all queued attempts.

## Good/Bad Signal Exploitation

Use deterministic signals for Phase 1 and keep richer scoring for later phases.

- Positive: `promoted`, `candidate_ready`, higher selected/candidate/champion score.
- Neutral: `kept_current`.
- Negative: `rejected`, `failed`.
- Taxonomy penalty: `asr|tone|speed|overall|infra` from advisor signal extraction.

Suggested lane score delta (Phase 2+):

- `promoted +0.03`
- `candidate_ready +0.02`
- `kept_current +0.00`
- `rejected -0.05`
- `failed -0.10`
- penalty: `asr -0.06`, `tone -0.03`, `speed -0.02`, `overall -0.02`, `infra -0.01`

Planner policy:

- ASR-heavy failures move to dataset-focused or stop.
- tone/speed-heavy failures shift lane and seed.
- if active collision, diversify config immediately.
- Phase 1 uses only deterministic failure caps and attempt budget.

## Guardrails and Early Stop

- `max_parallel_jobs = min(requested_parallelism, TRAINING_MAX_ACTIVE_JOBS_PER_VOICE)`
- budget cap: `attempt_count`
- early stop:
  - `asr_failures >= max_asr_failures`
  - `infra_failures >= max_infra_failures`
  - other heuristic stop rules are deferred to Phase 2+

## Implementation Plan

### Phase 1 (now)

1. Add schema and D1 accessors for campaigns.
2. Add campaign endpoints.
3. Add planner coordinator that creates one new attempt when capacity allows.
4. Persist campaign/attempt metadata into campaign summary + job summary.
5. Keep round/adoption untouched for campaign attempts (no `active_round_id` mutation from campaign path).
6. Wire planner sweep in scheduled supervisor flow.

### Phase 2

1. Frontend campaign card in Training page.
2. Campaign status polling UI.
3. Stop/cancel UX.

### Phase 3

1. Optional LLM tie-break planner (heuristic default remains mandatory fallback).
2. Improved budget model (runtime estimate caps).

## Rollback Safety

- Manual path untouched: `/v1/training/start` continues to work independently.
- Campaign logic is additive and can be disabled by route flag removal.
- Promotion remains governed by existing validated outcome logic.
