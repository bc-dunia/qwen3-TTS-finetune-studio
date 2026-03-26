# Qwen3-TTS Studio — Cloud Deployment

Cloud-native voice cloning platform built on Cloudflare and RunPod. The Worker API exposes an ElevenLabs-compatible REST interface backed by a React dashboard, with training fully automated through a 3-layer auto-adjusting algorithm that runs multi-attempt campaigns without manual intervention. Training jobs are queued, dispatched, and scored automatically; the system picks the best checkpoint and advances or terminates the campaign based on quality signals.

**Key capabilities:**

- ElevenLabs-compatible API — drop-in replacement for any ElevenLabs SDK client
- Autopilot campaign system with goal-first UI (get first checkpoint / improve best score / fix failed campaign)
- 3-layer auto-adjustment: heuristic advisor, LLM advisor (gpt-5.4), and 3-lane campaign planner
- Gated weighted checkpoint scoring (ASR, speaker similarity, tone, speed, health)
- Global queue with atomic job claiming, per-stage stale detection, and preprocess artifact reuse
- React dashboard with dark/light theme, live job polling, and per-voice training workspace
- Serverless inference via RunPod with scale-to-zero

---

## Architecture

```
Cloudflare Pages (React Frontend)
         |
         v
Cloudflare Workers (Hono API — ElevenLabs-compatible)
    /      |      |      \
   v       v      v       v
  R2      D1     KV     RunPod
(storage) (db) (cache)  /      \
                    Training  Serverless
                    Pods      Inference
```

- **Cloudflare Pages** — React SPA, deployed via `wrangler pages deploy`
- **Cloudflare Workers** — Hono-based API router; handles auth, TTS proxying, training orchestration, campaign management, and the cron sweep
- **R2** — stores checkpoints, audio samples, datasets, and presigned upload/download URLs
- **D1** — SQLite-backed relational store for voices, training jobs, campaigns, and checkpoint scores
- **KV** — TTS audio cache keyed by voice + text hash
- **RunPod Training Pods** — on-demand GPU pods launched per training job via the RunPod API
- **RunPod Serverless** — always-available (scale-to-zero) inference endpoint for TTS generation

---

## Training Pipeline

The training pipeline is the core of the cloud product. It covers everything from dataset upload through checkpoint scoring and campaign progression.

### Autopilot Campaign System

A campaign is a multi-attempt training plan for a single voice. Instead of configuring hyperparameters directly, the user picks a goal:

| Goal | Direction | Behavior |
|------|-----------|----------|
| Get first checkpoint | `balanced` | Mix safe retries with moderate exploration |
| Improve best score | `conservative` | Favor proven presets and stable quality |
| Fix failed campaign | `exploratory` | Search aggressively for better checkpoints |

The direction maps to lane weights in the campaign planner (see below). Campaigns have an attempt budget (1-12) and a parallelism setting that controls how many jobs run concurrently within the campaign.

**Campaign state machine:**

```
planning -> running -> completed
                    -> failed
                    -> blocked_dataset   (ASR failures dominate; dataset needs fixing)
                    -> blocked_budget    (attempt budget exhausted without a passing checkpoint)
                    -> cancelled         (user-initiated)
```

The cron sweep (`*/2 * * * *`) advances campaigns by checking terminal jobs, scoring checkpoints, and dispatching the next round of attempts. Immediate dispatch also fires via `waitUntil` callback when a job completes, so campaigns progress without waiting for the next cron tick.

### 3-Layer Auto-Adjustment Algorithm

Each campaign round, the system generates candidate training configs through three layers:

**Layer 1: Heuristic Advisor (`training-advisor.ts`)**

Analyzes job history and emits one of six modes:

| Mode | When used |
|------|-----------|
| `tone-explore` | Lower LR + seed rotation to preserve speaker phrasing |
| `stability-reset` | Conservative config to recover from repeated failures |
| `checkpoint-window` | Short sweep with `save_every_n_epochs=1` to find the sweet spot epoch |
| `compare-first` | Scores are too close; user should listen before spending another run |
| `dataset-first` | ASR failures dominate; fix dataset before retraining |
| `hold-current` | Current champion is stable; no urgent need for another run |

**Layer 2: LLM Advisor (`training-advisor-llm.ts`)**

Calls gpt-5.4 with `reasoning: { effort: "high" }` to produce a richer analysis of job history and recommend the next config. Falls back to the heuristic advisor if the LLM call fails or returns invalid JSON.

**Layer 3: Campaign Planner (`campaign-planner.ts`)**

Generates concrete training configs using a 3-lane strategy. Each lane targets a different part of the search space:

| Lane | Purpose |
|------|---------|
| `exploit` | Refine the best known config with small perturbations |
| `repair` | Recover from failure patterns (high ASR failure rate, infra issues) |
| `explore` | Try substantially different hyperparameter regions |

Lane weights by direction:

| Direction | exploit | repair | explore |
|-----------|---------|--------|---------|
| conservative | 70% | 25% | 5% |
| balanced | 50% | 30% | 20% |
| exploratory | 25% | 35% | 40% |

**Phase detection** adjusts weights further:

| Phase | Condition | Effect |
|-------|-----------|--------|
| `bootstrap` | No completed jobs yet | Shifts toward repair + explore |
| `searching` | Jobs completed, no passing checkpoint | Normal lane weights |
| `exploiting` | At least one passing checkpoint exists | Shifts toward exploit |
| `infeasible` | Budget exhausted, no passing checkpoint | Campaign terminates |

The planner also tracks exclusion zones (configs that have already been tried), family blocking (avoid configs too similar to known failures), and near-miss signals (configs that almost passed the gate).

### Quality Validation (Checkpoint Scoring)

Every checkpoint produced by a training job is scored against a set of evaluation samples. Scoring uses a two-stage gated system with an optional calibration overlay from the Voice Arena.

**Hard safety gates** — non-negotiable minimums that protect against broken checkpoints:

| Metric | Minimum | Bypassed by calibration? |
|--------|---------|--------------------------|
| ASR similarity | 0.75 | Never |
| Speaker similarity | 0.70 | Never |
| Health score | 0.65 | Never |

**Full validation gates** — used for standard checkpoint filtering:

| Metric | Minimum | Role |
|--------|---------|------|
| ASR similarity | 0.82 | Hard |
| Speaker similarity | 0.78 | Hard |
| Health score | 0.72 | Hard |
| Tone similarity | 0.50 | Soft |
| Style similarity | 0.55 | Soft |
| Speed similarity | 0.15 | Soft |
| Overall score | 0.80 | Soft |
| Duration accuracy | 0.50 | Soft |

**Ranking weights** — applied after gate passage to rank checkpoints. When arena calibration is `active`, these are blended with learned weights from human preference data:

| Metric | Default Weight |
|--------|---------------|
| ASR | 0.25 |
| Speaker | 0.25 |
| Style | 0.20 |
| Tone | 0.05 |
| Speed | 0.05 |
| Overall | 0.05 |
| Duration | 0.05 |

**ASR review (`review-asr.ts`)** — runs Whisper on generated audio and computes Levenshtein similarity against the target transcript.

**LLM transcript review (`transcript-review.ts`)** — calls Workers AI (`@cf/meta/llama-3.1-8b-instruct`) for transcript quality analysis, with OpenAI (`gpt-5-mini`) as fallback.

**Multi-preset evaluation** — each checkpoint is evaluated across multiple generation presets. A pass rate gate filters out checkpoints that only work on one preset.

### Queue Management

The queue system coordinates job dispatch across all voices and campaigns.

- **Global limit:** 3 concurrent pods (`MAX_CONCURRENT_PODS`)
- **Per-voice limit:** 1 active job by default, with spillover to idle global capacity when no other voices are queued
- **Priority tiers:** manual runs > first-checkpoint campaigns > improvement campaigns
- **Dispatch:** immediate via `waitUntil` callback on job completion; cron sweep every 2 minutes as recovery
- **Atomic job claiming:** D1 transactions prevent race conditions when multiple cron invocations overlap
- **Stale job detection:** per-stage timeouts (e.g., provisioning: 4 min, validation: 6 min) mark stuck jobs as failed
- **Preprocess artifact reuse:** sibling jobs sharing the same dataset snapshot reuse the preprocessing output rather than re-running it

---

## Training Domain Module (`training-domain.ts`)

`cloud/cloudflare/worker-api/src/lib/training-domain.ts` is the single source of truth for all training configuration. Every other module imports from here.

**Contents:**

- **Config defaults** — `DEFAULTS_0_6B` and `DEFAULTS_1_7B` presets with batch size, learning rate, epochs, gradient accumulation, subtalker loss weight, save frequency, seed, and GPU type
- **`getTrainingDefaults(modelSize)`** — returns the correct preset for a given model size
- **`sanitizeConfig(source, modelSize, language)`** — merges user overrides onto defaults with type-safe coercion
- **Validation thresholds** — `VALIDATION_GATE_THRESHOLDS` (full gate), `HARD_SAFETY_THRESHOLDS` (anti-garbage floor), `SOFT_PREFERENCE_THRESHOLDS` (calibration-adjustable)
- **`passesValidationGate(scores)`** — checks all hard minimums (full gate)
- **`passesHardSafetyGate(scores)`** — checks only ASR/speaker/health (never bypassed by calibration)
- **`computeRankingScore(scores, weights?)`** — weighted sum for checkpoint ranking; accepts optional calibrated weights
- **`computeCalibrationAlpha(matchupCount)`** — returns alpha blend factor based on arena data volume
- **`getEffectiveWeights(defaults, learned, alpha)`** — blends default and learned weights with per-weight shift cap
- **`grayZoneRescue(scores, weights)`** — rescues checkpoints that barely miss one soft threshold
- **Queue constants** — `MAX_CONCURRENT_PODS`, `DEFAULT_MAX_ACTIVE_TRAINING_JOBS_PER_VOICE`, `ACTIVE_JOB_STATUSES`, `TERMINAL_JOB_STATUSES`
- **Parsing helpers** — `readNumber`, `readText`, `readTimestamp`, `clamp`
- **Path helpers** — `stripSlashes`, `parseRunNameFromCheckpointPrefix`, `extractDatasetNameFromPrefix`

---

## Frontend Architecture

The React SPA is deployed to Cloudflare Pages. It uses React Router v7, Tailwind CSS v4 (CSS-first config via `@theme` directive), and a dark-first theme with full light mode support (`data-theme="light"`).

**Page routes:**

```
/           Dashboard — voice stats, active training jobs with live polling, queued count
/voices     Voice list with inline Train links
/voices/:voiceId
  /generate   TTS generation
  /training   VoiceTrainingTab (AutopilotPanel + active jobs + history)
  /dataset    Dataset management
  /compare    Checkpoint comparison (decomposed UI with score bands, filter tabs, sticky compare tray)
  /arena      Voice Arena — blind A/B checkpoint tournament
/playground Quick TTS playground
/queue      Global queue monitor (real pod capacity, per-voice stats)
```

**Smart default tab:** voices without a checkpoint open on the training tab; trained voices open on the generate tab.

**Key components:**

- `AutopilotPanel` — unified goal-first training interface. Merges the former `AutopilotCard` and `TrainingAdviceCard` into a single component. Exposes goal selector (first checkpoint / improve score / fix failed), attempt count, parallelism, and base config overrides.
- `TrainingJobRow` — single job status row with live progress
- `TrainingHistoryList` — paginated job history with checkpoint scores
- `QueuePage` — global queue view showing real pod capacity and per-voice queue depth
- `VoiceCompare` — checkpoint comparison page decomposed into focused sub-components (see Compare Components below)

**Compare Components (`components/compare/`):**

The checkpoint comparison page was refactored from a 1159-line monolith into composable sub-components:

- `DecisionHeader` — compact Trusted vs Recommended comparison header with score bands and one-line comparison summary
- `ComparisonSetupForm` — comparison text and seed always visible; stability/similarity/style/speed sliders and prompt controls collapse into an "Advanced Settings" accordion
- `CandidateGrid` — filterable grid (All / Passed / Rejected tabs) with quick-selection presets (Trusted + Recommended, Recommended + Rejected, Clear)
- `CandidateCard` — dual-mode card: browse mode (selection, score band pill, role badges, validation message) and listening mode (AudioPlayer, trait bars, Apply button)
- `ScoreBand` — visual score grade pill (Excellent / Strong / Borderline / Weak) replacing raw numeric display
- `TraitBars` — horizontal mini-bars for style, tone, and speed sub-scores
- `CompareTray` — sticky bottom bar showing selected checkpoint chips with a Generate CTA
- `RecommendationBanner` — post-generation decision support comparing trusted vs recommended scores with Promote / Keep Current actions
- `RunDetailsDisclosure` — collapsible expert-detail panel (run IDs, epochs, timestamps, raw scores)

---

## Current Stack (Mar 2026)

- **Worker API:** `https://qwen-tts-api.brian-367.workers.dev`
- **Frontend:** `https://qwen-tts-studio.pages.dev`
- **Cron schedule:** `*/2 * * * *` (training supervisor sweep + campaign progression)
- **D1 database:** `qwen-tts-db`
- **R2 bucket:** `qwen-tts-studio`
- **KV namespace:** `TTS_CACHE`
- **Runtime:** Wrangler v3 (`wrangler deploy`)

---

## File Structure

```
cloud/
├── README.md
├── runpod/
│   ├── Dockerfile.training            # GPU training pod image
│   ├── Dockerfile.inference           # Serverless inference image
│   ├── r2_storage.py                  # R2 client (shared by both handlers)
│   ├── training_handler.py            # Training orchestrator (runs on pod)
│   └── inference_handler.py           # TTS generation handler (serverless)
└── cloudflare/
    ├── worker-api/
    │   ├── wrangler.toml              # Worker configuration
    │   ├── schema.sql                 # D1 database schema
    │   ├── schema-arena.sql           # Arena tables (candidates, matches, sessions, calibration)
    │   ├── schema-arena-migration-001.sql  # Calibration state machine columns
    │   └── src/
    │       ├── index.ts               # Main Hono router
    │       ├── types.ts               # TypeScript types
    │       ├── middleware/auth.ts      # API key auth
    │       ├── routes/
    │       │   ├── tts.ts             # ElevenLabs-compatible TTS endpoint
    │       │   ├── voices.ts          # Voice CRUD
    │       │   ├── models.ts          # Model listing
    │       │   ├── training.ts        # Core training orchestration (7000+ lines)
    │       │   ├── training-callbacks.ts  # RunPod callback handler
    │       │   ├── dataset.ts         # Dataset management
    │       │   ├── upload.ts          # Presigned URL generation
    │       │   ├── arena.ts           # Voice Arena API (blind A/B tournament)
    │       │   └── admin.ts           # Admin utilities
    │       └── lib/
    │           ├── training-domain.ts     # Canonical config, validation, helpers
    │           ├── training-advisor.ts    # Heuristic advisor (6 modes)
    │           ├── training-advisor-llm.ts  # LLM advisor (gpt-5.4 with high-effort reasoning)
    │           ├── campaign-planner.ts    # 3-lane campaign strategy
    │           ├── training-checkout.ts   # Checkpoint selection logic
    │           ├── review-asr.ts          # ASR similarity scoring (Levenshtein via Whisper)
    │           ├── transcript-review.ts   # LLM transcript quality review
    │           ├── runpod.ts              # RunPod API client
    │           ├── r2.ts                  # R2 presigned URLs
    │           ├── d1.ts                  # D1 query helpers
    │           ├── arena.ts               # Arena match logic, Swiss/round-robin tournaments
    │           └── arena-calibration.ts   # Arena calibration utilities
    └── frontend/
        ├── package.json
        ├── vite.config.ts
        ├── tailwind.config.ts
        └── src/
            ├── main.tsx
            ├── App.tsx
            ├── index.css                  # Dark/light theme variables (@theme directive)
            ├── lib/
            │   ├── api.ts                 # API client + TypeScript types
            │   ├── training-domain.ts     # Frontend canonical config (mirrors worker)
            │   ├── trainingAdvisor.ts     # Client-side advice rendering
            │   ├── trainingCheckout.ts    # Checkpoint selection helpers
            │   └── voiceScoreUi.ts        # Score formatting, color, and band utilities
            ├── hooks/
            │   └── useTheme.ts
            ├── pages/
            │   ├── Dashboard.tsx
            │   ├── Voices.tsx
            │   ├── VoiceWorkspace.tsx     # Smart default tab routing
            │   ├── VoiceDetail.tsx
            │   ├── VoiceTrainingTab.tsx
            │   ├── VoiceDataset.tsx
            │   ├── VoiceCompare.tsx       # Orchestrator using compare/* components
            │   ├── VoiceArena.tsx         # Blind A/B checkpoint tournament
            │   ├── ArenaCalibration.tsx   # Arena calibration dashboard
            │   ├── Playground.tsx
            │   └── QueuePage.tsx
            └── components/
                ├── Layout.tsx
                ├── AudioPlayer.tsx            # Canvas waveform with play/pause/seek
                ├── VoiceCard.tsx
                ├── TrainingAdviceCard.tsx
                ├── compare/                   # Checkpoint comparison sub-components
                │   ├── types.ts               # Shared types (CheckpointCandidate, etc.)
                │   ├── utils.ts               # getCandidateTitle helper
                │   ├── ScoreBand.tsx           # Score grade pill + TraitBars
                │   ├── RunDetailsDisclosure.tsx  # Collapsible debug detail
                │   ├── DecisionHeader.tsx      # Trusted vs Recommended header
                │   ├── ComparisonSetupForm.tsx  # Text/seed + advanced accordion
                │   ├── CandidateCard.tsx       # Dual-mode browse/listening card
                │   ├── CandidateGrid.tsx       # Filterable candidate grid
                │   ├── CompareTray.tsx         # Sticky bottom selection bar
                │   └── RecommendationBanner.tsx  # Post-generation decision support
                └── training/
                    ├── AutopilotPanel.tsx      # Goal-first training UI
                    ├── AutopilotCard.tsx
                    ├── TrainingJobRow.tsx
                    └── TrainingHistoryList.tsx
```

---

## Quick Start

### Prerequisites

- [Cloudflare account](https://dash.cloudflare.com) (already logged in)
- [RunPod account](https://runpod.io) (already signed up)
- Node.js 20+, npm, Python 3.11+
- Docker (for building RunPod images)

### 1. Cloudflare Setup

```bash
cd cloud/cloudflare/worker-api

# Install dependencies
npm install

# Login to Cloudflare (if not already)
npx wrangler login

# Create R2 bucket
npx wrangler r2 bucket create qwen-tts-studio

# Create D1 database
npx wrangler d1 create qwen-tts-db
# Copy the database_id from output into wrangler.toml

# Run database migrations
cd cloud/cloudflare/worker-api
npm run db:migrate

# Create KV namespace
npx wrangler kv namespace create TTS_CACHE
# Copy the namespace id into wrangler.toml

# Create R2 API token (for presigned URLs)
# Go to: Cloudflare Dashboard > R2 > Manage R2 API Tokens > Create API Token
# Copy the access key ID and secret key

# Set secrets
npx wrangler secret put API_KEY              # Your API key for client auth
npx wrangler secret put RUNPOD_API_KEY       # RunPod API key
npx wrangler secret put R2_ACCESS_KEY_ID     # R2 S3-compatible access key
npx wrangler secret put R2_SECRET_ACCESS_KEY # R2 S3-compatible secret key

# Update wrangler.toml with:
# - D1 database_id
# - KV namespace id
# - R2_ENDPOINT_URL (https://<account_id>.r2.cloudflarestorage.com)
# - RUNPOD_ENDPOINT_ID (after creating RunPod endpoint)
# - RUNPOD_TRAINING_TEMPLATE_ID (after creating RunPod template)
# - TRAINING_MAX_ACTIVE_JOBS_PER_VOICE (default 1; controls per-voice concurrency)

# Deploy
npx wrangler deploy
```

### 2. RunPod Setup

#### Build Docker Images

Production path:
- Do not rely on a developer laptop to build/push the RunPod images.
- Pushing to `main` automatically runs `.github/workflows/docker-inference.yml`, which builds and pushes both the inference and training images to GHCR on an ephemeral GitHub runner.
- That workflow already frees runner disk before the training image build, so local Docker/Colima cache cleanup is only for emergency debugging, not the normal product path.

```bash
# From project root
cd /path/to/qwen3-tts-finetune-studio

# Local builds are for debugging only.
# Build training image
docker build -f cloud/runpod/Dockerfile.training -t qwen3-tts-training .

# Build inference image (downloads ~10GB of model weights)
docker build -f cloud/runpod/Dockerfile.inference -t qwen3-tts-inference .

# Push to Docker Hub (or GHCR)
docker tag qwen3-tts-training YOUR_REGISTRY/qwen3-tts-training:latest
docker tag qwen3-tts-inference YOUR_REGISTRY/qwen3-tts-inference:latest
docker push YOUR_REGISTRY/qwen3-tts-training:latest
docker push YOUR_REGISTRY/qwen3-tts-inference:latest
```

Automatic GHCR build triggers:
- `cloud/runpod/**`
- `third_party/Qwen3-TTS/finetuning/**`
- `.github/workflows/docker-inference.yml`

#### Create RunPod Training Template

1. Go to RunPod > Templates > New Template
2. Settings:
   - **Name**: qwen3-tts-training
   - **Container Image**: YOUR_REGISTRY/qwen3-tts-training:latest
   - **Docker Command**: `python3 -u training_handler.py`
   - **Environment Variables**:
     - `R2_ENDPOINT_URL`: `https://<account_id>.r2.cloudflarestorage.com`
     - `R2_ACCESS_KEY_ID`: (your R2 access key)
     - `R2_SECRET_ACCESS_KEY`: (your R2 secret key)
     - `R2_BUCKET`: `qwen-tts-studio`
     - `RUNPOD_API_KEY`: (your RunPod API key)
   - **Volume Mount**: `/runpod-vol` (optional, for model cache)
3. Copy the template ID into `wrangler.toml` as `RUNPOD_TRAINING_TEMPLATE_ID`

#### Create RunPod Serverless Endpoint

1. Go to RunPod > Serverless > New Endpoint
2. Settings:
   - **Name**: qwen3-tts-inference
   - **Container Image**: YOUR_REGISTRY/qwen3-tts-inference:latest
   - **GPU**: RTX 4090 (24GB) — inference uses ~4GB VRAM
   - **Min Workers**: 0 (scale to zero)
   - **Max Workers**: 3 (or as needed)
   - **Idle Timeout**: 300s (keep warm for 5 min)
   - **Environment Variables**:
     - `R2_ENDPOINT_URL`: `https://<account_id>.r2.cloudflarestorage.com`
     - `R2_ACCESS_KEY_ID`: (your R2 access key)
     - `R2_SECRET_ACCESS_KEY`: (your R2 secret key)
     - `R2_BUCKET`: `qwen-tts-studio`
3. Copy the endpoint ID into `wrangler.toml` as `RUNPOD_ENDPOINT_ID`
4. Redeploy the Worker: `npx wrangler deploy`

### 3. Frontend Deployment

```bash
cd cloud/cloudflare/frontend

# Install dependencies
npm install

# Set API URL
echo "VITE_API_URL=https://qwen-tts-api.YOUR_SUBDOMAIN.workers.dev" > .env

# Build
npm run build

# Deploy to Cloudflare Pages
npx wrangler pages deploy dist --project-name qwen-tts-studio --branch main
```

---

## API Usage

The API is ElevenLabs-compatible. Any ElevenLabs client SDK works by changing the base URL.

### Generate Speech

```bash
curl -X POST "https://qwen-tts-api.YOUR_SUBDOMAIN.workers.dev/v1/text-to-speech/VOICE_ID" \
  -H "xi-api-key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, how are you?"}' \
  --output speech.wav
```

### List Voices

```bash
curl "https://qwen-tts-api.YOUR_SUBDOMAIN.workers.dev/v1/voices" \
  -H "xi-api-key: YOUR_API_KEY"
```

### Start Training

```bash
curl -X POST "https://qwen-tts-api.YOUR_SUBDOMAIN.workers.dev/v1/training/start" \
  -H "xi-api-key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "voice_id": "VOICE_ID",
    "dataset_name": "my_dataset",
    "config": {
      "batch_size": 2,
      "learning_rate": 2e-5,
      "num_epochs": 3,
      "model_size": "1.7B"
    }
  }'
```

### Check Training Status

```bash
curl "https://qwen-tts-api.YOUR_SUBDOMAIN.workers.dev/v1/training/JOB_ID" \
  -H "xi-api-key: YOUR_API_KEY"
```

### Python Client (ElevenLabs SDK compatible)

```python
from elevenlabs import ElevenLabs

client = ElevenLabs(
    api_key="YOUR_API_KEY",
    base_url="https://qwen-tts-api.YOUR_SUBDOMAIN.workers.dev"
)

# Generate speech
audio = client.text_to_speech.convert(
    voice_id="VOICE_ID",
    text="Hello, this is my custom voice!",
    model_id="qwen3-tts-1.7b",
)

with open("output.wav", "wb") as f:
    for chunk in audio:
        f.write(chunk)
```

---

## Training Modes

### Single Training Job

Launches one training job immediately with explicit config overrides. Useful for manual experimentation or when you know exactly what config to run.

```bash
curl -X POST "https://qwen-tts-api.brian-367.workers.dev/v1/training/start" \
  -H "xi-api-key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "voice_id": "VOICE_ID",
    "config": {
      "model_size": "1.7B",
      "batch_size": 2,
      "learning_rate": 2e-5,
      "num_epochs": 3,
      "gradient_accumulation_steps": 4,
      "save_every_n_epochs": 1,
      "seed": 303,
      "whisper_language": "ko"
    }
  }'
```

### Autopilot Campaign

Creates a multi-attempt training plan. Campaign state is tracked in D1 and progressed by the cron sweep and `waitUntil` callbacks. The 3-layer advisor system selects configs for each round automatically.

```bash
curl -X POST "https://qwen-tts-api.brian-367.workers.dev/v1/training/campaigns" \
  -H "xi-api-key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "voice_id": "VOICE_ID",
    "attempt_count": 6,
    "parallelism": 3,
    "base_config_overrides": {
      "model_size": "1.7B",
      "batch_size": 2,
      "learning_rate": 2e-5,
      "num_epochs": 8,
      "seed": 42,
      "whisper_language": "ko"
    },
    "stop_rules": {
      "max_asr_failures": 2,
      "max_infra_failures": 2,
      "min_score_improvement": 0.005,
      "stagnation_window": 2
    }
  }'
```

Poll or cancel a campaign:

```bash
curl "https://qwen-tts-api.brian-367.workers.dev/v1/training/campaigns/CAMPAIGN_ID" \
  -H "xi-api-key: YOUR_API_KEY"

curl -X POST "https://qwen-tts-api.brian-367.workers.dev/v1/training/campaigns/CAMPAIGN_ID/cancel" \
  -H "xi-api-key: YOUR_API_KEY"
```

---

## Cost Estimates

### Cloudflare (monthly)

| Service | Usage | Cost |
|---------|-------|------|
| R2 Storage | 50GB checkpoints + 100GB audio | ~$2.25 |
| R2 Egress | Unlimited | $0 (free) |
| Workers | API proxy (10M requests included) | $5 base |
| D1 | Metadata | ~$0 (included) |
| Pages | Frontend hosting | $0 (free) |
| **Total** | | **~$7-10/month** |

### RunPod (usage-based)

| Task | GPU | Rate | Example |
|------|-----|------|---------|
| Training (1.7B) | A100-40GB spot | ~$0.40/hr | 2hr run = $0.80 |
| Training (0.6B) | RTX 4090 spot | ~$0.25/hr | 1hr run = $0.25 |
| Inference | RTX 4090 serverless | ~$0.00053/sec | 100 req/day = ~$0.27/day |
| Storage | Network Volume | $0.07/GB/mo | 100GB = $7/mo |

### Total: ~$25-35/month (moderate usage)

---

## GPU Recommendations

| Use Case | GPU | VRAM | Reason |
|----------|-----|------|--------|
| Training 1.7B | A100-40GB | 40GB | Full SFT + AdamW states ~20-24GB |
| Training 0.6B | RTX 4090 | 24GB | Full SFT fits in 24GB |
| Inference 1.7B | RTX 4090 | 24GB | Model uses ~4GB VRAM |
| Inference 0.6B | RTX 4090 | 24GB | Model uses ~2.5GB VRAM |

---

## Troubleshooting

### Training pod doesn't start

- Check RunPod dashboard for pod status
- Verify R2 credentials are set correctly in the template
- Check `jobs/{job_id}/status.json` in R2 for error messages

### Campaign stuck in `planning` or `running`

- The cron sweep runs every 2 minutes; wait one cycle and check again
- If a job is stale (stuck in provisioning or validation beyond the per-stage timeout), the sweep will mark it failed and dispatch the next attempt
- Check D1 directly via the Cloudflare dashboard for campaign and job rows

### Campaign terminates as `blocked_dataset`

- ASR failures are dominating the campaign; the dataset likely has transcript mismatches or audio quality issues
- Re-upload the dataset with corrected transcripts and restart the campaign

### Inference cold start is slow

- First request per voice downloads the checkpoint from R2 (~3-5GB)
- Increase the serverless idle timeout to keep warm instances longer
- Consider using a RunPod Network Volume for checkpoint caching

### Audio quality issues

- Ensure training data is 24kHz mono WAV
- Upload the full training set, not just one reference clip
- Prefer Qwen3-TTS 0.6B for cloud fine-tuning and serving unless you have a strong reason to use 1.7B
- Use 3-15 second clean clips from one speaker, ideally 10+ minutes total
- Keep one stable reference clip and reference transcript when building curated datasets
- The cloud pipeline validates checkpoints using ASR similarity plus reference-based speaker/tone/speed similarity when reference metadata is available
- Use conservative voice settings first; raise style only after the similarity baseline is stable
- Use the Quality tab in the local Gradio UI for pre-training checks

## Voice Arena

The Voice Arena is a blind A/B tournament system for comparing TTS checkpoints using human preference. Instead of relying solely on automated scores, users listen to anonymized audio samples and vote on which sounds better.

### How It Works

1. **Start Arena** — assembles candidates from the voice's current champion, previously carried 2nd/3rd place checkpoints, and new checkpoints from recent training jobs
2. **Audio Generation** — fires async RunPod TTS requests for every candidate × test text combination; frontend polls for progress
3. **Blind Voting** — candidates are shuffled into pairwise matches; the user hears "Sample 1" vs "Sample 2" without knowing which checkpoint produced each. Four vote options: Sample 1, Sample 2, Tie, or Both Bad
4. **Tournament** — round-robin (≤6 candidates) or Swiss-system (>6 candidates) determines final rankings
5. **Promote Winner** — the champion checkpoint is applied to the voice, becoming the base for future training

### Tournament Algorithms

| Algorithm | When | Rounds | Matches |
|-----------|------|--------|---------|
| Round Robin | ≤6 candidates | 1 | All pairs |
| Swiss | >6 candidates | ceil(log2(n))+1 | Paired by standing each round; ranked by wins + Buchholz tie-break |

### Candidate Assembly

Candidates are drawn from three sources:

| Source | Behavior |
|--------|----------|
| Champion carry | Always included (defending champion) |
| 2nd/3rd carry | Included unless the checkpoint has appeared as `second_carry` or `third_carry` in ≥2 consecutive sessions |
| New checkpoints | From recent completed training jobs; must pass hard safety gate |

New checkpoints that fail the full validation gate but pass hard safety can be rescued via **gray-zone rescue** when calibration is in canary or active state (see Calibration below).

### Audio Generation Pipeline

Audio is generated asynchronously via RunPod serverless inference:

1. POST `/v1/arena/sessions/:id/generate` fires all RunPod async requests (batched at concurrency 5)
2. Job IDs are tracked in the session's `notes` field as JSON
3. GET `/v1/arena/sessions/:id` lazy-polls up to 5 pending jobs per request
4. Completed audio is decoded from base64, stored in R2, and match audio keys are assigned
5. When all jobs finish, the session transitions from `generating` to `active`
6. GET `/v1/arena/audio/:r2Key` serves WAV files directly from R2

### Calibration Feedback Loop

Arena results feed back into the training pipeline through an automatic calibration system. This reduces training waste by learning which automated scores actually predict human preference for each voice.

**What it learns:** logistic regression over score deltas (ASR, speaker, style, tone, speed, overall, duration) weighted by vote confidence. Adjacent-rank pseudo-pairs from final standings contribute weak signal (weight 0.25).

**Alpha blending** — learned weights are blended with defaults, not replaced:

| Effective Matches | Alpha | Effect |
|-------------------|-------|--------|
| <10 | 0 | Defaults only |
| 10-19 | 0.10 | Barely perceptible |
| 20-39 | 0.25 | Moderate influence |
| 40-79 | 0.50 | Equal blend |
| 80+ | 0.75 | Learned weights dominate |

Per-weight shift is capped at ±0.10 per calibration cycle to prevent wild swings.

**State machine:**

```
shadow → canary → active
                → rolled_back
```

| State | Matches | Effect |
|-------|---------|--------|
| shadow | <20 | Compute calibration but don't use it |
| canary | 20-39 | Only affects gray-zone rescue (lets near-miss checkpoints into arena) |
| active | 40+ (accuracy ≥0.60) | Affects both ranking weights and gray-zone rescue |
| rolled_back | Any (accuracy <0.50 or both_bad >0.35) | Auto-disabled, reverts to defaults. Sticky — stays rolled_back until manually re-applied |

**Auto-calibration trigger:** runs automatically via `waitUntil` whenever a session is finalized (through the vote endpoint's round advance or the manual complete endpoint).

**Gate split:**

| Gate Type | Metrics | Behavior |
|-----------|---------|----------|
| Hard safety | ASR ≥0.75, Speaker ≥0.70, Health ≥0.65 | Never bypassed, even by calibration |
| Soft preference | tone, style, speed, duration, overall | Still causes rejection by default, but can be relaxed via gray-zone rescue when calibration is canary/active |

Note: `tone`, `speed`, and `style` gate checks all apply independently. `computeRankingScore` uses `style_score` directly when available (falling back to `tone*0.6 + speed*0.4` when absent for the style weight), plus tone and speed always contribute via their own weights.

**Global fallback chain** for calibrated weights: voice-specific → `__global__` → hardcoded defaults.

### How Arena Results Affect Training

The calibration feedback loop connects arena outcomes to the training pipeline through these mechanisms:

1. **Ranking weights** — when calibration is `active`, `computeRankingScore()` uses blended weights instead of hardcoded defaults. This changes which checkpoints are ranked higher during candidate assembly and checkpoint selection.

2. **Gray-zone rescue** — when calibration is `canary` or `active`, checkpoints that barely miss one soft threshold (within 0.03) can enter the arena if their blended ranking score is high enough. Without calibration, these checkpoints would be silently discarded.

3. **Champion promotion** — the arena winner's checkpoint becomes the voice's `checkpoint_r2_prefix`. The next training run uses this as the base, so the training pipeline inherits the human-validated quality signal.

4. **Carry limits** — non-champion candidates that place 2nd/3rd in two consecutive arenas without improving are retired. This prevents stale checkpoints from occupying arena slots indefinitely, giving new training results more opportunities.

---

### TypeScript errors in Worker

```bash
cd cloud/cloudflare/worker-api
npx tsc --noEmit  # Check for type errors
npx wrangler dev   # Local development server
```
