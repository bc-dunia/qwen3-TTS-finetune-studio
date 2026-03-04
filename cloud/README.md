# Qwen3-TTS Studio — Cloud Deployment

Cloud deployment architecture for Qwen3-TTS Studio. Splits the application across:
- **RunPod** — GPU compute for training and inference
- **Cloudflare** — API proxy, storage, database, and frontend

## Architecture

```
                         Cloudflare Pages
                       (React Frontend)
                             |
                             v
                     Cloudflare Workers
                    (ElevenLabs-compatible API)
                      /        |        \
                     v         v         v
              Cloudflare R2  Cloudflare D1  RunPod
              (Checkpoints,  (Metadata,     (GPU Training Pods
               Audio,        Voices,         + Serverless
               Datasets)     Jobs)           Inference)
```

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
npx wrangler d1 execute qwen-tts-db --file=schema.sql

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

# Deploy
npx wrangler deploy
```

### 2. RunPod Setup

#### Build Docker Images

```bash
# From project root
cd /path/to/qwen3-tts-finetune-studio

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
npx wrangler pages deploy dist --project-name qwen-tts-studio
```

## API Usage

The API is ElevenLabs-compatible. You can use any ElevenLabs client SDK by changing the base URL.

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

## Cost Estimates

### Cloudflare (monthly)

| Service | Usage | Cost |
|---------|-------|------|
| R2 Storage | 50GB checkpoints + 100GB audio | ~$2.25 |
| R2 Egress | Unlimited | $0 (free!) |
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

## File Structure

```
cloud/
├── README.md                          # This file
├── runpod/
│   ├── Dockerfile.training            # GPU training pod
│   ├── Dockerfile.inference           # Serverless inference
│   ├── build.sh                       # Docker build helper
│   ├── requirements.txt               # Python dependencies
│   ├── r2_storage.py                  # R2 client (shared)
│   ├── training_handler.py            # Training orchestrator
│   └── inference_handler.py           # TTS generation handler
└── cloudflare/
    ├── worker-api/
    │   ├── wrangler.toml              # Worker configuration
    │   ├── package.json
    │   ├── tsconfig.json
    │   ├── schema.sql                 # D1 database schema
    │   └── src/
    │       ├── index.ts               # Main Hono router
    │       ├── types.ts               # TypeScript types
    │       ├── middleware/auth.ts      # API key auth
    │       ├── routes/
    │       │   ├── tts.ts             # TTS generation (ElevenLabs-compatible)
    │       │   ├── voices.ts          # Voice CRUD
    │       │   ├── models.ts          # Model listing
    │       │   ├── training.ts        # Training management
    │       │   └── upload.ts          # Presigned URL generation
    │       └── lib/
    │           ├── runpod.ts          # RunPod API client
    │           ├── r2.ts              # R2 presigned URLs
    │           └── d1.ts              # D1 query helpers
    └── frontend/                      # React SPA (Cloudflare Pages)
        ├── package.json
        ├── vite.config.ts
        ├── index.html
        └── src/
            ├── main.tsx
            ├── App.tsx
            ├── lib/api.ts
            ├── pages/
            │   ├── Dashboard.tsx
            │   ├── Voices.tsx
            │   ├── VoiceDetail.tsx
            │   ├── Playground.tsx
            │   └── Training.tsx
            └── components/
                ├── Layout.tsx
                ├── AudioPlayer.tsx
                └── VoiceCard.tsx
```

## GPU Recommendations

| Use Case | GPU | VRAM | Reason |
|----------|-----|------|--------|
| Training 1.7B | A100-40GB | 40GB | Full SFT + AdamW states ≈ 20-24GB |
| Training 0.6B | RTX 4090 | 24GB | Full SFT fits in 24GB |
| Inference 1.7B | RTX 4090 | 24GB | Model uses ~4GB VRAM |
| Inference 0.6B | RTX 4090 | 24GB | Model uses ~2.5GB VRAM |

## Troubleshooting

### Training pod doesn't start
- Check RunPod dashboard for pod status
- Verify R2 credentials are set correctly in the template
- Check `jobs/{job_id}/status.json` in R2 for error messages

### Inference cold start is slow
- First request per voice downloads checkpoint from R2 (~3-5GB)
- Increase serverless idle timeout to keep warm instances longer
- Consider using RunPod Network Volume for checkpoint caching

### Audio quality issues
- Ensure training data is 24kHz mono WAV
- Try different voice_settings (lower stability = more expressive)
- Use the Quality tab in the local Gradio UI for pre-training checks

### TypeScript errors in Worker
```bash
cd cloud/cloudflare/worker-api
npx tsc --noEmit  # Check for type errors
npx wrangler dev   # Local development server
```
