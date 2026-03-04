#!/usr/bin/env python3
"""
Local dataset preparation pipeline using Cloudflare Workers AI Whisper.

Steps:
  1. Extract audio from MP4 → 24kHz mono WAV (ffmpeg)
  2. Segment audio via silence-based VAD (numpy/soundfile)
  3. Transcribe each segment via Cloudflare Workers AI (@cf/openai/whisper-large-v3-turbo)
  3.5. LLM-based transcription review (quality scoring + correction)
  4. Build train_raw.jsonl + select ref_audio
  5. Upload everything to R2

Usage:
  python scripts/prepare_dataset_cf_whisper.py \
    --input ~/Desktop/서재형\ 대표님\ 음성_0821ver.mp4 \
    --speaker seo_jaehyung \
    --r2-prefix datasets/seo_jaehyung
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

# ── Config ────────────────────────────────────────────────────────────
SAMPLE_RATE = 24000
MIN_DUR = 3.0
MAX_DUR = 15.0
SILENCE_THRESH_DB = -35
MIN_SILENCE_DUR = 0.4
MAX_SEGMENTS = 400

CF_ACCOUNT_ID = "3672c083409da5ca133c50e0d06fd0e0"
CF_WHISPER_MODEL = "@cf/openai/whisper-large-v3-turbo"
R2_BUCKET = "qwen-tts-studio"


def get_cf_token() -> str:
    """Read OAuth token from wrangler config."""
    config_path = Path.home() / "Library/Preferences/.wrangler/config/default.toml"
    if not config_path.exists():
        # Try Linux/other paths
        for p in [
            Path.home() / ".config/.wrangler/config/default.toml",
            Path.home() / ".wrangler/config/default.toml",
        ]:
            if p.exists():
                config_path = p
                break
    if not config_path.exists():
        print("ERROR: wrangler config not found. Run 'npx wrangler login' first.")
        sys.exit(1)

    text = config_path.read_text()
    for line in text.splitlines():
        if line.startswith("oauth_token"):
            return line.split("=", 1)[1].strip().strip('"')
    print("ERROR: oauth_token not found in wrangler config.")
    sys.exit(1)


def get_r2_credentials() -> dict:
    """Read R2 credentials from environment or wrangler secrets."""
    endpoint = os.environ.get(
        "R2_ENDPOINT_URL",
        f"https://{CF_ACCOUNT_ID}.r2.cloudflarestorage.com",
    )
    access_key = os.environ.get("R2_ACCESS_KEY_ID", "")
    secret_key = os.environ.get("R2_SECRET_ACCESS_KEY", "")

    if not access_key or not secret_key:
        print("WARNING: R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY not set.")
        print("  Set them as env vars, or upload will use wrangler CLI fallback.")
        return {}

    return {
        "endpoint_url": endpoint,
        "access_key_id": access_key,
        "secret_access_key": secret_key,
    }


# ── Step 1: Extract Audio ────────────────────────────────────────────
def extract_audio(mp4: Path, work_dir: Path) -> Path:
    wav = work_dir / "full_audio.wav"
    if wav.exists():
        dur = sf.info(str(wav)).duration
        print(f"  [skip] WAV already exists: {dur:.0f}s ({dur / 60:.1f}min)")
        return wav

    print(f"Extracting audio from {mp4.name} → 24kHz mono WAV ...")
    subprocess.run(
        [
            "ffmpeg",
            "-i",
            str(mp4),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(SAMPLE_RATE),
            "-ac",
            "1",
            str(wav),
            "-y",
        ],
        check=True,
        capture_output=True,
    )
    dur = sf.info(str(wav)).duration
    print(f"  Duration: {dur:.0f}s ({dur / 60:.1f}min)")
    return wav


# ── Step 2: Segment Audio ────────────────────────────────────────────
def find_silence_boundaries(audio: np.ndarray, sr: int) -> list[int]:
    from numpy.lib.stride_tricks import sliding_window_view

    frame_len = int(sr * 0.025)
    hop = int(sr * 0.010)
    n_frames = (len(audio) - frame_len) // hop
    frames = sliding_window_view(audio.astype(np.float64), frame_len)[::hop][:n_frames]
    rms = np.sqrt(np.mean(frames**2, axis=1))
    rms_db = np.where(rms < 1e-10, -100.0, 20.0 * np.log10(rms / 32768.0))

    is_silent = rms_db < SILENCE_THRESH_DB
    min_silent_frames = int(MIN_SILENCE_DUR / 0.010)
    boundaries = []
    silent_start = None
    for i in range(len(is_silent)):
        if is_silent[i]:
            if silent_start is None:
                silent_start = i
        else:
            if silent_start is not None and (i - silent_start) >= min_silent_frames:
                boundaries.append(((silent_start + i) // 2) * hop)
            silent_start = None
    return boundaries


def segment_audio(wav: Path, segments_dir: Path) -> list[tuple[Path, float]]:
    # Check if segments already exist
    existing = sorted(segments_dir.glob("seg_*.wav"))
    if existing:
        results = []
        for p in existing:
            dur = sf.info(str(p)).duration
            results.append((p, dur))
        total = sum(d for _, d in results)
        print(f"  [skip] {len(results)} segments already exist, total {total:.0f}s")
        return results

    print("Loading WAV into memory ...")
    audio, sr = sf.read(str(wav), dtype="int16")
    assert sr == SAMPLE_RATE

    print("Finding silence boundaries ...")
    boundaries = find_silence_boundaries(audio, sr)
    print(f"  {len(boundaries)} silence points")

    positions = [0] + boundaries + [len(audio)]
    segments = []
    for i in range(len(positions) - 1):
        start, end = positions[i], positions[i + 1]
        dur = (end - start) / sr
        if MIN_DUR <= dur <= MAX_DUR:
            segments.append((start, end))
        elif dur > MAX_DUR:
            chunk = int(MAX_DUR * sr)
            pos = start
            while pos + int(MIN_DUR * sr) < end:
                ce = min(pos + chunk, end)
                if (ce - pos) / sr >= MIN_DUR:
                    segments.append((pos, ce))
                pos = ce

    if len(segments) > MAX_SEGMENTS:
        segments = segments[:MAX_SEGMENTS]

    segments_dir.mkdir(parents=True, exist_ok=True)
    results = []
    total = 0.0
    for i, (start, end) in enumerate(segments):
        seg = audio[start:end]
        dur = len(seg) / sr
        total += dur
        p = segments_dir / f"seg_{i:04d}.wav"
        sf.write(str(p), seg, sr, subtype="PCM_16")
        results.append((p, dur))

    print(f"  {len(results)} segments, total {total:.0f}s ({total / 60:.1f}min)")
    return results


# ── Step 3: Transcribe via Cloudflare Workers AI ─────────────────────
def transcribe_segment(seg_path: Path, token: str, retries: int = 3) -> dict | None:
    """Transcribe a single audio segment using CF Workers AI Whisper (base64 JSON)."""
    import base64
    import requests

    url = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/run/{CF_WHISPER_MODEL}"
    audio_b64 = base64.b64encode(seg_path.read_bytes()).decode("utf-8")

    payload = {
        "audio": audio_b64,
        "language": "ko",
        "task": "transcribe",
        "vad_filter": False,
        "condition_on_previous_text": False,
        "beam_size": 5,
    }

    for attempt in range(retries):
        try:
            resp = requests.post(
                url,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=60,
            )

            if resp.status_code == 429:
                wait = 2 ** attempt + 1
                print(f"    Rate limited on {seg_path.name}, retrying in {wait}s...")
                time.sleep(wait)
                continue

            if resp.status_code != 200:
                print(f"    ERROR {resp.status_code} on {seg_path.name}: {resp.text[:200]}")
                return None

            result = resp.json()
            if not result.get("success"):
                print(f"    API error on {seg_path.name}: {result.get('errors')}")
                return None

            text = result.get("result", {}).get("text", "").strip()
            if text and len(text) >= 2:
                return {"text": text, "segment": seg_path.name}
            return None

        except Exception as e:
            if attempt == retries - 1:
                print(f"    Exception on {seg_path.name}: {e}")
                return None
            time.sleep(1)

    return None


def transcribe_all(
    seg_paths: list[tuple[Path, float]],
    work_dir: Path,
    token: str,
    max_workers: int = 8,
) -> list[dict]:
    """Transcribe all segments in parallel with resume support."""
    import concurrent.futures

    progress_file = work_dir / "transcription_progress.json"
    results = []

    # Resume from previous run
    if progress_file.exists():
        with open(progress_file) as f:
            results = json.load(f)
        done_names = {r["segment"] for r in results}
        print(f"  Resuming: {len(results)} already transcribed")
    else:
        done_names = set()

    remaining = [(p, d) for p, d in seg_paths if p.name not in done_names]
    total = len(seg_paths)
    print(f"  {len(remaining)} segments to transcribe ({len(done_names)} done)")

    if not remaining:
        print(f"  All {total} segments already transcribed.")
        return results

    completed = len(done_names)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_info = {
            executor.submit(transcribe_segment, seg_path, token): (seg_path, dur)
            for seg_path, dur in remaining
        }
        for future in concurrent.futures.as_completed(future_to_info):
            seg_path, dur = future_to_info[future]
            result = future.result()
            completed += 1

            if result:
                result["duration"] = round(dur, 2)
                results.append(result)

            if completed % 20 == 0 or completed == total:
                valid = len(results)
                print(f"  [{completed}/{total}] {valid} valid transcriptions")
                with open(progress_file, "w") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)

    # Final save + sort by segment name for deterministic order
    results.sort(key=lambda r: r["segment"])
    with open(progress_file, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"  Total: {len(results)} valid transcriptions out of {total} segments")
    return results


# ── Step 3.5: LLM Transcription Review (OpenAI GPT-5.2) ──────────────
OPENAI_MODEL = "gpt-5.2"

REVIEW_SYSTEM_PROMPT = """당신은 한국어 음성 전사(STT) 품질 검수 전문가입니다.
음성 인식 결과를 검토하여 오류를 찾고 수정합니다.

각 전사 텍스트에 대해 JSON으로 응답하세요:
{
  "score": 1-5 (5=완벽, 4=사소한오류, 3=수정필요, 2=심각한오류, 1=사용불가),
  "corrected": "수정된 텍스트 (수정 없으면 원본 그대로)",
  "issues": ["발견된 문제들"]
}

검수 기준:
- 반복/할루시네이션: 같은 구절이 의미없이 반복되면 감점
- 잘린 문장: 문장이 중간에 끊겼으면 표시
- 오탈자/동음이의어: 문맥상 잘못된 단어 수정 (예: '주식을 팔다' vs '주식을 펄다')
- 의미불명: 문맥상 이해 불가능한 부분 표시
- 숫자/고유명사: 명백한 오류만 수정 (추측하지 말 것)

TTS 훈련용이므로 발화 그대로의 구어체를 유지하세요. 문어체로 바꾸지 마세요.
반드시 JSON만 출력하세요. 다른 텍스트 없이."""


def get_openai_key() -> str:
    """Find OpenAI API key from env or known .env files."""
    key = os.environ.get("OPENAI_API_KEY", "")
    if key:
        return key
    # Search known locations
    for env_path in [
        Path.home() / "Desktop/projects/kompas/kompas/backend/.env",
        Path.home() / ".env",
        Path.cwd() / ".env",
    ]:
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("OPENAI_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"')
    print("ERROR: OPENAI_API_KEY not found. Set it as env var or in .env")
    sys.exit(1)


def review_batch_with_llm(
    entries: list[dict],
    api_key: str,
    retries: int = 2,
) -> list[dict]:
    """Review a batch of transcriptions using OpenAI GPT-5.2."""
    import requests

    url = "https://api.openai.com/v1/chat/completions"

    batch_text = "\n".join(
        f'[{i+1}] (seg={e["segment"]}, {e["duration"]}s): "{e["text"]}"'
        for i, e in enumerate(entries)
    )
    user_prompt = (
        f"{len(entries)}개의 전사 텍스트를 검수하세요. "
        f"JSON 배열로 응답하세요. [{len(entries)}개 객체]\n\n{batch_text}"
    )

    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": REVIEW_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "max_completion_tokens": 4096,
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
    }

    for attempt in range(retries):
        try:
            resp = requests.post(
                url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=120,
            )

            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", str(2 ** attempt + 1)))
                print(f"    Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue

            if resp.status_code != 200:
                print(f"    OpenAI API error {resp.status_code}: {resp.text[:200]}")
                return [{"score": 3, "corrected": e["text"], "issues": []} for e in entries]

            result = resp.json()
            response_text = result["choices"][0]["message"]["content"]

            # Parse JSON from response
            try:
                cleaned = response_text.strip()
                if cleaned.startswith("```"):
                    cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0].strip()
                parsed = json.loads(cleaned)
                # Handle {"reviews": [...]} wrapper or direct array
                if isinstance(parsed, dict):
                    reviews = parsed.get("reviews", parsed.get("results", [parsed]))
                elif isinstance(parsed, list):
                    reviews = parsed
                else:
                    reviews = [parsed]
                return reviews
            except json.JSONDecodeError:
                import re
                json_objects = re.findall(r'\{[^{}]+\}', response_text)
                reviews = []
                for obj_str in json_objects:
                    try:
                        reviews.append(json.loads(obj_str))
                    except json.JSONDecodeError:
                        continue
                if reviews:
                    return reviews
                print(f"    Failed to parse LLM response, using originals")
                return [{"score": 3, "corrected": e["text"], "issues": []} for e in entries]

        except Exception as e:
            if attempt == retries - 1:
                print(f"    LLM exception: {e}")
                return [{"score": 3, "corrected": e_["text"], "issues": []} for e_ in entries]
            time.sleep(1)

    return [{"score": 3, "corrected": e["text"], "issues": []} for e in entries]


def review_transcriptions(
    transcriptions: list[dict],
    work_dir: Path,
    api_key: str,
    batch_size: int = 10,
) -> list[dict]:
    """Review all transcriptions with LLM, filter low quality, apply corrections."""
    review_file = work_dir / "review_results.json"

    # Resume support
    if review_file.exists():
        with open(review_file) as f:
            cached = json.load(f)
        if len(cached) == len(transcriptions):
            print(f"  [skip] Review already done ({len(cached)} entries)")
            # Apply filter
            accepted = [c for c in cached if c.get("score", 3) >= 3]
            print(f"  Accepted: {len(accepted)}/{len(cached)} (score >= 3)")
            return accepted

    all_reviews = []
    total_batches = (len(transcriptions) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(transcriptions))
        batch = transcriptions[start:end]

        reviews = review_batch_with_llm(batch, api_key)

        # Merge reviews with original data
        for i, entry in enumerate(batch):
            review = reviews[i] if i < len(reviews) else {"score": 3, "corrected": entry["text"], "issues": []}
            merged = {
                "segment": entry["segment"],
                "original_text": entry["text"],
                "text": review.get("corrected", entry["text"]),
                "score": review.get("score", 3),
                "issues": review.get("issues", []),
                "duration": entry.get("duration", 0),
            }
            all_reviews.append(merged)

        if (batch_idx + 1) % 10 == 0 or batch_idx == total_batches - 1:
            reviewed = len(all_reviews)
            print(f"  [{reviewed}/{len(transcriptions)}] reviewed")
            with open(review_file, "w") as f:
                json.dump(all_reviews, f, ensure_ascii=False, indent=2)

        time.sleep(0.3)  # Rate limit courtesy

    # Final save
    with open(review_file, "w") as f:
        json.dump(all_reviews, f, ensure_ascii=False, indent=2)

    # Stats
    scores = [r["score"] for r in all_reviews]
    from collections import Counter
    dist = Counter(scores)
    print(f"  Score distribution: {dict(sorted(dist.items()))}")

    accepted = [r for r in all_reviews if r["score"] >= 3]
    corrected = sum(1 for r in accepted if r["text"] != r["original_text"])
    rejected = len(all_reviews) - len(accepted)
    print(f"  Accepted: {len(accepted)}, Corrected: {corrected}, Rejected: {rejected}")

    return accepted


# ── Step 4: Build train_raw.jsonl ─────────────────────────────────────
def pick_ref_audio(seg_paths: list[tuple[Path, float]], work_dir: Path) -> Path:
    best = min(seg_paths, key=lambda x: abs(x[1] - 6.0))[0]
    ref = work_dir / "ref_audio.wav"
    shutil.copy2(str(best), str(ref))
    print(f"  ref_audio: {best.name} (copied to ref_audio.wav)")
    return ref


def build_jsonl(
    transcriptions: list[dict],
    work_dir: Path,
) -> Path:
    jsonl = work_dir / "train_raw.jsonl"
    with open(jsonl, "w", encoding="utf-8") as f:
        for entry in transcriptions:
            record = {
                "audio": f"segments/{entry['segment']}",
                "text": entry["text"],
                "ref_audio": "ref_audio.wav",
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"  train_raw.jsonl: {len(transcriptions)} utterances")
    return jsonl


# ── Step 5: Upload to R2 ─────────────────────────────────────────────
def upload_to_r2_wrangler(work_dir: Path, r2_prefix: str, speaker_name: str):
    """Upload using wrangler CLI (auth handled automatically)."""
    worker_dir = Path(__file__).resolve().parent.parent / "cloud/cloudflare/worker-api"

    def put(local: Path, key: str):
        subprocess.run(
            [
                "npx",
                "wrangler",
                "r2",
                "object",
                "put",
                f"{R2_BUCKET}/{key}",
                "--file",
                str(local),
            ],
            check=True,
            capture_output=True,
            cwd=str(worker_dir),
        )

    # Upload train_raw.jsonl
    jsonl = work_dir / "train_raw.jsonl"
    print(f"  Uploading {jsonl.name} ...")
    put(jsonl, f"{r2_prefix}/train_raw.jsonl")

    # Upload ref_audio
    ref = work_dir / "ref_audio.wav"
    print(f"  Uploading ref_audio.wav ...")
    put(ref, f"{r2_prefix}/ref_audio.wav")

    # Upload segments
    segments_dir = work_dir / "segments"
    seg_files = sorted(segments_dir.glob("seg_*.wav"))
    print(f"  Uploading {len(seg_files)} segments ...")
    for i, seg in enumerate(seg_files):
        put(seg, f"{r2_prefix}/segments/{seg.name}")
        if (i + 1) % 50 == 0:
            print(f"    {i + 1}/{len(seg_files)}")

    # Upload manifest
    manifest = {
        "speaker_name": speaker_name,
        "total_segments": len(seg_files),
        "jsonl_key": f"{r2_prefix}/train_raw.jsonl",
        "ref_audio_key": f"{r2_prefix}/ref_audio.wav",
    }
    manifest_path = work_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    put(manifest_path, f"{r2_prefix}/manifest.json")

    print(f"  Upload complete! R2 prefix: {r2_prefix}/")


def main():
    parser = argparse.ArgumentParser(description="Prepare TTS dataset with CF Whisper")
    parser.add_argument("--input", required=True, help="Path to MP4/audio file")
    parser.add_argument("--speaker", default="seo_jaehyung", help="Speaker name")
    parser.add_argument(
        "--r2-prefix", default="datasets/seo_jaehyung", help="R2 key prefix"
    )
    parser.add_argument(
        "--work-dir", default="/tmp/audio_prep", help="Working directory"
    )
    parser.add_argument("--skip-upload", action="store_true", help="Skip R2 upload")
    parser.add_argument("--skip-review", action="store_true", help="Skip LLM review step")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    work_dir = Path(args.work_dir)
    segments_dir = work_dir / "segments"
    work_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)

    print(f"=== Qwen3-TTS Dataset Prep (CF Whisper) ===")
    print(f"  Input: {input_path}")
    print(f"  Speaker: {args.speaker}")
    print(f"  Work dir: {work_dir}")
    print()

    # Step 1: Extract audio
    print("[1/6] Extract audio")
    wav = extract_audio(input_path, work_dir)
    print()

    # Step 2: Segment
    print("[2/6] Segment audio")
    seg_paths = segment_audio(wav, segments_dir)
    print()

    # Step 3: Transcribe
    print("[3/6] Transcribe via Cloudflare Workers AI Whisper")
    token = get_cf_token()
    transcriptions = transcribe_all(seg_paths, work_dir, token)
    print()

    # Step 4: LLM Review (OpenAI GPT-5.2)
    if not args.skip_review:
        print("[4/6] LLM transcription review (OpenAI GPT-5.2)")
        openai_key = get_openai_key()
        reviewed = review_transcriptions(transcriptions, work_dir, openai_key, batch_size=10)
        print()
    else:
        print("[4/6] LLM review skipped (--skip-review)")
        reviewed = transcriptions
        print()

    # Step 5: Build JSONL + ref audio
    print("[5/6] Build train_raw.jsonl")
    ref = pick_ref_audio(seg_paths, work_dir)
    jsonl = build_jsonl(reviewed, work_dir)

    # Step 6: Upload to R2
    if not args.skip_upload:
        print("[6/6] Upload to R2")
        upload_to_r2_wrangler(work_dir, args.r2_prefix, args.speaker)
    else:
        print("[6/6] Upload skipped (--skip-upload)")
    print()

    print("=== DONE ===")
    print(f"  Segments: {len(seg_paths)}")
    print(f"  Transcribed: {len(transcriptions)}")
    if not args.skip_review:
        print(f"  After review: {len(reviewed)} accepted")
    print(f"  JSONL: {jsonl}")
    if not args.skip_upload:
        print(f"  R2: {args.r2_prefix}/")


if __name__ == "__main__":
    main()
