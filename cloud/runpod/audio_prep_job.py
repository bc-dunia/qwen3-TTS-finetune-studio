#!/usr/bin/env python3
"""
RunPod one-shot job: download MP4 from R2, segment, transcribe, upload dataset back.
Runs on GPU for fast Whisper transcription.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

# ── Config ────────────────────────────────────────────────────────────
SAMPLE_RATE = 24000
MIN_DUR = 3.0
MAX_DUR = 15.0
SILENCE_THRESH_DB = -35
MIN_SILENCE_DUR = 0.4
MAX_SEGMENTS = 400  # ~40-60 min of training audio

WORK = Path("/tmp/audio_prep")
SEGMENTS_DIR = WORK / "segments"

R2_BUCKET = os.environ.get("R2_BUCKET", "qwen-tts-studio")
R2_INPUT_KEY = os.environ.get("R2_INPUT_KEY", "raw/voice_recording.mp4")
R2_OUTPUT_PREFIX = os.environ.get("R2_OUTPUT_PREFIX", "datasets/seo_jaehyung")
SPEAKER_NAME = os.environ.get("SPEAKER_NAME", "seo_jaehyung")


def get_s3():
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT_URL"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        config=Config(s3={"addressing_style": "path"}, signature_version="s3v4"),
        region_name="auto",
    )


def download_mp4(s3):
    mp4 = WORK / "input.mp4"
    print(f"Downloading s3://{R2_BUCKET}/{R2_INPUT_KEY} ...")
    s3.download_file(R2_BUCKET, R2_INPUT_KEY, str(mp4))
    print(f"  Downloaded {mp4.stat().st_size / 1024 / 1024:.0f}MB")
    return mp4


def extract_audio(mp4: Path) -> Path:
    wav = WORK / "full_audio.wav"
    print("Extracting audio → 24kHz mono WAV ...")
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
    mp4.unlink()  # free space
    return wav


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


def segment_audio(wav: Path) -> list[tuple[Path, float]]:
    print("Loading WAV into memory ...")
    audio, sr = sf.read(str(wav), dtype="int16")
    assert sr == SAMPLE_RATE

    print("Finding silence boundaries (vectorized) ...")
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

    SEGMENTS_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    total = 0.0
    for i, (start, end) in enumerate(segments):
        seg = audio[start:end]
        dur = len(seg) / sr
        total += dur
        p = SEGMENTS_DIR / f"seg_{i:04d}.wav"
        sf.write(str(p), seg, sr, subtype="PCM_16")
        results.append((p, dur))

    wav.unlink()  # free space
    print(f"  {len(results)} segments, total {total:.0f}s ({total / 60:.1f}min)")
    return results


def pick_ref_audio(seg_paths: list[tuple[Path, float]]) -> Path:
    best = min(seg_paths, key=lambda x: abs(x[1] - 6.0))[0]
    ref = WORK / "ref_audio.wav"
    shutil.copy2(str(best), str(ref))
    return ref


def transcribe(seg_paths: list[tuple[Path, float]], ref: Path) -> Path:
    print("Loading faster-whisper (large-v3, GPU) ...")
    from faster_whisper import WhisperModel

    model = WhisperModel("large-v3", device="cuda", compute_type="float16")

    jsonl = WORK / "train_raw.jsonl"
    count = 0
    with open(jsonl, "w", encoding="utf-8") as f:
        for i, (seg_path, dur) in enumerate(seg_paths):
            try:
                segs, info = model.transcribe(
                    str(seg_path), language=None, vad_filter=True, beam_size=5
                )
                text = " ".join(s.text.strip() for s in segs).strip()
            except Exception as e:
                print(f"  SKIP {seg_path.name}: {e}")
                continue
            if not text or len(text) < 2:
                continue
            entry = {
                "audio": f"segments/{seg_path.name}",
                "text": text,
                "ref_audio": "ref_audio.wav",
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            count += 1
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(seg_paths)} transcribed ({count} valid)")

    print(f"  {count} utterances in {jsonl.name}")
    return jsonl


def upload_dataset(s3, ref: Path, jsonl: Path, seg_paths: list[tuple[Path, float]]):
    print(f"Uploading dataset to R2 ({R2_OUTPUT_PREFIX}/) ...")
    prefix = R2_OUTPUT_PREFIX

    s3.upload_file(str(jsonl), R2_BUCKET, f"{prefix}/train_raw.jsonl")
    s3.upload_file(str(ref), R2_BUCKET, f"{prefix}/ref_audio.wav")

    for seg_path, _ in seg_paths:
        key = f"{prefix}/segments/{seg_path.name}"
        s3.upload_file(str(seg_path), R2_BUCKET, key)

    # Upload a manifest
    manifest = {
        "speaker_name": SPEAKER_NAME,
        "total_segments": len(seg_paths),
        "jsonl_key": f"{prefix}/train_raw.jsonl",
        "ref_audio_key": f"{prefix}/ref_audio.wav",
    }
    manifest_path = WORK / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    s3.upload_file(str(manifest_path), R2_BUCKET, f"{prefix}/manifest.json")
    print("  Upload complete!")


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    s3 = get_s3()

    mp4 = download_mp4(s3)
    wav = extract_audio(mp4)
    seg_paths = segment_audio(wav)
    ref = pick_ref_audio(seg_paths)
    jsonl = transcribe(seg_paths, ref)
    upload_dataset(s3, ref, jsonl, seg_paths)

    print("\n=== DONE ===")
    with open(str(jsonl)) as f:
        lines = f.readlines()
    print(f"Dataset: {len(lines)} utterances")
    print(f"R2 prefix: {R2_OUTPUT_PREFIX}/")


if __name__ == "__main__":
    main()
