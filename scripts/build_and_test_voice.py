#!/usr/bin/env python3
from __future__ import annotations

import sys
import argparse
import gc
import json
import math
import random
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf
import torch
from faster_whisper import WhisperModel
from qwen_tts import Qwen3TTSModel


TEST_TEXTS = [
    "안녕하세요. 오늘 시장 이야기를 간단하게 말씀드리겠습니다.",
    "최근 반도체 섹터가 강세를 보이고 있는데, 그 이유를 분석해보겠습니다.",
    "투자에서 가장 중요한 것은 리스크 관리입니다. 절대 잊지 마세요.",
    "자, 그러면 오늘의 주요 종목들을 하나씩 살펴볼까요?",
    "이 기업의 실적이 예상보다 좋았고, 앞으로의 전망도 밝습니다.",
    "여러분, 감사합니다. 다음 시간에 또 뵙겠습니다.",
]


@dataclass
class SegmentRow:
    audio_path: str
    audio_name: str
    text: str
    duration_sec: float
    metrics: dict[str, float]
    speaker_cos: float
    audio_quality_score: float
    duration_score: float
    ref_score: float


@dataclass
class ReviewRow:
    text_index: int
    text: str
    seed: int
    wav: str
    speaker_cos: float
    asr_sim: float
    duration_sec: float
    target_duration_sec: float
    speed_score: float
    f0_median: float
    f0_std: float
    f0_consistency: float
    final_score: float
    asr_text: str
    error: str | None = None


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _choose_device() -> tuple[str, torch.dtype]:
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    if torch.cuda.is_available():
        return "cuda:0", torch.bfloat16
    return "cpu", torch.float32


def _find_default_base_model() -> str:
    root = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--Qwen--Qwen3-TTS-12Hz-0.6B-Base"
        / "snapshots"
    )
    if not root.exists():
        return "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
    snaps = sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name)
    if not snaps:
        return "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
    return str(snaps[-1].resolve())


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= 1e-12:
        return v.copy()
    return v / n


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb + 1e-12))


def _extract_emb_from_wav(model: Qwen3TTSModel, wav_24k: np.ndarray) -> np.ndarray:
    emb = model.model.extract_speaker_embedding(wav_24k, 24000)
    if isinstance(emb, torch.Tensor):
        emb = emb.detach().float().cpu().numpy()
    return _unit(np.asarray(emb, dtype=np.float32))


def _normalize_text(s: str) -> str:
    return re.sub(r"[^0-9a-zA-Z가-힣]+", "", (s or "").lower())


def _levenshtein_ratio(a: str, b: str) -> float:
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    n = len(b)
    prev = list(range(n + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i] + [0] * n
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    dist = prev[n]
    return 1.0 - (dist / max(len(a), len(b), 1))


def _asr_similarity(target: str, pred: str) -> float:
    return float(_levenshtein_ratio(_normalize_text(target), _normalize_text(pred)))


def _transcribe_ko(whisper: WhisperModel, wav_path: Path) -> str:
    segments, _ = whisper.transcribe(
        str(wav_path),
        language="ko",
        beam_size=5,
        vad_filter=True,
        condition_on_previous_text=False,
    )
    return "".join(seg.text for seg in segments).strip()


def _signal_metrics(wav: np.ndarray, sr: int = 24000) -> dict[str, float]:
    try:
        if not isinstance(wav, np.ndarray) or wav.size == 0:
            return {}
        if wav.ndim > 1:
            wav = np.mean(wav, axis=1)
        wav = wav.astype(np.float32)

        peak = float(np.max(np.abs(wav)))
        rms = float(np.sqrt(np.mean(wav**2) + 1e-12))
        dc_offset = float(abs(np.mean(wav)))
        clipping_ratio = float(np.mean(np.abs(wav) >= 0.999))
        silent_ratio = float(np.mean(np.abs(wav) < 1e-4))

        frame_size = max(1, int(sr * 0.02))
        n_frames = len(wav) // frame_size
        if n_frames >= 4:
            frame_rms = np.array(
                [
                    float(
                        np.sqrt(
                            np.mean(wav[i * frame_size : (i + 1) * frame_size] ** 2)
                            + 1e-12
                        )
                    )
                    for i in range(n_frames)
                ],
                dtype=np.float32,
            )
            noise_floor = float(np.percentile(frame_rms, 10))
            if noise_floor > 1e-8 and rms > 1e-8:
                snr = float(max(0.0, min(60.0, 20.0 * np.log10(rms / noise_floor))))
            else:
                snr = 40.0
        else:
            snr = 20.0

        return {
            "peak": peak,
            "rms": rms,
            "dc_offset": dc_offset,
            "clipping_ratio": clipping_ratio,
            "silent_ratio": silent_ratio,
            "snr": snr,
        }
    except Exception:
        return {}


def _f0_stats(wav: np.ndarray, sr: int = 24000) -> tuple[float, float]:
    try:
        fmin = float(librosa.note_to_hz("C2"))
        fmax = float(librosa.note_to_hz("C6"))
        f0, _, _ = librosa.pyin(
            wav.astype(np.float32),
            fmin=fmin,
            fmax=fmax,
            sr=sr,
            frame_length=1024,
            hop_length=256,
        )
        v = f0[np.isfinite(f0)]
        if len(v) == 0:
            return 0.0, 0.0
        return float(np.median(v)), float(np.std(v))
    except Exception:
        return 0.0, 0.0


def _run_checked(cmd: list[str], *, desc: str) -> subprocess.CompletedProcess[str]:
    try:
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return proc
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or "").strip()
        stdout = (e.stdout or "").strip()
        detail = stderr if stderr else stdout
        raise RuntimeError(f"{desc} failed: {detail}") from e


def _ffprobe_duration(path: Path) -> float:
    proc = _run_checked(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        desc="ffprobe",
    )
    try:
        return float((proc.stdout or "").strip())
    except Exception as e:
        raise RuntimeError(f"Unable to parse ffprobe duration for {path}") from e


def _extract_audio_if_needed(
    source_mp4: Path, out_wav: Path, skip_extract: bool, max_duration_sec: float = 0
) -> tuple[bool, float]:
    duration = _ffprobe_duration(source_mp4)
    effective_duration = duration
    if max_duration_sec > 0:
        effective_duration = min(duration, max_duration_sec)
    if out_wav.exists() and (
        skip_extract or out_wav.stat().st_mtime >= source_mp4.stat().st_mtime
    ):
        return False, effective_duration
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(source_mp4),
    ]
    if max_duration_sec > 0:
        cmd += ["-t", str(max_duration_sec)]
    cmd += [
        "-vn",
        "-ac",
        "1",
        "-ar",
        "24000",
        "-c:a",
        "pcm_s16le",
        str(out_wav),
    ]
    _run_checked(cmd, desc="ffmpeg extraction")
    return True, effective_duration


def _merge_intervals(
    intervals: list[tuple[int, int]], max_gap_samples: int
) -> list[tuple[int, int]]:
    if not intervals:
        return []
    merged: list[list[int]] = [[int(intervals[0][0]), int(intervals[0][1])]]
    for s, e in intervals[1:]:
        if int(s) - merged[-1][1] < max_gap_samples:
            merged[-1][1] = max(merged[-1][1], int(e))
        else:
            merged.append([int(s), int(e)])
    return [(s, e) for s, e in merged]


def _split_long_interval(
    y: np.ndarray,
    sr: int,
    start: int,
    end: int,
    min_seg_sec: float,
    max_seg_sec: float,
    split_top_db: float,
) -> list[tuple[int, int]]:
    max_len = int(max_seg_sec * sr)
    min_len = int(min_seg_sec * sr)
    if end - start <= max_len:
        return [(start, end)]

    chunk = y[start:end].astype(np.float32)
    frame = 2048
    hop = 512
    rms = librosa.feature.rms(y=chunk, frame_length=frame, hop_length=hop)[0]
    db = librosa.amplitude_to_db(rms + 1e-9, ref=np.max)
    threshold = -float(split_top_db) + 4.0
    silence_frames = np.where(db <= threshold)[0]
    silence_samples = np.asarray(silence_frames * hop, dtype=np.int64)

    target_len = int(((min_seg_sec + max_seg_sec) * 0.5) * sr)
    out: list[tuple[int, int]] = []
    cur = int(start)
    while end - cur > max_len:
        low = cur + min_len
        high = min(cur + max_len, end - min_len)
        candidates = silence_samples[
            (silence_samples >= (low - start)) & (silence_samples <= (high - start))
        ]
        if candidates.size > 0:
            target_abs = cur + target_len
            split_abs = int(
                start + candidates[np.argmin(np.abs((start + candidates) - target_abs))]
            )
        else:
            split_abs = int(cur + target_len)
            split_abs = min(max(split_abs, low), high)
        out.append((cur, split_abs))
        cur = split_abs
    out.append((cur, end))
    return out


def _trim_with_padding(
    y: np.ndarray, top_db: float, pad_sec: float, sr: int = 24000
) -> np.ndarray:
    if y.size == 0:
        return y
    trimmed, idx = librosa.effects.trim(y, top_db=top_db)
    if trimmed.size == 0:
        return y
    pad = int(pad_sec * sr)
    s = max(0, int(idx[0]) - pad)
    e = min(len(y), int(idx[1]) + pad)
    return y[s:e]


def _segment_audio(
    full_wav: Path,
    segments_dir: Path,
    min_seg_sec: float,
    max_seg_sec: float,
    split_top_db: float,
) -> list[Path]:
    wav, sr_loaded = librosa.load(str(full_wav), sr=24000, mono=True)
    sr = int(sr_loaded)
    if sr != 24000:
        raise RuntimeError(f"Unexpected sample rate after extraction: {sr}")
    intervals = librosa.effects.split(wav, top_db=split_top_db)
    merged = _merge_intervals(
        [(int(s), int(e)) for s, e in intervals], max_gap_samples=int(0.3 * sr)
    )

    all_spans: list[tuple[int, int]] = []
    for s, e in merged:
        chunks = _split_long_interval(
            wav,
            sr,
            s,
            e,
            min_seg_sec=min_seg_sec,
            max_seg_sec=max_seg_sec,
            split_top_db=split_top_db,
        )
        all_spans.extend(chunks)

    segments_dir.mkdir(parents=True, exist_ok=True)
    for old in sorted(segments_dir.glob("seg_*.wav")):
        old.unlink()

    saved: list[Path] = []
    idx = 0
    for s, e in all_spans:
        seg = wav[s:e]
        seg = _trim_with_padding(seg, top_db=split_top_db, pad_sec=0.05, sr=sr)
        d = float(len(seg) / sr)
        if d < 0.5:
            continue
        idx += 1
        out = segments_dir / f"seg_{idx:04d}.wav"
        sf.write(str(out), seg.astype(np.float32), sr)
        saved.append(out)
    return saved


def _audio_quality_score(metrics: dict[str, float]) -> float:
    snr = float(metrics.get("snr", 0.0))
    return float(np.clip((snr - 12.0) / (40.0 - 12.0), 0.0, 1.0))


def _duration_score(duration_sec: float) -> float:
    if 5.0 <= duration_sec <= 10.0:
        return 1.0
    if duration_sec < 5.0:
        return float(max(0.0, duration_sec / 5.0))
    return float(max(0.0, 1.0 - (duration_sec - 10.0) / 8.0))


def _is_repetitive_text(text: str) -> bool:
    norm = _normalize_text(text)
    if len(norm) < 10:
        return True
    uniq_ratio = len(set(norm)) / max(len(norm), 1)
    if uniq_ratio < 0.22:
        return True
    if len(norm) >= 8:
        tri = [norm[i : i + 3] for i in range(len(norm) - 2)]
        if tri:
            top = max(tri.count(t) for t in set(tri))
            if top / len(tri) > 0.35:
                return True
    return False


def _load_tts_model(model_path: str, device: str, dtype: torch.dtype) -> Qwen3TTSModel:
    candidates = [
        {
            "device_map": device,
            "dtype": dtype,
            "attn_implementation": "flash_attention_2"
            if device.startswith("cuda")
            else None,
        },
        {
            "device_map": device,
            "torch_dtype": dtype,
            "attn_implementation": "flash_attention_2"
            if device.startswith("cuda")
            else None,
        },
        {"device_map": device, "dtype": dtype},
        {"device_map": device},
    ]
    last_error: Exception | None = None
    for kwargs in candidates:
        kw = {k: v for k, v in kwargs.items() if v is not None}
        try:
            model = Qwen3TTSModel.from_pretrained(model_path, **kw)
            try:
                model.model.eval()
            except Exception:
                pass
            return model
        except Exception as e:
            last_error = e
    if last_error:
        raise RuntimeError(
            f"Failed to load model {model_path}: {last_error}"
        ) from last_error
    raise RuntimeError(f"Failed to load model {model_path}")


# Module-level cache for CPU fallback model (loaded lazily on first MPS failure)
_cpu_fallback_model: Qwen3TTSModel | None = None


def _is_probability_tensor_error(exc: Exception) -> bool:
    """Detect MPS/float16 numerical instability during sampling."""
    msg = str(exc).lower()
    return "probability tensor contains either" in msg and (
        "nan" in msg or "inf" in msg or "element < 0" in msg
    )


def _generate_with_fallback(
    model: Qwen3TTSModel,
    *,
    text: str,
    speaker_name: str,
    language: str,
    params: dict[str, Any],
    checkpoint_path: str = "",
) -> tuple[np.ndarray, int]:
    """Generate speech with multi-level fallback for MPS/float16 instability.

    Retry strategy:
      1. Original params on primary model (MPS/float16)
      2. Lower temperature (0.2) + reduced top_k on primary model
      3. CPU + float32 fallback model (loaded once, cached)
    """
    global _cpu_fallback_model

    kwargs = {k: v for k, v in params.items() if v is not None}

    # --- Attempt 1: original params ---
    try:
        wavs, sr = model.generate_custom_voice(
            text=text, speaker=speaker_name, language=language, **kwargs
        )
        return np.asarray(wavs[0], dtype=np.float32), int(sr)
    except TypeError:
        # Some model versions don't accept subtalker_* params; strip them
        fallback = {
            k: kwargs[k]
            for k in [
                "temperature", "top_k", "top_p",
                "repetition_penalty", "max_new_tokens",
            ]
            if k in kwargs
        }
        try:
            wavs, sr = model.generate_custom_voice(
                text=text, speaker=speaker_name, language=language, **fallback
            )
            return np.asarray(wavs[0], dtype=np.float32), int(sr)
        except RuntimeError as e2:
            if not _is_probability_tensor_error(e2):
                raise
            # Fall through to retry 2
    except RuntimeError as e:
        if not _is_probability_tensor_error(e):
            raise
        # Fall through to retry 2

    # --- Attempt 2: lower temperature on same model ---
    conservative = {**kwargs, "temperature": 0.2, "top_k": 10, "top_p": 0.8}
    # Strip subtalker params for safety
    safe_keys = ["temperature", "top_k", "top_p", "repetition_penalty", "max_new_tokens"]
    conservative = {k: conservative[k] for k in safe_keys if k in conservative}
    try:
        print("    [RETRY] lower temperature (0.2) on primary device...")
        wavs, sr = model.generate_custom_voice(
            text=text, speaker=speaker_name, language=language, **conservative
        )
        return np.asarray(wavs[0], dtype=np.float32), int(sr)
    except (RuntimeError, TypeError):
        pass  # Fall through to CPU fallback

    # --- Attempt 3: CPU + float32 fallback ---
    if not checkpoint_path:
        raise RuntimeError(
            "probability tensor error: no checkpoint_path for CPU fallback"
        )

    if _cpu_fallback_model is None:
        print("    [RETRY] loading CPU float32 model (one-time)...")
        _cpu_fallback_model = _load_tts_model(checkpoint_path, "cpu", torch.float32)

    cpu_params = {k: conservative[k] for k in safe_keys if k in conservative}
    print("    [RETRY] generating on CPU float32...")
    wavs, sr = _cpu_fallback_model.generate_custom_voice(
        text=text, speaker=speaker_name, language=language, **cpu_params
    )
    return np.asarray(wavs[0], dtype=np.float32), int(sr)


def _profile_from_rows(rows: list[SegmentRow]) -> dict[str, float]:
    if not rows:
        return {"chars_per_sec": 6.0, "f0_median": 140.0, "f0_std": 25.0}
    cps_values: list[float] = []
    f0_medians: list[float] = []
    f0_stds: list[float] = []
    for r in rows:
        n_chars = len(_normalize_text(r.text))
        cps_values.append(float(n_chars / max(r.duration_sec, 1e-6)))
        try:
            wav, _ = librosa.load(r.audio_path, sr=24000, mono=True)
            f0_med, f0_std = _f0_stats(wav, 24000)
            if f0_med > 0:
                f0_medians.append(f0_med)
            if f0_std > 0:
                f0_stds.append(f0_std)
        except Exception:
            continue
    return {
        "chars_per_sec": float(np.median(cps_values) if cps_values else 6.0),
        "f0_median": float(np.median(f0_medians) if f0_medians else 140.0),
        "f0_std": float(np.median(f0_stds) if f0_stds else 25.0),
    }


def main() -> int:
    # Force line-buffered stdout so progress shows through pipes/tee
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    sys.stderr.reconfigure(line_buffering=True)  # type: ignore[attr-defined]

    parser = argparse.ArgumentParser(
        description="Build high-quality voice dataset and run generation review."
    )
    parser.add_argument(
        "--source-mp4", default="/Users/rentamac/Desktop/서재형 대표님 음성_0821ver.mp4"
    )
    parser.add_argument(
        "--output-dir", default=f"workspace/imports/quality_build_{_timestamp()}"
    )
    parser.add_argument("--whisper-model", default="base")
    parser.add_argument("--min-seg-sec", type=float, default=3.0)
    parser.add_argument("--max-seg-sec", type=float, default=15.0)
    parser.add_argument("--split-top-db", type=float, default=28.0)
    parser.add_argument("--num-refs", type=int, default=8)
    parser.add_argument(
        "--speaker-filter", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--speaker-name", default="seojaehyung")
    parser.add_argument("--language", default="korean")
    parser.add_argument("--seed", type=int, default=20260223)
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--max-segments", type=int, default=0)
    parser.add_argument("--max-duration-min", type=float, default=0,
                        help="Limit source audio to N minutes (0 = no limit). Prevents OOM on Mac.")
    parser.add_argument("--base-speaker-model", default=_find_default_base_model())
    args = parser.parse_args()

    _seed_everything(int(args.seed))

    output_dir = Path(args.output_dir).expanduser().resolve()
    source_mp4 = Path(args.source_mp4).expanduser().resolve()
    full_wav = output_dir / "full_audio_24k.wav"
    segments_dir = output_dir / "segments"
    transcript_path = output_dir / "transcript.jsonl"
    filtered_path = output_dir / "filtered_segments.jsonl"
    train_raw_path = output_dir / "train_raw.jsonl"
    ref_bank_dir = output_dir / "ref_bank"
    test_samples_dir = output_dir / "test_samples"
    review_report_path = output_dir / "review_report.json"
    final_report_path = output_dir / "build_report.json"

    output_dir.mkdir(parents=True, exist_ok=True)

    if not source_mp4.exists():
        raise FileNotFoundError(f"Source MP4 not found: {source_mp4}")

    device, dtype = _choose_device()
    print(f"[INFO] device={device} dtype={dtype}")
    print(f"[INFO] output_dir={output_dir}")

    report: dict[str, Any] = {
        "args": vars(args),
        "device": device,
        "dtype": str(dtype),
        "paths": {
            "source_mp4": str(source_mp4),
            "full_audio_24k": str(full_wav),
            "segments_dir": str(segments_dir),
            "transcript": str(transcript_path),
            "filtered_segments": str(filtered_path),
            "train_raw": str(train_raw_path),
            "ref_bank_dir": str(ref_bank_dir),
            "review_report": str(review_report_path),
        },
        "phases": {},
    }

    max_dur_sec = float(args.max_duration_min) * 60.0 if float(args.max_duration_min) > 0 else 0
    print("[PHASE 1] MP4 audio extraction")
    extracted, src_duration = _extract_audio_if_needed(
        source_mp4, full_wav, bool(args.skip_extract), max_duration_sec=max_dur_sec
    )
    if max_dur_sec > 0:
        print(f"[INFO] duration limit: {float(args.max_duration_min):.1f} min")
    print(f"[INFO] source duration: {src_duration / 60.0:.2f} min")
    print(f"[INFO] extraction: {'done' if extracted else 'skipped'} -> {full_wav}")
    report["phases"]["phase1"] = {
        "source_duration_sec": src_duration,
        "max_duration_min": float(args.max_duration_min),
        "extracted": extracted,
        "wav": str(full_wav),
    }

    print("[PHASE 2] VAD-style intelligent segmentation")
    existing_segs = sorted(segments_dir.glob("seg_*.wav")) if segments_dir.exists() else []
    if existing_segs:
        print(f"[INFO] reusing {len(existing_segs)} existing segments")
        seg_paths = existing_segs
    else:
        seg_paths = _segment_audio(
            full_wav,
            segments_dir,
            min_seg_sec=float(args.min_seg_sec),
            max_seg_sec=float(args.max_seg_sec),
            split_top_db=float(args.split_top_db),
        )
    if int(args.max_segments) > 0:
        seg_paths = seg_paths[: int(args.max_segments)]
    if not seg_paths:
        raise RuntimeError("No segments were produced.")
    seg_durations = [float(sf.info(str(p)).duration) for p in seg_paths]
    print(f"[INFO] segments produced: {len(seg_paths)}")
    print(f"[INFO] segment duration mean: {float(np.mean(seg_durations)):.2f}s")
    report["phases"]["phase2"] = {
        "segment_count": len(seg_paths),
        "duration_mean": float(np.mean(seg_durations)),
        "duration_min": float(np.min(seg_durations)),
        "duration_max": float(np.max(seg_durations)),
    }

    print("[PHASE 3] High-quality transcription")
    try:
        whisper = WhisperModel(args.whisper_model, device="cpu", compute_type="int8")
    except Exception as e:
        raise RuntimeError(
            f"Failed to load faster-whisper model '{args.whisper_model}': {e}"
        ) from e

    transcript_rows: list[dict[str, Any]] = []
    for i, p in enumerate(seg_paths, start=1):
        text = _transcribe_ko(whisper, p)
        transcript_rows.append({"audio": p.name, "text": text})
        if i % 25 == 0 or i == len(seg_paths):
            print(f"[INFO] transcribed {i}/{len(seg_paths)}")

    transcript_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in transcript_rows) + "\n",
        encoding="utf-8",
    )
    avg_text_len = (
        float(np.mean([len(r["text"]) for r in transcript_rows]))
        if transcript_rows
        else 0.0
    )
    print(
        f"[INFO] transcript rows: {len(transcript_rows)} avg_text_len={avg_text_len:.2f}"
    )
    report["phases"]["phase3"] = {
        "whisper_model": args.whisper_model,
        "row_count": len(transcript_rows),
        "avg_text_len": avg_text_len,
        "transcript_jsonl": str(transcript_path),
    }


    # Free whisper before heavy filtering phase to save memory
    del whisper
    gc.collect()
    print("[INFO] released whisper model memory")

    print("[PHASE 4] Quality filtering")
    initial_rows: list[SegmentRow] = []
    for row in transcript_rows:
        p = segments_dir / str(row["audio"])
        d = float(sf.info(str(p)).duration)
        initial_rows.append(
            SegmentRow(
                audio_path=str(p.resolve()),
                audio_name=p.name,
                text=str(row.get("text", "")).strip(),
                duration_sec=d,
                metrics={},
                speaker_cos=1.0,
                audio_quality_score=0.0,
                duration_score=0.0,
                ref_score=0.0,
            )
        )

    n0 = len(initial_rows)
    duration_kept = [
        r
        for r in initial_rows
        if float(args.min_seg_sec) <= r.duration_sec <= float(args.max_seg_sec)
    ]
    print(
        f"[FILTER] duration: kept {len(duration_kept)}/{n0}, dropped {n0 - len(duration_kept)}"
    )

    signal_kept: list[SegmentRow] = []
    for r in duration_kept:
        wav, _ = librosa.load(r.audio_path, sr=24000, mono=True)
        metrics = _signal_metrics(wav, 24000)
        r.metrics = metrics
        if (
            float(metrics.get("snr", 0.0)) > 12.0
            and float(metrics.get("clipping_ratio", 1.0)) < 0.005
            and float(metrics.get("silent_ratio", 1.0)) < 0.50
            and float(metrics.get("rms", 0.0)) > 0.005
        ):
            signal_kept.append(r)
    print(
        f"[FILTER] signal: kept {len(signal_kept)}/{len(duration_kept)}, "
        f"dropped {len(duration_kept) - len(signal_kept)}"
    )

    text_kept = [
        r
        for r in signal_kept
        if len(_normalize_text(r.text)) >= 10 and not _is_repetitive_text(r.text)
    ]
    print(
        f"[FILTER] text: kept {len(text_kept)}/{len(signal_kept)}, dropped {len(signal_kept) - len(text_kept)}"
    )

    speaker_filter_enabled = bool(args.speaker_filter)
    speaker_kept: list[SegmentRow] = text_kept
    speaker_drop = 0
    base_model: Qwen3TTSModel | None = None
    centroid: np.ndarray | None = None
    if speaker_filter_enabled:
        try:
            print(
                f"[INFO] loading base model for speaker filter: {args.base_speaker_model}"
            )
            base_model = _load_tts_model(str(args.base_speaker_model), device, dtype)
            embs: list[np.ndarray] = []
            for r in text_kept:
                wav, _ = librosa.load(r.audio_path, sr=24000, mono=True)
                emb = _extract_emb_from_wav(base_model, wav.astype(np.float32))
                embs.append(emb)
            if embs:
                centroid = _unit(np.mean(np.stack(embs, axis=0), axis=0))
                tmp: list[SegmentRow] = []
                for r, emb in zip(text_kept, embs):
                    r.speaker_cos = _cos(emb, centroid)
                    if r.speaker_cos >= 0.92:
                        tmp.append(r)
                speaker_kept = tmp
            speaker_drop = len(text_kept) - len(speaker_kept)
            print(
                f"[FILTER] speaker: kept {len(speaker_kept)}/{len(text_kept)}, dropped {speaker_drop}"
            )
        except Exception as e:
            print(f"[INFO] speaker filter skipped (base model unavailable): {e}")
            speaker_kept = text_kept
            speaker_drop = 0
    else:
        print(
            f"[FILTER] speaker: kept {len(speaker_kept)}/{len(text_kept)}, dropped 0 (disabled)"
        )

    if not speaker_kept:
        raise RuntimeError(
            "All samples were filtered out. Adjust filtering thresholds."
        )

    for r in speaker_kept:
        r.audio_quality_score = _audio_quality_score(r.metrics)
        r.duration_score = _duration_score(r.duration_sec)
        r.ref_score = (
            0.4 * r.audio_quality_score
            + 0.3 * float(r.speaker_cos)
            + 0.3 * r.duration_score
        )

    filtered_path.write_text(
        "\n".join(
            json.dumps(
                {
                    "audio": r.audio_name,
                    "text": r.text,
                    "duration_sec": r.duration_sec,
                    "metrics": r.metrics,
                    "speaker_cos": r.speaker_cos,
                    "audio_quality_score": r.audio_quality_score,
                    "duration_score": r.duration_score,
                    "ref_score": r.ref_score,
                },
                ensure_ascii=False,
            )
            for r in speaker_kept
        )
        + "\n",
        encoding="utf-8",
    )

    report["phases"]["phase4"] = {
        "input_count": n0,
        "duration_kept": len(duration_kept),
        "signal_kept": len(signal_kept),
        "text_kept": len(text_kept),
        "speaker_kept": len(speaker_kept),
        "speaker_dropped": speaker_drop,
        "filtered_jsonl": str(filtered_path),
    }

    print("[PHASE 5] Reference bank construction")
    rows_sorted = sorted(speaker_kept, key=lambda x: x.ref_score, reverse=True)
    top_refs = rows_sorted[: max(1, int(args.num_refs))]

    ref_bank_dir.mkdir(parents=True, exist_ok=True)
    for old in sorted(ref_bank_dir.glob("*.wav")):
        old.unlink()

    ref_rows: list[dict[str, Any]] = []
    for i, r in enumerate(top_refs, start=1):
        dst = ref_bank_dir / f"ref_{i:02d}_{Path(r.audio_name).stem}.wav"
        shutil.copy2(r.audio_path, dst)
        ref_rows.append(
            {
                "rank": i,
                "src_audio": r.audio_path,
                "dst_audio": str(dst.resolve()),
                "text": r.text,
                "duration_sec": r.duration_sec,
                "audio_quality_score": r.audio_quality_score,
                "speaker_centroid_cos": r.speaker_cos,
                "duration_score": r.duration_score,
                "ref_score": r.ref_score,
            }
        )

    best_ref_src = Path(top_refs[0].audio_path)
    best_ref = ref_bank_dir / "best_ref.wav"
    shutil.copy2(best_ref_src, best_ref)
    ref_meta = {
        "num_refs": len(top_refs),
        "best_ref": str(best_ref.resolve()),
        "rows": ref_rows,
    }
    (ref_bank_dir / "ref_bank.json").write_text(
        json.dumps(ref_meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[INFO] reference bank size: {len(top_refs)}")
    print(f"[INFO] best ref: {best_ref}")
    report["phases"]["phase5"] = ref_meta

    print("[PHASE 6] train_raw.jsonl generation")
    train_rows = [
        {
            "audio": str(Path(r.audio_path).resolve()),
            "text": r.text,
            "ref_audio": str(best_ref.resolve()),
            "language": "korean",
        }
        for r in speaker_kept
    ]
    train_raw_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in train_rows) + "\n",
        encoding="utf-8",
    )
    total_dur = float(sum(r.duration_sec for r in speaker_kept))
    avg_dur = total_dur / max(len(speaker_kept), 1)
    avg_txt = (
        float(np.mean([len(r.text) for r in speaker_kept])) if speaker_kept else 0.0
    )
    print(
        f"[INFO] dataset stats: samples={len(speaker_kept)} total={total_dur / 60.0:.2f}min "
        f"avg_dur={avg_dur:.2f}s avg_text_len={avg_txt:.2f}"
    )
    report["phases"]["phase6"] = {
        "samples": len(speaker_kept),
        "total_duration_sec": total_dur,
        "avg_duration_sec": avg_dur,
        "avg_text_len": avg_txt,
        "train_raw_jsonl": str(train_raw_path),
    }

    phase7_report: dict[str, Any] = {
        "executed": False,
        "reason": "",
        "review_report": str(review_report_path),
    }
    if bool(args.skip_test):
        phase7_report["reason"] = "--skip-test enabled"
        print("[PHASE 7] skipped (--skip-test)")
    elif not str(args.checkpoint).strip():
        phase7_report["reason"] = "--checkpoint not provided"
        print("[PHASE 7] skipped (no checkpoint)")
    else:
        print("[PHASE 7] Sample generation + comprehensive review")
        # Reload whisper for review (was freed after Phase 3 to save memory)
        whisper = WhisperModel(args.whisper_model, device="cpu", compute_type="int8")
        checkpoint = str(Path(args.checkpoint).expanduser())
        if base_model is None:
            try:
                base_model = _load_tts_model(
                    str(args.base_speaker_model), device, dtype
                )
            except Exception as e:
                raise RuntimeError(f"Phase 7 requires base speaker model: {e}") from e
        try:
            custom_model = _load_tts_model(checkpoint, device, dtype)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load checkpoint model '{checkpoint}': {e}"
            ) from e

        profile = _profile_from_rows(speaker_kept)
        ref_wav, _ = librosa.load(str(best_ref), sr=24000, mono=True)
        ref_emb = _extract_emb_from_wav(base_model, ref_wav.astype(np.float32))

        sim_params = {
            "temperature": 0.45,
            "top_k": 24,
            "top_p": 0.88,
            "repetition_penalty": 1.03,
            "max_new_tokens": 260,
            "subtalker_temperature": 0.55,
            "subtalker_top_k": 24,
            "subtalker_top_p": 0.88,
        }

        candidates: list[ReviewRow] = []
        test_samples_dir.mkdir(parents=True, exist_ok=True)
        candidate_dir = test_samples_dir / "candidates"
        candidate_dir.mkdir(parents=True, exist_ok=True)

        for text_idx, text in enumerate(TEST_TEXTS, start=1):
            for j in range(3):
                seed = int(args.seed) + text_idx * 100 + j
                out_wav = candidate_dir / f"text{text_idx:02d}_seed{seed}.wav"
                try:
                    _seed_everything(seed)
                    wav, sr = _generate_with_fallback(
                        custom_model,
                        text=text,
                        speaker_name=args.speaker_name,
                        language=args.language,
                        params=sim_params,
                        checkpoint_path=checkpoint,
                    )
                    sf.write(str(out_wav), wav, sr)
                    wav24 = (
                        wav
                        if int(sr) == 24000
                        else librosa.resample(wav, orig_sr=int(sr), target_sr=24000)
                    )
                    emb = _extract_emb_from_wav(base_model, wav24.astype(np.float32))
                    speaker_cos = _cos(emb, ref_emb)

                    asr_text = _transcribe_ko(whisper, out_wav)
                    asr_sim = _asr_similarity(text, asr_text)

                    dur = float(len(wav24) / 24000.0)
                    target_dur = float(
                        len(_normalize_text(text)) / max(profile["chars_per_sec"], 1e-6)
                    )
                    speed_err = abs(math.log((dur + 1e-6) / max(target_dur, 1e-6)))
                    speed_score = float(max(0.0, 1.0 - 1.5 * speed_err))

                    f0_med, f0_std = _f0_stats(wav24, 24000)
                    f0_std_err = abs(
                        math.log((f0_std + 1e-3) / max(profile["f0_std"], 1e-3))
                    )
                    f0_med_err = abs(
                        math.log((f0_med + 1e-3) / max(profile["f0_median"], 1e-3))
                    )
                    f0_consistency = float(
                        max(0.0, 1.0 - 0.65 * f0_std_err - 0.35 * f0_med_err)
                    )

                    final_score = (
                        0.55 * speaker_cos
                        + 0.15 * asr_sim
                        + 0.15 * speed_score
                        + 0.15 * f0_consistency
                    )
                    row = ReviewRow(
                        text_index=text_idx,
                        text=text,
                        seed=seed,
                        wav=str(out_wav.resolve()),
                        speaker_cos=speaker_cos,
                        asr_sim=asr_sim,
                        duration_sec=dur,
                        target_duration_sec=target_dur,
                        speed_score=speed_score,
                        f0_median=f0_med,
                        f0_std=f0_std,
                        f0_consistency=f0_consistency,
                        final_score=float(final_score),
                        asr_text=asr_text,
                    )
                except Exception as e:
                    row = ReviewRow(
                        text_index=text_idx,
                        text=text,
                        seed=seed,
                        wav=str(out_wav.resolve()),
                        speaker_cos=0.0,
                        asr_sim=0.0,
                        duration_sec=0.0,
                        target_duration_sec=0.0,
                        speed_score=0.0,
                        f0_median=0.0,
                        f0_std=0.0,
                        f0_consistency=0.0,
                        final_score=-1.0,
                        asr_text="",
                        error=str(e),
                    )
                candidates.append(row)
                print(
                    f"[REVIEW text={text_idx} seed={seed}] cos={row.speaker_cos:.4f} "
                    f"asr={row.asr_sim:.4f} speed={row.speed_score:.4f} f0={row.f0_consistency:.4f} "
                    f"score={row.final_score:.4f}"
                )

        best_rows: list[ReviewRow] = []
        for text_idx in range(1, len(TEST_TEXTS) + 1):
            pool = [r for r in candidates if r.text_index == text_idx]
            pool_sorted = sorted(pool, key=lambda x: x.final_score, reverse=True)
            if not pool_sorted:
                continue
            best = pool_sorted[0]
            dst = test_samples_dir / f"sample_{text_idx:02d}.wav"
            try:
                shutil.copy2(best.wav, dst)
            except Exception:
                pass
            best_rows.append(best)

        print("text_idx | score  | spk_cos | asr_sim | speed  | f0_cons")
        for r in best_rows:
            print(
                f"{r.text_index:>7d} | {r.final_score:>6.4f} | {r.speaker_cos:>7.4f} | "
                f"{r.asr_sim:>7.4f} | {r.speed_score:>6.4f} | {r.f0_consistency:>7.4f}"
            )

        review_report = {
            "checkpoint": checkpoint,
            "speaker_name": args.speaker_name,
            "language": args.language,
            "best_ref": str(best_ref.resolve()),
            "profile": profile,
            "generation_params": sim_params,
            "texts": TEST_TEXTS,
            "best_per_text": [asdict(r) for r in best_rows],
            "all_candidates": [
                asdict(r)
                for r in sorted(candidates, key=lambda x: x.final_score, reverse=True)
            ],
        }
        review_report_path.write_text(
            json.dumps(review_report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        phase7_report = {
            "executed": True,
            "review_report": str(review_report_path),
            "best_samples": [asdict(r) for r in best_rows],
            "candidate_count": len(candidates),
        }

        del custom_model
        del whisper
        gc.collect()

    report["phases"]["phase7"] = phase7_report

    final_report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[INFO] final report: {final_report_path}")

    if base_model is not None:
        del base_model
    global _cpu_fallback_model
    if _cpu_fallback_model is not None:
        del _cpu_fallback_model
        _cpu_fallback_model = None
    gc.collect()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[ERROR] {exc}")
        raise SystemExit(1)
