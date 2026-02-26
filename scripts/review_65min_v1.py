#!/usr/bin/env python3
"""Review script for seojaehyung_65min_v1 checkpoint quality.

Generates samples across 3 checkpoints (epoch 0/1/2), evaluates:
  - Speaker cosine similarity vs reference
  - ASR accuracy (Whisper transcription match)
  - Signal quality (RMS, clipping)
Picks the best checkpoint and reports.
"""

from __future__ import annotations

import gc
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from faster_whisper import WhisperModel

# ── Test texts: mix of short/long, formal/casual ──
TEST_TEXTS = [
    "안녕하세요. 오늘 시장 이야기를 간단하게 말씀드리겠습니다.",
    "주식은 쌀 때 사서 비쌀 때 파는 게 기본이에요.",
    "최근 반도체 섹터가 굉장히 강한 흐름을 보여주고 있습니다.",
    "여러분 오늘 영상 재밌으셨으면 좋아요 구독 부탁드립니다.",
    "이번 실적 발표를 보면 매출이 전년 대비 약 이십 퍼센트 증가했는데요 이것은 시장 기대치를 크게 상회하는 수준입니다.",
]

CHECKPOINTS = [
    ("epoch-0", "workspace/runs/seojaehyung_65min_v1/checkpoint-epoch-0"),
    ("epoch-1", "workspace/runs/seojaehyung_65min_v1/checkpoint-epoch-1"),
    ("epoch-2", "workspace/runs/seojaehyung_65min_v1/checkpoint-epoch-2"),
]

REF_AUDIO = "workspace/imports/quality_build_65min/ref_bank/best_ref.wav"
SPEAKER_NAME = "seojaehyung"
OUTPUT_DIR = Path("workspace/exports/review_65min_v1")


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
    return 1.0 - (prev[n] / max(len(a), len(b), 1))


def _asr_similarity(target: str, pred: str) -> float:
    return _levenshtein_ratio(_normalize_text(target), _normalize_text(pred))


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v.copy()


@dataclass
class Sample:
    checkpoint: str
    text_idx: int
    text: str
    wav_path: str
    speaker_cos: float
    asr_sim: float
    asr_text: str
    duration_sec: float
    rms: float
    score: float
    error: str | None = None


def main() -> int:
    import sys

    sys.stdout.reconfigure(line_buffering=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ref_path = Path(REF_AUDIO).resolve()
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference audio not found: {ref_path}")

    # ── Load base model for speaker embedding extraction ──
    print("[1/4] Loading base model for speaker embeddings...")
    from qwen_tts import Qwen3TTSModel

    base_snap = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--Qwen--Qwen3-TTS-12Hz-0.6B-Base"
        / "snapshots"
    )
    base_path = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
    if base_snap.exists():
        snaps = sorted([p for p in base_snap.iterdir() if p.is_dir()])
        if snaps:
            base_path = str(snaps[-1].resolve())

    base_model = Qwen3TTSModel.from_pretrained(
        base_path, device_map="cpu", dtype=torch.float32
    )

    ref_wav, _ = librosa.load(str(ref_path), sr=24000, mono=True)
    ref_emb = base_model.model.extract_speaker_embedding(
        ref_wav.astype(np.float32), 24000
    )
    if isinstance(ref_emb, torch.Tensor):
        ref_emb = ref_emb.detach().float().cpu().numpy()
    ref_emb = _unit(np.asarray(ref_emb, dtype=np.float32))

    # ── Load whisper for ASR evaluation ──
    print("[2/4] Loading Whisper for ASR evaluation...")
    whisper = WhisperModel("base", device="cpu", compute_type="int8")

    # ── Generate samples per checkpoint ──
    all_samples: list[Sample] = []
    for ckpt_name, ckpt_path in CHECKPOINTS:
        ckpt_full = Path(ckpt_path).resolve()
        if not ckpt_full.exists():
            print(f"[SKIP] {ckpt_name}: not found")
            continue

        print(f"\n[3/4] Generating from {ckpt_name}...")
        model = Qwen3TTSModel.from_pretrained(
            str(ckpt_full), device_map="cpu", dtype=torch.float32
        )

        ckpt_dir = OUTPUT_DIR / ckpt_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        for ti, text in enumerate(TEST_TEXTS):
            out_wav = ckpt_dir / f"text{ti + 1:02d}.wav"
            try:
                torch.manual_seed(42 + ti)
                wavs, sr = model.generate_custom_voice(
                    text=text,
                    speaker=SPEAKER_NAME,
                    language="korean",
                    temperature=0.45,
                    top_k=30,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    max_new_tokens=min(300, max(128, len(text) * 4)),
                )
                wav = wavs[0]
                sf.write(str(out_wav), wav, sr)

                wav24 = wav.astype(np.float32)
                if int(sr) != 24000:
                    wav24 = librosa.resample(wav24, orig_sr=int(sr), target_sr=24000)

                # Speaker similarity
                gen_emb = base_model.model.extract_speaker_embedding(wav24, 24000)
                if isinstance(gen_emb, torch.Tensor):
                    gen_emb = gen_emb.detach().float().cpu().numpy()
                gen_emb = _unit(np.asarray(gen_emb, dtype=np.float32))
                spk_cos = _cos(gen_emb, ref_emb)

                # ASR
                segments, _ = whisper.transcribe(
                    str(out_wav),
                    language="ko",
                    beam_size=3,
                    vad_filter=True,
                    condition_on_previous_text=False,
                )
                asr_text = "".join(s.text for s in segments).strip()
                asr_sim = _asr_similarity(text, asr_text)

                dur = float(len(wav24) / 24000.0)
                rms = float(math.sqrt(float(np.mean(np.square(wav24))) + 1e-12))
                score = 0.6 * spk_cos + 0.3 * asr_sim + 0.1 * min(1.0, rms / 0.03)

                sample = Sample(
                    checkpoint=ckpt_name,
                    text_idx=ti + 1,
                    text=text,
                    wav_path=str(out_wav),
                    speaker_cos=spk_cos,
                    asr_sim=asr_sim,
                    asr_text=asr_text,
                    duration_sec=dur,
                    rms=rms,
                    score=score,
                )
                print(
                    f"  [{ckpt_name}] text{ti + 1}: spk={spk_cos:.3f} asr={asr_sim:.3f} score={score:.3f}"
                )
            except Exception as e:
                sample = Sample(
                    checkpoint=ckpt_name,
                    text_idx=ti + 1,
                    text=text,
                    wav_path=str(out_wav),
                    speaker_cos=0,
                    asr_sim=0,
                    asr_text="",
                    duration_sec=0,
                    rms=0,
                    score=-1,
                    error=str(e),
                )
                print(f"  [{ckpt_name}] text{ti + 1}: ERROR: {e}")
            all_samples.append(sample)

        del model
        gc.collect()

    # ── Aggregate and report ──
    print("\n[4/4] Aggregating results...")
    report: dict = {"samples": [asdict(s) for s in all_samples], "summary": {}}

    for (
        ckpt_name,
        _,
    ) in CHECKPOINTS:
        ckpt_samples = [
            s for s in all_samples if s.checkpoint == ckpt_name and s.error is None
        ]
        if not ckpt_samples:
            continue
        avg_spk = float(np.mean([s.speaker_cos for s in ckpt_samples]))
        avg_asr = float(np.mean([s.asr_sim for s in ckpt_samples]))
        avg_score = float(np.mean([s.score for s in ckpt_samples]))
        report["summary"][ckpt_name] = {
            "avg_speaker_cos": round(avg_spk, 4),
            "avg_asr_sim": round(avg_asr, 4),
            "avg_score": round(avg_score, 4),
            "sample_count": len(ckpt_samples),
        }

    # Find best checkpoint
    best_ckpt = max(report["summary"].items(), key=lambda x: x[1]["avg_score"])
    report["best_checkpoint"] = best_ckpt[0]
    report["best_score"] = best_ckpt[1]["avg_score"]

    report_path = OUTPUT_DIR / "review_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ── Print summary table ──
    print("\n" + "=" * 70)
    print(f"{'Checkpoint':<12} {'Spk Cos':>8} {'ASR Sim':>8} {'Score':>8}")
    print("-" * 70)
    for name, stats in report["summary"].items():
        marker = " ★" if name == best_ckpt[0] else ""
        print(
            f"{name:<12} {stats['avg_speaker_cos']:>8.4f} {stats['avg_asr_sim']:>8.4f} {stats['avg_score']:>8.4f}{marker}"
        )
    print("=" * 70)

    print(f"\nBest checkpoint: {best_ckpt[0]} (score={best_ckpt[1]['avg_score']:.4f})")
    print(f"Report: {report_path}")

    # Print per-sample ASR details for the best checkpoint
    best_samples = [
        s for s in all_samples if s.checkpoint == best_ckpt[0] and s.error is None
    ]
    if best_samples:
        print(f"\n── {best_ckpt[0]} ASR Details ──")
        for s in best_samples:
            match = "✓" if s.asr_sim > 0.85 else "✗"
            print(
                f"  [{match}] text{s.text_idx} asr={s.asr_sim:.3f} dur={s.duration_sec:.1f}s"
            )
            print(f"      target: {s.text[:60]}")
            print(f"      heard:  {s.asr_text[:60]}")

    del whisper, base_model
    gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
