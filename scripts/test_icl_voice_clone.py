#!/usr/bin/env python3
"""Quick ICL (In-Context Learning) voice clone test.

Uses Qwen3-TTS Base model's generate_voice_clone() with ICL mode
(x_vector_only_mode=False) — no fine-tuning needed.
This prepends actual reference audio tokens as context, preserving
prosody/rhythm better than embedding-only approaches.
"""

from __future__ import annotations

import sys
import gc
import json
import time
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf
import torch
from faster_whisper import WhisperModel
from qwen_tts import Qwen3TTSModel

# Force line-buffered stdout
sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]

# ---------- Test texts (same as build_and_test_voice.py) ----------
TEST_TEXTS = [
    "안녕하세요. 오늘 시장 이야기를 간단하게 말씀드리겠습니다.",
    "최근 반도체 섹터가 강세를 보이고 있는데, 그 이유를 분석해보겠습니다.",
    "투자에서 가장 중요한 것은 리스크 관리입니다. 절대 잊지 마세요.",
    "자, 그러면 오늘의 주요 종목들을 하나씩 살펴볼까요?",
    "이 기업의 실적이 예상보다 좋았고, 앞으로의 전망도 밝습니다.",
    "여러분, 감사합니다. 다음 시간에 또 뵙겠습니다.",
]

# ---------- Reference audio bank with transcripts ----------
REF_BANK = [
    {
        "audio": "workspace/imports/quality_test_quick/ref_bank/ref_01_seg_0194.wav",
        "text": "묻혀 있고 인도네시아도 묻혀 있고 미국도 묻혀 있고 그래요. 그래서 하여튼 이 부분도",
    },
    {
        "audio": "workspace/imports/quality_test_quick/ref_bank/ref_02_seg_0112.wav",
        "text": "사람의 성장곡선을 따졌을 때 21살, 22살이면",
    },
    {
        "audio": "workspace/imports/quality_test_quick/ref_bank/ref_03_seg_0159.wav",
        "text": "마이크로소프트, 메타, 이건 물리적 AI에 관련된 거예요. 샤워폼, BYD.",
    },
]


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v.copy()


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb + 1e-12))


def _extract_emb(model: Qwen3TTSModel, wav_24k: np.ndarray) -> np.ndarray:
    emb = model.model.extract_speaker_embedding(wav_24k, 24000)
    if isinstance(emb, torch.Tensor):
        emb = emb.detach().float().cpu().numpy()
    return _unit(np.asarray(emb, dtype=np.float32))


def main() -> int:
    out_dir = Path("workspace/imports/icl_test")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Model loading (BASE model, CPU float32 for stability) ----
    base_path = str(
        Path.home()
        / ".cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-Base"
        / "snapshots/5d83992436eae1d760afd27aff78a71d676296fc"
    )
    if not Path(base_path).exists():
        base_path = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"

    print(f"[INFO] Loading base model from: {base_path}")
    print("[INFO] Using CPU float32 for stable generation")
    model = Qwen3TTSModel.from_pretrained(
        base_path, device_map="cpu", dtype=torch.float32
    )
    model.model.eval()
    print(f"[INFO] Model loaded. tts_model_type={model.model.tts_model_type}")

    # ---- Load whisper for ASR evaluation ----
    print("[INFO] Loading whisper large-v3 for ASR evaluation...")
    whisper = WhisperModel("large-v3", device="cpu", compute_type="int8")

    # ---- Reference embedding (for cosine comparison) ----
    best_ref_path = "workspace/imports/quality_test_quick/ref_bank/best_ref.wav"
    ref_wav, _ = librosa.load(best_ref_path, sr=24000, mono=True)
    ref_emb = _extract_emb(model, ref_wav.astype(np.float32))

    # ---- Official generation params (from Qwen3-TTS defaults) ----
    gen_kwargs_official = dict(
        do_sample=True,
        temperature=0.9,
        top_k=50,
        top_p=1.0,
        repetition_penalty=1.05,
        subtalker_dosample=True,
        subtalker_top_k=50,
        subtalker_top_p=1.0,
        subtalker_temperature=0.9,
        max_new_tokens=2048,
    )

    # ---- Slightly conservative params for stability ----
    gen_kwargs_conservative = dict(
        do_sample=True,
        temperature=0.7,
        top_k=30,
        top_p=0.9,
        repetition_penalty=1.05,
        subtalker_dosample=True,
        subtalker_top_k=30,
        subtalker_top_p=0.9,
        subtalker_temperature=0.7,
        max_new_tokens=2048,
    )

    results: list[dict[str, Any]] = []

    # ---- Test 3 modes: ICL, x-vector only, and two param sets ----
    configs = [
        ("ICL_official", False, gen_kwargs_official),
        ("ICL_conservative", False, gen_kwargs_conservative),
        ("xvec_only_official", True, gen_kwargs_official),
    ]

    for config_name, xvec_only, gen_kwargs in configs:
        print(f"\n{'=' * 60}")
        print(f"[CONFIG] {config_name} (x_vector_only={xvec_only})")
        print(f"{'=' * 60}")

        for text_idx, text in enumerate(TEST_TEXTS, start=1):
            # Use ref 1 for all (best quality ref)
            ref = REF_BANK[0]
            ref_audio_path = str(Path(ref["audio"]).resolve())
            ref_text = ref["text"] if not xvec_only else None

            out_wav_path = out_dir / f"{config_name}_text{text_idx:02d}.wav"

            try:
                t0 = time.time()
                wavs, sr = model.generate_voice_clone(
                    text=text,
                    language="Korean",
                    ref_audio=ref_audio_path,
                    ref_text=ref_text,
                    x_vector_only_mode=xvec_only,
                    **gen_kwargs,
                )
                elapsed = time.time() - t0

                wav = np.asarray(wavs[0], dtype=np.float32)
                sf.write(str(out_wav_path), wav, sr)

                # Resample to 24k for evaluation
                wav24 = (
                    wav
                    if int(sr) == 24000
                    else librosa.resample(wav, orig_sr=int(sr), target_sr=24000)
                )

                # Speaker cosine
                gen_emb = _extract_emb(model, wav24.astype(np.float32))
                spk_cos = _cos(gen_emb, ref_emb)

                # ASR check
                segments, _ = whisper.transcribe(
                    str(out_wav_path),
                    language="ko",
                    beam_size=5,
                    vad_filter=True,
                    condition_on_previous_text=False,
                )
                asr_text = "".join(seg.text for seg in segments).strip()

                # Duration
                dur = float(len(wav24) / 24000.0)

                row = {
                    "config": config_name,
                    "text_idx": text_idx,
                    "text": text,
                    "cos": round(spk_cos, 4),
                    "asr_text": asr_text,
                    "duration_sec": round(dur, 2),
                    "gen_time_sec": round(elapsed, 1),
                    "wav": str(out_wav_path.resolve()),
                    "error": None,
                }
                print(
                    f"  [text={text_idx}] cos={spk_cos:.4f} dur={dur:.1f}s "
                    f'gen={elapsed:.1f}s asr="{asr_text[:50]}..."'
                )

            except Exception as e:
                row = {
                    "config": config_name,
                    "text_idx": text_idx,
                    "text": text,
                    "cos": 0.0,
                    "asr_text": "",
                    "duration_sec": 0.0,
                    "gen_time_sec": 0.0,
                    "wav": str(out_wav_path.resolve()),
                    "error": str(e),
                }
                print(f"  [text={text_idx}] ERROR: {e}")

            results.append(row)

    # ---- Also test with multiple ref audios (ICL with 3 refs) ----
    print(f"\n{'=' * 60}")
    print("[CONFIG] ICL_multi_ref (3 reference audios, conservative params)")
    print(f"{'=' * 60}")

    ref_audios = [str(Path(r["audio"]).resolve()) for r in REF_BANK]
    ref_texts = [r["text"] for r in REF_BANK]

    # Pre-build prompt items for multi-ref
    try:
        prompt_items = model.create_voice_clone_prompt(
            ref_audio=ref_audios,
            ref_text=ref_texts,
            x_vector_only_mode=False,
        )

        for text_idx, text in enumerate(TEST_TEXTS, start=1):
            out_wav_path = out_dir / f"ICL_multi_ref_text{text_idx:02d}.wav"
            try:
                t0 = time.time()
                wavs, sr = model.generate_voice_clone(
                    text=text,
                    language="Korean",
                    voice_clone_prompt=prompt_items,
                    **gen_kwargs_conservative,
                )
                elapsed = time.time() - t0

                wav = np.asarray(wavs[0], dtype=np.float32)
                sf.write(str(out_wav_path), wav, sr)
                wav24 = (
                    wav
                    if int(sr) == 24000
                    else librosa.resample(wav, orig_sr=int(sr), target_sr=24000)
                )
                gen_emb = _extract_emb(model, wav24.astype(np.float32))
                spk_cos = _cos(gen_emb, ref_emb)

                segments, _ = whisper.transcribe(
                    str(out_wav_path),
                    language="ko",
                    beam_size=5,
                    vad_filter=True,
                    condition_on_previous_text=False,
                )
                asr_text = "".join(seg.text for seg in segments).strip()
                dur = float(len(wav24) / 24000.0)

                row = {
                    "config": "ICL_multi_ref",
                    "text_idx": text_idx,
                    "text": text,
                    "cos": round(spk_cos, 4),
                    "asr_text": asr_text,
                    "duration_sec": round(dur, 2),
                    "gen_time_sec": round(elapsed, 1),
                    "wav": str(out_wav_path.resolve()),
                    "error": None,
                }
                print(
                    f"  [text={text_idx}] cos={spk_cos:.4f} dur={dur:.1f}s "
                    f'gen={elapsed:.1f}s asr="{asr_text[:50]}..."'
                )
            except Exception as e:
                row = {
                    "config": "ICL_multi_ref",
                    "text_idx": text_idx,
                    "text": text,
                    "cos": 0.0,
                    "asr_text": "",
                    "duration_sec": 0.0,
                    "gen_time_sec": 0.0,
                    "wav": str(out_wav_path.resolve()),
                    "error": str(e),
                }
                print(f"  [text={text_idx}] ERROR: {e}")
            results.append(row)
    except Exception as e:
        print(f"[ERROR] Multi-ref prompt creation failed: {e}")

    # ---- Summary ----
    print(f"\n{'=' * 60}")
    print("[SUMMARY]")
    print(f"{'=' * 60}")
    print(f"{'config':<25} {'text':>4} {'cos':>7} {'dur':>6} {'asr_ok':>6}")
    print("-" * 55)
    for r in results:
        asr_ok = "YES" if len(r.get("asr_text", "")) > 5 else "NO"
        if r.get("error"):
            asr_ok = "ERR"
        print(
            f"{r['config']:<25} {r['text_idx']:>4} {r['cos']:>7.4f} "
            f"{r['duration_sec']:>6.1f} {asr_ok:>6}"
        )

    # ---- Save report ----
    report_path = out_dir / "icl_test_report.json"
    report_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n[INFO] Report saved: {report_path}")
    print(f"[INFO] Audio files in: {out_dir}")

    del model, whisper
    gc.collect()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[ERROR] {exc}")
        raise SystemExit(1)
