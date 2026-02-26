#!/usr/bin/env python3
"""ICL voice clone test — no finetuning, just base model + reference audio."""

from __future__ import annotations

import gc
import json
import math
import re
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from faster_whisper import WhisperModel

TEST_TEXTS = [
    "안녕하세요. 오늘 시장 이야기를 간단하게 말씀드리겠습니다.",
    "주식은 쌀 때 사서 비쌀 때 파는 게 기본이에요.",
    "최근 반도체 섹터가 굉장히 강한 흐름을 보여주고 있습니다.",
    "여러분 오늘 영상 재밌으셨으면 좋아요 구독 부탁드립니다.",
    "이번 실적 발표를 보면 매출이 전년 대비 약 이십 퍼센트 증가했는데요 이것은 시장 기대치를 크게 상회하는 수준입니다.",
]

REF_AUDIO = "workspace/imports/quality_build_65min/ref_bank/best_ref.wav"
REF_TEXT = "거기서 나와요. 그러니까 매매에서 나온 거 아니에요. 그분들 꼭 아셨으면 좋겠어요. 다음 페이지."
OUTPUT_DIR = Path("workspace/exports/icl_test_65min")


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


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v.copy()


def main() -> int:
    import sys

    sys.stdout.reconfigure(line_buffering=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ref_path = Path(REF_AUDIO).resolve()

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

    print("[1/3] Loading base model...")
    model = Qwen3TTSModel.from_pretrained(
        base_path, device_map="cpu", dtype=torch.float32
    )

    ref_wav, _ = librosa.load(str(ref_path), sr=24000, mono=True)
    ref_emb = model.model.extract_speaker_embedding(ref_wav.astype(np.float32), 24000)
    if isinstance(ref_emb, torch.Tensor):
        ref_emb = ref_emb.detach().float().cpu().numpy()
    ref_emb = _unit(np.asarray(ref_emb, dtype=np.float32))

    print("[2/3] Loading Whisper for evaluation...")
    whisper = WhisperModel("base", device="cpu", compute_type="int8")

    print("[3/3] Generating ICL voice clones...")
    results = []
    for ti, text in enumerate(TEST_TEXTS):
        out_wav = OUTPUT_DIR / f"icl_text{ti + 1:02d}.wav"
        torch.manual_seed(42 + ti)
        try:
            wavs, sr = model.generate_voice_clone(
                text=text,
                language="korean",
                ref_audio=str(ref_path),
                ref_text=REF_TEXT,
                x_vector_only_mode=False,
                do_sample=True,
                subtalker_dosample=True,
                temperature=0.5,
                top_k=30,
                top_p=0.9,
                repetition_penalty=1.1,
                max_new_tokens=min(512, max(128, len(text) * 5)),
            )
            wav = wavs[0]
            sf.write(str(out_wav), wav, sr)

            wav24 = wav.astype(np.float32)
            if int(sr) != 24000:
                wav24 = librosa.resample(wav24, orig_sr=int(sr), target_sr=24000)

            gen_emb = model.model.extract_speaker_embedding(wav24, 24000)
            if isinstance(gen_emb, torch.Tensor):
                gen_emb = gen_emb.detach().float().cpu().numpy()
            gen_emb = _unit(np.asarray(gen_emb, dtype=np.float32))
            spk_cos = _cos(gen_emb, ref_emb)

            segs, _ = whisper.transcribe(
                str(out_wav),
                language="ko",
                beam_size=3,
                vad_filter=True,
                condition_on_previous_text=False,
            )
            asr_text = "".join(s.text for s in segs).strip()
            asr_sim = _levenshtein_ratio(
                _normalize_text(text), _normalize_text(asr_text)
            )
            dur = float(len(wav24) / 24000.0)

            score = (
                0.6 * spk_cos
                + 0.3 * asr_sim
                + 0.1
                * min(1.0, float(math.sqrt(np.mean(np.square(wav24)) + 1e-12)) / 0.03)
            )
            results.append(
                {
                    "text_idx": ti + 1,
                    "spk": spk_cos,
                    "asr": asr_sim,
                    "score": score,
                    "asr_text": asr_text,
                    "dur": dur,
                    "error": None,
                }
            )
            print(
                f"  text{ti + 1}: spk={spk_cos:.3f} asr={asr_sim:.3f} score={score:.3f} dur={dur:.1f}s"
            )
            print(f"    target: {text[:60]}")
            print(f"    heard:  {asr_text[:60]}")
        except Exception as e:
            results.append(
                {"text_idx": ti + 1, "spk": 0, "asr": 0, "score": -1, "error": str(e)}
            )
            print(f"  text{ti + 1}: ERROR: {e}")

    valid = [r for r in results if r["error"] is None]
    if valid:
        avg_spk = np.mean([r["spk"] for r in valid])
        avg_asr = np.mean([r["asr"] for r in valid])
        avg_score = np.mean([r["score"] for r in valid])
        print(f"\n{'=' * 60}")
        print(
            f"ICL Voice Clone: spk={avg_spk:.4f} asr={avg_asr:.4f} score={avg_score:.4f}"
        )
        print(f"{'=' * 60}")

    report = {
        "mode": "icl_voice_clone",
        "ref_audio": str(ref_path),
        "ref_text": REF_TEXT,
        "results": results,
    }
    (OUTPUT_DIR / "icl_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    del model, whisper
    gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
