#!/usr/bin/env python3
"""Test ICL voice clone with sentence-boundary ref audio to fix ref bleed."""

import json
import time
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Ref candidates (all end on clean sentence boundary) ──────────────
REFS = {
    "seg_0038": {
        "audio": ROOT / "workspace/imports/quality_test_quick/segments/seg_0038.wav",
        "text": "알래스카에서 회담이 열린다고 합니다.",
        "note": "formal -합니다 ending, 3.1s",
    },
    "seg_0026": {
        "audio": ROOT / "workspace/imports/quality_test_quick/segments/seg_0026.wav",
        "text": "새로운 역사의 전환점의 입구에 있는 것 같아요.",
        "note": "casual -같아요 ending, 3.7s",
    },
    "seg_0068": {
        "audio": ROOT / "workspace/imports/quality_test_quick/segments/seg_0068.wav",
        "text": "이 기업들은 주가들이 거의 다 올라 있어요.",
        "note": "casual -있어요 ending, 3.7s",
    },
}

# ── Test texts (including the problematic ones) ──────────────────────
TEXTS = [
    "안녕하세요. 오늘 시장 이야기를 간단하게 말씀드리겠습니다.",
    "자, 그러면 오늘의 주요 종목들을 하나씩 살펴볼까요?",  # was bleeding before
    "투자에서 가장 중요한 것은 리스크 관리입니다. 절대 잊지 마세요.",  # had prefix leak
]

OUT_DIR = ROOT / "workspace/imports/better_ref_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_base_model():
    from qwen_tts import Qwen3TTSModel
    snapshots = (
        Path.home() / ".cache/huggingface/hub"
        / "models--Qwen--Qwen3-TTS-12Hz-0.6B-Base/snapshots"
    )
    base = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
    if snapshots.exists():
        cands = sorted(snapshots.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        if cands:
            base = str(cands[0])
    print(f"Loading base model from: {base}")
    model = Qwen3TTSModel.from_pretrained(base, device_map="cpu")
    model.model.eval()
    return model


def compute_cosine(model, wav_path, ref_path):
    """Compute speaker cosine via x-vector."""
    try:
        emb_gen = model.extract_x_vector(str(wav_path))
        emb_ref = model.extract_x_vector(str(ref_path))
        cos = torch.nn.functional.cosine_similarity(
            torch.tensor(emb_gen).unsqueeze(0),
            torch.tensor(emb_ref).unsqueeze(0),
        ).item()
        return cos
    except Exception as e:
        return f"error: {e}"


def transcribe_whisper(wav_path):
    try:
        import whisper
        wmodel = whisper.load_model("base")
        result = wmodel.transcribe(str(wav_path), language="ko")
        return result.get("text", "").strip()
    except Exception as e:
        return f"error: {e}"


def main():
    model = load_base_model()
    results = []
    total = len(REFS) * len(TEXTS)
    idx = 0

    for ref_name, ref_info in REFS.items():
        ref_audio = str(ref_info["audio"])
        ref_text = ref_info["text"]
        print(f"\n{'='*60}")
        print(f"REF: {ref_name} — {ref_info['note']}")
        print(f"REF TEXT: {ref_text}")
        print(f"{'='*60}")

        for ti, text in enumerate(TEXTS, 1):
            idx += 1
            print(f"\n[{idx}/{total}] Generating: {text[:50]}...")
            t0 = time.time()
            try:
                wavs, sr = model.generate_voice_clone(
                    text=text,
                    language=None,
                    ref_audio=ref_audio,
                    ref_text=ref_text,
                    x_vector_only_mode=False,
                    do_sample=True,
                    subtalker_dosample=True,
                    temperature=0.9,
                    top_k=50,
                    top_p=1.0,
                    repetition_penalty=1.05,
                    subtalker_temperature=0.9,
                    subtalker_top_k=50,
                    subtalker_top_p=1.0,
                    max_new_tokens=2048,
                )
                audio = wavs[0]
                gen_time = time.time() - t0
                fname = f"{ref_name}_text{ti:02d}.wav"
                out_path = OUT_DIR / fname
                sf.write(out_path, audio, sr)
                dur = len(audio) / sr

                cos = compute_cosine(model, out_path, ref_audio)
                print(f"  Duration: {dur:.2f}s | Gen time: {gen_time:.1f}s | Cosine: {cos}")

                results.append({
                    "ref": ref_name,
                    "ref_text": ref_text,
                    "text_idx": ti,
                    "text": text,
                    "wav": str(out_path),
                    "duration_sec": round(dur, 2),
                    "gen_time_sec": round(gen_time, 1),
                    "cos": round(cos, 4) if isinstance(cos, float) else cos,
                    "error": None,
                })
            except Exception as e:
                gen_time = time.time() - t0
                print(f"  FAILED: {e}")
                results.append({
                    "ref": ref_name,
                    "text_idx": ti,
                    "text": text,
                    "wav": "",
                    "duration_sec": 0,
                    "gen_time_sec": round(gen_time, 1),
                    "cos": 0,
                    "error": str(e),
                })

    # Run ASR on all successful wavs
    print(f"\n{'='*60}")
    print("Running Whisper ASR on all generated files...")
    print(f"{'='*60}")
    for r in results:
        if r["error"] is None and r["wav"]:
            asr = transcribe_whisper(r["wav"])
            r["asr_text"] = asr
            print(f"  {Path(r['wav']).name}: {asr}")
        else:
            r["asr_text"] = ""

    report_path = OUT_DIR / "better_ref_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"REPORT saved: {report_path}")
    print(f"{'='*60}")
    print(f"\nSummary:")
    for r in results:
        status = "OK" if r["error"] is None else "FAIL"
        bleed = ""
        if r.get("asr_text") and r["text"]:
            target_start = r["text"][:6]
            if r["asr_text"] and not r["asr_text"].startswith(target_start[:3]):
                bleed = " ⚠️ POSSIBLE BLEED"
        print(f"  [{status}] {r['ref']}/text{r['text_idx']:02d} cos={r['cos']} dur={r['duration_sec']}s{bleed}")
        if r.get("asr_text"):
            print(f"           ASR: {r['asr_text'][:60]}")


if __name__ == "__main__":
    main()