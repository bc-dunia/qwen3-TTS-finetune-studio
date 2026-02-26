#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import itertools
import json
import math
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf
import torch
from faster_whisper import WhisperModel
from qwen_tts import Qwen3TTSModel


def _norm_text(s: str) -> str:
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
    d = prev[n]
    return 1.0 - (d / max(len(a), len(b), 1))


def _asr_similarity(target: str, pred: str) -> float:
    return float(_levenshtein_ratio(_norm_text(target), _norm_text(pred)))


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb + 1e-12))


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= 1e-12:
        return v.copy()
    return v / n


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _load_wav_24k(path: str | Path) -> np.ndarray:
    wav, _ = librosa.load(str(path), sr=24000, mono=True)
    return wav.astype(np.float32)


def _wav_duration_sec(wav: np.ndarray, sr: int = 24000) -> float:
    return float(len(wav) / float(sr))


def _f0_median_std(wav: np.ndarray, sr: int = 24000) -> tuple[float, float]:
    try:
        f0, _, _ = librosa.pyin(
            wav.astype(np.float32),
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C6"),
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


def _style_score(
    *,
    duration: float,
    target_duration: float,
    f0_std: float,
    target_f0_std: float,
) -> tuple[float, float, float]:
    speed_err = abs(duration - target_duration) / max(target_duration, 1e-6)
    speed_score = max(0.0, 1.0 - 1.8 * speed_err)
    pitch_rel_err = abs(math.log((f0_std + 1e-3) / max(target_f0_std, 1e-3)))
    pitch_score = max(0.0, 1.0 - 2.0 * pitch_rel_err)
    style = 0.8 * speed_score + 0.2 * pitch_score
    return float(style), float(speed_score), float(pitch_score)


def _final_score(*, speaker_cos: float, asr_sim: float, style_score: float) -> float:
    score = 0.58 * speaker_cos + 0.17 * asr_sim + 0.25 * style_score
    if asr_sim < 0.98:
        score -= (0.98 - asr_sim) * 2.0
    if speaker_cos < 0.984:
        score -= (0.984 - speaker_cos) * 1.2
    return float(score)


def _transcribe_ko(whisper: WhisperModel, wav_path: Path) -> str:
    segments, _ = whisper.transcribe(
        str(wav_path),
        language="ko",
        beam_size=3,
        vad_filter=True,
        condition_on_previous_text=False,
    )
    return "".join(seg.text for seg in segments).strip()


def _extract_emb(model_base: Qwen3TTSModel, wav_24k: np.ndarray) -> np.ndarray:
    emb = model_base.model.extract_speaker_embedding(wav_24k, 24000)
    if isinstance(emb, torch.Tensor):
        emb = emb.detach().float().cpu().numpy()
    return np.asarray(emb, dtype=np.float32)


@dataclass
class Row:
    stage: str
    idx: int
    seed: int
    text: str
    language: str
    params: dict[str, Any]
    wav: str
    speaker_cos_ref: float
    asr_sim: float
    style_score: float
    speed_score: float
    pitch_score: float
    duration_sec: float
    duration_ratio: float
    f0_median: float
    f0_std: float
    final_score: float
    asr_text: str
    error: str | None = None


def _choose_device() -> tuple[str, torch.dtype]:
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    if torch.cuda.is_available():
        return "cuda:0", torch.bfloat16
    return "cpu", torch.float32


def _default_base_model() -> str:
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


def _target_duration_from_anchor(anchor_audio: Path, anchor_text: str, target_text: str) -> float:
    wav = _load_wav_24k(anchor_audio)
    d = _wav_duration_sec(wav, 24000)
    a_chars = len(_norm_text(anchor_text))
    t_chars = len(_norm_text(target_text))
    cps = float(a_chars / max(d, 1e-6))
    return float(t_chars / max(cps, 1e-6))


def _read_anchor_text(transcript_jsonl: Path, seg_stem: str) -> str:
    for line in transcript_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if Path(str(row.get("audio", ""))).stem == seg_stem:
            return str(row.get("text", ""))
    return ""


def _eval_one(
    *,
    idx: int,
    stage: str,
    seed: int,
    text: str,
    language: str,
    params: dict[str, Any],
    out_wav: Path,
    speaker_name: str,
    custom_model: Qwen3TTSModel,
    base_model: Qwen3TTSModel,
    whisper: WhisperModel,
    ref_emb: np.ndarray,
    target_text_for_asr: str,
    target_duration: float,
    target_f0_std: float,
) -> Row:
    _seed_all(seed)
    try:
        wavs, sr = custom_model.generate_custom_voice(
            text=text,
            speaker=speaker_name,
            language=language,
            **params,
        )
        wav = np.asarray(wavs[0], dtype=np.float32)
        out_wav.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_wav), wav, int(sr))

        wav24 = wav if int(sr) == 24000 else librosa.resample(wav, orig_sr=int(sr), target_sr=24000)
        emb = _extract_emb(base_model, wav24)
        cos_ref = _cos(_unit(emb), _unit(ref_emb))
        asr_text = _transcribe_ko(whisper, out_wav)
        asr_sim = _asr_similarity(target_text_for_asr, asr_text)
        dur = _wav_duration_sec(wav24, 24000)
        f0_med, f0_std = _f0_median_std(wav24, 24000)
        style, speed_s, pitch_s = _style_score(
            duration=dur,
            target_duration=target_duration,
            f0_std=f0_std,
            target_f0_std=target_f0_std,
        )
        final = _final_score(speaker_cos=cos_ref, asr_sim=asr_sim, style_score=style)
        return Row(
            stage=stage,
            idx=idx,
            seed=seed,
            text=text,
            language=language,
            params=params,
            wav=str(out_wav),
            speaker_cos_ref=cos_ref,
            asr_sim=asr_sim,
            style_score=style,
            speed_score=speed_s,
            pitch_score=pitch_s,
            duration_sec=dur,
            duration_ratio=dur / max(target_duration, 1e-6),
            f0_median=f0_med,
            f0_std=f0_std,
            final_score=final,
            asr_text=asr_text,
            error=None,
        )
    except Exception as e:
        return Row(
            stage=stage,
            idx=idx,
            seed=seed,
            text=text,
            language=language,
            params=params,
            wav=str(out_wav),
            speaker_cos_ref=0.0,
            asr_sim=0.0,
            style_score=0.0,
            speed_score=0.0,
            pitch_score=0.0,
            duration_sec=0.0,
            duration_ratio=0.0,
            f0_median=0.0,
            f0_std=0.0,
            final_score=-1.0,
            asr_text="",
            error=str(e),
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Style/speed boost for Korean custom voice.")
    parser.add_argument(
        "--checkpoint",
        default="workspace/runs/seojaehyung_speaker_only_ref8_20260223/checkpoint-epoch-0",
    )
    parser.add_argument("--base-speaker-model", default=_default_base_model())
    parser.add_argument("--speaker-name", default="seojaehyung")
    parser.add_argument("--reference-audio", default="workspace/imports/seojaehyung_0821/ref_8s.wav")
    parser.add_argument("--anchor-segment", default="workspace/imports/seojaehyung_0821/segments_12s/seg_0032.wav")
    parser.add_argument("--anchor-transcript-jsonl", default="workspace/imports/seojaehyung_0821/transcript_whisper_base.jsonl")
    parser.add_argument(
        "--target-text",
        default="안녕하세요. 오늘 시장 이야기를 간단하게 말씀드리겠습니다.",
    )
    parser.add_argument("--output-dir", default="workspace/exports/style_speed_boost")
    parser.add_argument("--seed", type=int, default=20260223)
    parser.add_argument("--whisper-model", default="base")
    args = parser.parse_args()

    checkpoint = str(Path(args.checkpoint).expanduser().resolve())
    base_model_path = str(Path(args.base_speaker_model).expanduser())
    ref_audio = Path(args.reference_audio).expanduser().resolve()
    anchor_audio = Path(args.anchor_segment).expanduser().resolve()
    anchor_transcript = Path(args.anchor_transcript_jsonl).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    s1_dir = output_dir / "stage1"
    s2_dir = output_dir / "stage2"
    output_dir.mkdir(parents=True, exist_ok=True)
    s1_dir.mkdir(parents=True, exist_ok=True)
    s2_dir.mkdir(parents=True, exist_ok=True)

    anchor_stem = anchor_audio.stem
    anchor_text = _read_anchor_text(anchor_transcript, anchor_stem)
    if not anchor_text:
        raise RuntimeError(f"Anchor text not found for {anchor_stem} in {anchor_transcript}")

    anchor_wav = _load_wav_24k(anchor_audio)
    _, anchor_f0_std = _f0_median_std(anchor_wav)
    target_duration = _target_duration_from_anchor(anchor_audio, anchor_text, args.target_text)

    device, dtype = _choose_device()
    print(f"[INFO] device={device} dtype={dtype}")
    print(f"[INFO] checkpoint={checkpoint}")
    print(f"[INFO] base_model={base_model_path}")
    print(f"[INFO] target_duration={target_duration:.4f}s anchor_f0_std={anchor_f0_std:.4f}")

    base_model = Qwen3TTSModel.from_pretrained(base_model_path, device_map=device, dtype=dtype)
    custom_model = Qwen3TTSModel.from_pretrained(checkpoint, device_map=device, dtype=dtype)
    whisper = WhisperModel(args.whisper_model, device="cpu", compute_type="int8")
    ref_emb = _extract_emb(base_model, _load_wav_24k(ref_audio))

    stage1_texts = [
        args.target_text,
        "안녕하세요, 오늘 시장 이야기를 간단하게 말씀드리겠습니다.",
        "안녕하세요 오늘 시장 이야기를 간단하게 말씀드리겠습니다.",
    ]
    stage1_grid = list(
        itertools.product(
            stage1_texts,
            ["auto", "korean"],
            [176, 188, 200, 212, 220],
            [1.08, 1.10],
        )
    )
    print(f"[INFO] stage1 candidates={len(stage1_grid)}")

    stage1_rows: list[Row] = []
    for i, (txt, lang, max_new, rep) in enumerate(stage1_grid, start=1):
        params = {
            "temperature": 0.45,
            "top_k": 30,
            "top_p": 0.9,
            "repetition_penalty": float(rep),
            "max_new_tokens": int(max_new),
            "subtalker_temperature": 0.8,
            "subtalker_top_k": 30,
            "subtalker_top_p": 0.9,
        }
        row = _eval_one(
            idx=i,
            stage="stage1",
            seed=int(args.seed),
            text=txt,
            language=lang,
            params=params,
            out_wav=s1_dir / f"cand_{i:03d}.wav",
            speaker_name=args.speaker_name,
            custom_model=custom_model,
            base_model=base_model,
            whisper=whisper,
            ref_emb=ref_emb,
            target_text_for_asr=args.target_text,
            target_duration=target_duration,
            target_f0_std=anchor_f0_std,
        )
        stage1_rows.append(row)
        print(
            f"[S1 {i:03d}/{len(stage1_grid):03d}] "
            f"cos={row.speaker_cos_ref:.4f} asr={row.asr_sim:.4f} style={row.style_score:.4f} "
            f"dur_ratio={row.duration_ratio:.3f} final={row.final_score:.4f}"
        )

    stage1_sorted = sorted(stage1_rows, key=lambda x: x.final_score, reverse=True)
    top_for_stage2 = [r for r in stage1_sorted if r.speaker_cos_ref >= 0.983 and r.asr_sim >= 0.98][:4]
    if not top_for_stage2:
        top_for_stage2 = stage1_sorted[:4]
    print(f"[INFO] stage2 seeds per candidate=12, candidates={len(top_for_stage2)}")

    stage2_rows: list[Row] = []
    seeds = [args.seed + i for i in range(1, 13)]
    idx = 0
    for base_idx, base_row in enumerate(top_for_stage2, start=1):
        for seed in seeds:
            idx += 1
            row = _eval_one(
                idx=idx,
                stage=f"stage2_top{base_idx}",
                seed=int(seed),
                text=base_row.text,
                language=base_row.language,
                params=base_row.params,
                out_wav=s2_dir / f"cand_{idx:03d}.wav",
                speaker_name=args.speaker_name,
                custom_model=custom_model,
                base_model=base_model,
                whisper=whisper,
                ref_emb=ref_emb,
                target_text_for_asr=args.target_text,
                target_duration=target_duration,
                target_f0_std=anchor_f0_std,
            )
            stage2_rows.append(row)
            print(
                f"[S2 {idx:03d}/{len(top_for_stage2)*len(seeds):03d}] "
                f"cos={row.speaker_cos_ref:.4f} asr={row.asr_sim:.4f} style={row.style_score:.4f} "
                f"dur_ratio={row.duration_ratio:.3f} final={row.final_score:.4f}"
            )

    all_rows = stage1_rows + stage2_rows
    all_sorted = sorted(all_rows, key=lambda x: x.final_score, reverse=True)
    best = all_sorted[0]

    best_wav = Path(best.wav)
    best_out = output_dir / "best.wav"
    bw = _load_wav_24k(best_wav)
    sf.write(str(best_out), bw, 24000)

    report = {
        "target_text": args.target_text,
        "anchor_segment": str(anchor_audio),
        "anchor_text": anchor_text,
        "target_duration_sec": target_duration,
        "anchor_f0_std": anchor_f0_std,
        "best": asdict(best),
        "rows": [asdict(r) for r in all_sorted],
        "best_wav": str(best_out),
        "stage1_top4": [asdict(r) for r in stage1_sorted[:4]],
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_lines = [
        f"BEST_FINAL={best.final_score:.6f}",
        f"BEST_COS_REF={best.speaker_cos_ref:.6f}",
        f"BEST_ASR={best.asr_sim:.6f}",
        f"BEST_STYLE={best.style_score:.6f}",
        f"BEST_SPEED={best.speed_score:.6f}",
        f"BEST_PITCH={best.pitch_score:.6f}",
        f"BEST_DURATION_SEC={best.duration_sec:.6f}",
        f"BEST_DURATION_RATIO={best.duration_ratio:.6f}",
        f"BEST_SEED={best.seed}",
        f"BEST_STAGE={best.stage}",
        f"BEST_LANGUAGE={best.language}",
        f"BEST_TEXT={best.text}",
        f"BEST_PARAMS={json.dumps(best.params, ensure_ascii=False)}",
        f"BEST_ASR_TEXT={best.asr_text}",
        f"BEST_WAV={best_out}",
        f"REPORT={report_path}",
    ]
    summary_path = output_dir / "summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print("\n".join(summary_lines))

    del whisper
    del custom_model
    del base_model
    gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
