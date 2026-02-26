#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import itertools
import json
import math
import os
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
            curr[j] = min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + cost,
            )
        prev = curr
    dist = prev[n]
    return 1.0 - (dist / max(len(a), len(b), 1))


def _asr_similarity(target: str, pred: str) -> float:
    return float(_levenshtein_ratio(_normalize_text(target), _normalize_text(pred)))


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


def _load_wav_24k(path: Path) -> np.ndarray:
    wav, _ = librosa.load(str(path), sr=24000, mono=True)
    return wav.astype(np.float32)


def _rms(wav: np.ndarray) -> float:
    if wav.size == 0:
        return 0.0
    return float(math.sqrt(float(np.mean(np.square(wav), dtype=np.float64)) + 1e-12))


def _clip_ratio(wav: np.ndarray, threshold: float = 0.99) -> float:
    if wav.size == 0:
        return 1.0
    return float(np.mean(np.abs(wav) >= threshold))


@dataclass
class EvalRow:
    idx: int
    name: str
    wav: str
    params: dict[str, Any]
    speaker_cos_ref: float
    speaker_cos_centroid: float
    speaker_score: float
    asr_sim: float
    score: float
    asr_text: str
    error: str | None = None


def _choose_device() -> tuple[str, torch.dtype]:
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    if torch.cuda.is_available():
        return "cuda:0", torch.bfloat16
    return "cpu", torch.float32


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _extract_emb_from_wav(base_model: Qwen3TTSModel, wav_24k: np.ndarray) -> np.ndarray:
    emb = base_model.model.extract_speaker_embedding(wav_24k, 24000)
    if isinstance(emb, torch.Tensor):
        emb = emb.detach().float().cpu().numpy()
    return _unit(np.asarray(emb, dtype=np.float32))


def _transcribe_ko(whisper: WhisperModel, wav_path: Path) -> str:
    segments, _ = whisper.transcribe(
        str(wav_path),
        language="ko",
        beam_size=3,
        vad_filter=True,
        condition_on_previous_text=False,
    )
    return "".join(seg.text for seg in segments).strip()


def _score(speaker_score: float, asr_sim: float) -> float:
    val = 0.88 * speaker_score + 0.12 * asr_sim
    if asr_sim < 0.98:
        val -= (0.98 - asr_sim) * 2.0
    return float(val)


def _build_embedding_candidates(
    seg_paths: list[Path],
    seg_embs: list[np.ndarray],
    seg_joint_scores: list[float],
    ref_emb: np.ndarray,
    max_individual: int = 8,
) -> dict[str, np.ndarray]:
    arr = np.stack(seg_embs, axis=0)
    centroid_all = _unit(arr.mean(axis=0))

    idx_joint = np.argsort(np.asarray(seg_joint_scores))[::-1]
    idx_ref = np.argsort(np.asarray([_cos(e, ref_emb) for e in seg_embs]))[::-1]
    idx_cent = np.argsort(np.asarray([_cos(e, centroid_all) for e in seg_embs]))[::-1]

    top_joint_20 = idx_joint[: min(20, len(seg_embs))]
    top_ref_10 = idx_ref[: min(10, len(seg_embs))]
    top_cent_10 = idx_cent[: min(10, len(seg_embs))]

    cent_joint_20 = _unit(arr[top_joint_20].mean(axis=0))
    cent_ref_10 = _unit(arr[top_ref_10].mean(axis=0))
    cent_cent_10 = _unit(arr[top_cent_10].mean(axis=0))

    w = np.asarray([max(seg_joint_scores[i], 0.0) for i in top_joint_20], dtype=np.float32)
    if float(np.sum(w)) <= 1e-8:
        w = np.ones_like(w) / max(len(w), 1)
    else:
        w = w / float(np.sum(w))
    weighted_joint_20 = _unit(np.sum(arr[top_joint_20] * w[:, None], axis=0))

    cand: dict[str, np.ndarray] = {}
    cand["ref8"] = _unit(ref_emb)
    cand["centroid_all"] = centroid_all
    cand["centroid_top20_joint"] = cent_joint_20
    cand["centroid_top10_ref"] = cent_ref_10
    cand["centroid_top10_centroid"] = cent_cent_10
    cand["weighted_top20_joint"] = weighted_joint_20
    cand["blend_ref90_joint10"] = _unit(0.90 * ref_emb + 0.10 * cent_joint_20)
    cand["blend_ref80_joint20"] = _unit(0.80 * ref_emb + 0.20 * cent_joint_20)
    cand["blend_ref70_joint30"] = _unit(0.70 * ref_emb + 0.30 * cent_joint_20)

    for rank, i in enumerate(idx_joint[: max_individual], start=1):
        stem = seg_paths[int(i)].stem
        cand[f"seg_joint_rank{rank:02d}_{stem}"] = seg_embs[int(i)]

    return cand


def _evaluate(
    *,
    idx: int,
    name: str,
    output_wav: Path,
    custom_model: Qwen3TTSModel,
    base_model: Qwen3TTSModel,
    whisper: WhisperModel,
    speaker_id: int,
    speaker_name: str,
    language: str,
    target_text: str,
    params: dict[str, Any],
    embedding_vec: np.ndarray,
    ref_emb: np.ndarray,
    centroid_emb: np.ndarray,
    seed: int,
    embedding_scale: float,
) -> EvalRow:
    _seed_everything(seed)
    weight = custom_model.model.talker.model.codec_embedding.weight
    with torch.no_grad():
        row = torch.from_numpy(_unit(embedding_vec) * float(embedding_scale)).to(weight.device).to(weight.dtype)
        weight[int(speaker_id)].copy_(row)

    try:
        wavs, sr = custom_model.generate_custom_voice(
            text=target_text,
            speaker=speaker_name,
            language=language,
            **params,
        )
        wav = wavs[0]
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_wav), wav, sr)

        wav24 = wav.astype(np.float32)
        if int(sr) != 24000:
            wav24 = librosa.resample(wav24, orig_sr=int(sr), target_sr=24000)
        gen_emb = _extract_emb_from_wav(base_model, wav24)
        cos_ref = _cos(gen_emb, ref_emb)
        cos_cent = _cos(gen_emb, centroid_emb)
        speaker_score = 0.7 * cos_ref + 0.3 * cos_cent
        asr_text = _transcribe_ko(whisper, output_wav)
        asr_sim = _asr_similarity(target_text, asr_text)
        score = _score(speaker_score, asr_sim)
        return EvalRow(
            idx=idx,
            name=name,
            wav=str(output_wav),
            params=params,
            speaker_cos_ref=cos_ref,
            speaker_cos_centroid=cos_cent,
            speaker_score=speaker_score,
            asr_sim=asr_sim,
            score=score,
            asr_text=asr_text,
            error=None,
        )
    except Exception as e:
        return EvalRow(
            idx=idx,
            name=name,
            wav=str(output_wav),
            params=params,
            speaker_cos_ref=0.0,
            speaker_cos_centroid=0.0,
            speaker_score=0.0,
            asr_sim=0.0,
            score=-1.0,
            asr_text="",
            error=str(e),
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Boost Korean speaker similarity with embedding + param search.")
    parser.add_argument(
        "--source-checkpoint",
        default="workspace/runs/seojaehyung_speaker_only_ref8_20260223/checkpoint-epoch-0",
    )
    parser.add_argument("--base-speaker-model", default=_find_default_base_model())
    parser.add_argument("--segments-dir", default="workspace/imports/seojaehyung_0821/segments_12s")
    parser.add_argument("--reference-audio", default="workspace/imports/seojaehyung_0821/ref_8s.wav")
    parser.add_argument("--speaker-name", default="seojaehyung")
    parser.add_argument("--speaker-id", type=int, default=3000)
    parser.add_argument("--language", default="korean")
    parser.add_argument(
        "--target-text",
        default="안녕하세요. 오늘 시장 이야기를 간단하게 말씀드리겠습니다.",
    )
    parser.add_argument("--output-dir", default="workspace/exports/korean_similarity_push")
    parser.add_argument("--whisper-model", default="base")
    parser.add_argument("--seed", type=int, default=20260223)
    parser.add_argument("--max-individual-embeddings", type=int, default=8)
    args = parser.parse_args()

    source_ckpt = Path(args.source_checkpoint).expanduser().resolve()
    base_model_path = str(Path(args.base_speaker_model).expanduser())
    segments_dir = Path(args.segments_dir).expanduser().resolve()
    ref_path = Path(args.reference_audio).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    emb_dir = output_dir / "embedding_stage"
    param_dir = output_dir / "param_stage"
    emb_dir.mkdir(parents=True, exist_ok=True)
    param_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] source checkpoint: {source_ckpt}")
    print(f"[INFO] base speaker model: {base_model_path}")
    print(f"[INFO] segments dir: {segments_dir}")
    print(f"[INFO] output dir: {output_dir}")

    seg_paths = sorted(segments_dir.glob("*.wav"))
    if not seg_paths:
        raise FileNotFoundError(f"No wav files found in: {segments_dir}")
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference audio not found: {ref_path}")

    device, dtype = _choose_device()
    print(f"[INFO] device={device} dtype={dtype}")

    _seed_everything(args.seed)
    base_model = Qwen3TTSModel.from_pretrained(base_model_path, device_map=device, dtype=dtype)
    custom_model = Qwen3TTSModel.from_pretrained(str(source_ckpt), device_map=device, dtype=dtype)
    whisper = WhisperModel(args.whisper_model, device="cpu", compute_type="int8")

    with torch.no_grad():
        embedding_scale = float(
            torch.norm(custom_model.model.talker.model.codec_embedding.weight[int(args.speaker_id)])
            .detach()
            .float()
            .cpu()
            .item()
        )
    print(f"[INFO] speaker embedding scale (id={args.speaker_id}): {embedding_scale:.6f}")

    ref_wav = _load_wav_24k(ref_path)
    ref_emb = _extract_emb_from_wav(base_model, ref_wav)

    seg_embs: list[np.ndarray] = []
    seg_meta: list[dict[str, Any]] = []
    for p in seg_paths:
        wav = _load_wav_24k(p)
        emb = _extract_emb_from_wav(base_model, wav)
        seg_embs.append(emb)
        seg_meta.append(
            {
                "wav": str(p),
                "rms": _rms(wav),
                "clip_ratio": _clip_ratio(wav),
                "duration_sec": float(len(wav) / 24000.0),
            }
        )

    centroid_all = _unit(np.mean(np.stack(seg_embs, axis=0), axis=0))
    seg_joint_scores: list[float] = []
    for i, emb in enumerate(seg_embs):
        cos_ref = _cos(emb, ref_emb)
        cos_cent = _cos(emb, centroid_all)
        clip_penalty = min(1.0, seg_meta[i]["clip_ratio"] * 50.0)
        loud_bonus = min(1.0, seg_meta[i]["rms"] / 0.05)
        joint = 0.55 * cos_ref + 0.45 * cos_cent + 0.02 * loud_bonus - 0.05 * clip_penalty
        seg_meta[i]["cos_ref"] = cos_ref
        seg_meta[i]["cos_centroid"] = cos_cent
        seg_meta[i]["joint_score"] = joint
        seg_joint_scores.append(joint)

    emb_candidates = _build_embedding_candidates(
        seg_paths=seg_paths,
        seg_embs=seg_embs,
        seg_joint_scores=seg_joint_scores,
        ref_emb=ref_emb,
        max_individual=max(1, int(args.max_individual_embeddings)),
    )

    base_params = {
        "temperature": 0.45,
        "top_k": 30,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
        "max_new_tokens": 220,
        "subtalker_temperature": 0.8,
        "subtalker_top_k": 30,
        "subtalker_top_p": 0.9,
    }

    emb_rows: list[EvalRow] = []
    print(f"[INFO] embedding candidates: {len(emb_candidates)}")
    for i, (name, vec) in enumerate(emb_candidates.items(), start=1):
        out_wav = emb_dir / f"emb_{i:02d}_{name}.wav"
        row = _evaluate(
            idx=i,
            name=name,
            output_wav=out_wav,
            custom_model=custom_model,
            base_model=base_model,
            whisper=whisper,
            speaker_id=args.speaker_id,
            speaker_name=args.speaker_name,
            language=args.language,
            target_text=args.target_text,
            params=base_params,
            embedding_vec=vec,
            ref_emb=ref_emb,
            centroid_emb=centroid_all,
            seed=args.seed,
            embedding_scale=embedding_scale,
        )
        emb_rows.append(row)
        print(
            f"[EMB {i:02d}/{len(emb_candidates):02d}] {name} "
            f"speaker={row.speaker_score:.4f} asr={row.asr_sim:.4f} score={row.score:.4f}"
        )

    emb_rows_sorted = sorted(emb_rows, key=lambda x: x.score, reverse=True)
    best_emb_row = emb_rows_sorted[0]
    best_emb_vec = emb_candidates[best_emb_row.name]
    print(
        f"[INFO] best embedding: {best_emb_row.name} "
        f"speaker={best_emb_row.speaker_score:.4f} asr={best_emb_row.asr_sim:.4f} score={best_emb_row.score:.4f}"
    )

    param_grid = list(
        itertools.product(
            [0.40, 0.45, 0.50],   # temperature
            [0.88, 0.90],         # top_p
            [1.08, 1.10],         # repetition_penalty
            [200, 210, 220],      # max_new_tokens
            [0.75, 0.80],         # subtalker_temperature
        )
    )
    print(f"[INFO] param candidates: {len(param_grid)}")

    param_rows: list[EvalRow] = []
    for i, (temp, top_p, rep, max_new, sub_t) in enumerate(param_grid, start=1):
        params = {
            "temperature": float(temp),
            "top_k": 30,
            "top_p": float(top_p),
            "repetition_penalty": float(rep),
            "max_new_tokens": int(max_new),
            "subtalker_temperature": float(sub_t),
            "subtalker_top_k": 30,
            "subtalker_top_p": float(top_p),
        }
        out_wav = param_dir / f"cand_{i:03d}.wav"
        row = _evaluate(
            idx=i,
            name=f"cand_{i:03d}",
            output_wav=out_wav,
            custom_model=custom_model,
            base_model=base_model,
            whisper=whisper,
            speaker_id=args.speaker_id,
            speaker_name=args.speaker_name,
            language=args.language,
            target_text=args.target_text,
            params=params,
            embedding_vec=best_emb_vec,
            ref_emb=ref_emb,
            centroid_emb=centroid_all,
            seed=args.seed,
            embedding_scale=embedding_scale,
        )
        param_rows.append(row)
        print(
            f"[PAR {i:03d}/{len(param_grid):03d}] "
            f"speaker={row.speaker_score:.4f} asr={row.asr_sim:.4f} score={row.score:.4f}"
        )

    param_rows_sorted = sorted(param_rows, key=lambda x: x.score, reverse=True)
    best_param_row = param_rows_sorted[0]
    best_wav = Path(best_param_row.wav)
    best_out = output_dir / "best.wav"
    sf.write(str(best_out), _load_wav_24k(best_wav), 24000)

    report = {
        "source_checkpoint": str(source_ckpt),
        "base_speaker_model": base_model_path,
        "segments_dir": str(segments_dir),
        "reference_audio": str(ref_path),
        "target_text": args.target_text,
        "language": args.language,
        "embedding_best": asdict(best_emb_row),
        "final_best": asdict(best_param_row),
        "embedding_rows": [asdict(r) for r in emb_rows_sorted],
        "param_rows": [asdict(r) for r in param_rows_sorted],
        "best_embedding_name": best_emb_row.name,
        "best_embedding_wav": best_emb_row.wav,
        "best_wav": str(best_out),
    }
    report_path = output_dir / "report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    seg_report_path = output_dir / "segment_quality.json"
    with seg_report_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "rows": sorted(seg_meta, key=lambda x: x["joint_score"], reverse=True),
                "centroid_embedding_note": "Computed from normalized speaker embeddings.",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    summary_lines = [
        f"BEST_EMBEDDING={best_emb_row.name}",
        f"BEST_EMB_SCORE={best_emb_row.score:.4f}",
        f"BEST_FINAL_IDX={best_param_row.idx}",
        f"BEST_FINAL_SPK_SCORE={best_param_row.speaker_score:.4f}",
        f"BEST_FINAL_COS_REF={best_param_row.speaker_cos_ref:.4f}",
        f"BEST_FINAL_COS_CENTROID={best_param_row.speaker_cos_centroid:.4f}",
        f"BEST_FINAL_ASR={best_param_row.asr_sim:.4f}",
        f"BEST_FINAL_SCORE={best_param_row.score:.4f}",
        f"BEST_FINAL_WAV={best_out}",
        f"BEST_FINAL_ASR_TEXT={best_param_row.asr_text}",
        f"BEST_FINAL_PARAMS={json.dumps(best_param_row.params, ensure_ascii=False)}",
        f"REPORT={report_path}",
    ]
    summary_path = output_dir / "summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print("\n".join(summary_lines))
    print(f"[INFO] report: {report_path}")
    print(f"[INFO] segment report: {seg_report_path}")

    del whisper
    del custom_model
    del base_model
    gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
