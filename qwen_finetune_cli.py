#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def cmd_validate(args: argparse.Namespace) -> int:
    from finetune_studio.quality import (
        format_quality_report,
        save_quality_report,
        validate_dataset,
    )

    report = validate_dataset(args.raw_jsonl)
    print(format_quality_report(report))
    if args.output_report:
        out = save_quality_report(report, args.output_report)
        print(f"\nSaved report: {out}")
    return 0 if report.get("summary", {}).get("ok") else 2


def _run_preflight_check(args: argparse.Namespace, *, label: str) -> int:
    from finetune_studio.quality import (
        format_preflight_report,
        run_preflight_review,
        save_preflight_report,
    )

    report = run_preflight_review(
        raw_jsonl_path=args.raw_jsonl,
        init_model_path=args.init_model_path,
        prepare_device=args.prepare_device,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
    )
    print(format_preflight_report(report))
    if args.output_report:
        out = save_preflight_report(report, args.output_report)
        print(f"\nSaved {label} report: {out}")

    decision = str(report.get("decision", "")).lower()
    status = str(report.get("status", "")).lower()
    if decision == "no-go" or status == "blocked":
        return 2
    return 0


def cmd_preflight(args: argparse.Namespace) -> int:
    return _run_preflight_check(args, label="preflight")


def cmd_precheck(args: argparse.Namespace) -> int:
    return _run_preflight_check(args, label="precheck")


def cmd_normalize(args: argparse.Namespace) -> int:
    from finetune_studio.audio_prep import normalize_dataset_audio

    out_dir, out_raw = normalize_dataset_audio(
        raw_jsonl_path=args.raw_jsonl,
        normalized_dataset_name=args.name,
        target_sr=args.target_sr,
        peak_normalize=args.peak_normalize,
    )
    print(f"Normalized dataset dir: {out_dir}")
    print(f"Normalized raw jsonl: {out_raw}")
    return 0


def cmd_prepare(args: argparse.Namespace) -> int:
    from finetune_studio.training_ops import run_prepare_data

    final_event = None
    last_log_line = ""
    last_status = ""
    for event in run_prepare_data(
        device=args.device,
        tokenizer_model_path=args.tokenizer_model_path,
        input_jsonl=args.input_jsonl,
        output_filename=args.output_filename,
        batch_infer_num=args.batch_infer_num,
    ):
        final_event = event
        logs = event.get("logs")
        if logs:
            last_line = logs.splitlines()[-1]
            if last_line != last_log_line:
                print(last_line)
                last_log_line = last_line
        status = event.get("status")
        if status and status != last_status:
            print(status)
            last_status = status
    if not final_event:
        print("Prepare stage produced no events.", file=sys.stderr)
        return 1
    return 0 if final_event.get("success") else 1


def cmd_train(args: argparse.Namespace) -> int:
    from finetune_studio.training_ops import run_training

    final_event = None
    last_line = ""
    for event in run_training(
        init_model_path=args.init_model_path,
        train_jsonl=args.train_jsonl,
        run_name=args.run_name,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        speaker_name=args.speaker_name,
        speaker_id=args.speaker_id,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        subtalker_loss_weight=args.subtalker_loss_weight,
        attn_implementation=args.attn_implementation,
        torch_dtype=args.torch_dtype,
        log_every_n_steps=args.log_every_n_steps,
        save_every_n_epochs=args.save_every_n_epochs,
        max_steps=args.max_steps,
        random_seed=args.seed,
    ):
        final_event = event
        progress = event.get("progress", "")
        status = event.get("status", "")
        line = f"{status} {progress}".strip()
        if line and line != last_line:
            print(line)
            last_line = line
    if not final_event:
        print("Train stage produced no events.", file=sys.stderr)
        return 1
    return 0 if final_event.get("success") else 1


def cmd_pipeline(args: argparse.Namespace) -> int:
    from finetune_studio.pipeline_ops import run_full_pipeline

    final_event = None
    last_line = ""
    for event in run_full_pipeline(
        raw_jsonl_path=args.raw_jsonl,
        prepare_device=args.prepare_device,
        tokenizer_model_path=args.tokenizer_model_path,
        prepare_output_filename=args.prepare_output_filename,
        prepare_batch_infer_num=args.prepare_batch_infer_num,
        init_model_path=args.init_model_path,
        run_name=args.run_name,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        speaker_name=args.speaker_name,
        speaker_id=args.speaker_id,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        subtalker_loss_weight=args.subtalker_loss_weight,
        attn_implementation=args.attn_implementation,
        torch_dtype=args.torch_dtype,
        log_every_n_steps=args.log_every_n_steps,
        save_every_n_epochs=args.save_every_n_epochs,
        max_steps=args.max_steps,
        random_seed=args.seed,
    ):
        final_event = event
        stage = event.get("stage", "")
        status = event.get("status", "")
        progress = event.get("progress", "")
        prefix = f"[{stage}] " if stage else ""
        line = f"{prefix}{status} {progress}".strip()
        if line and line != last_line:
            print(line)
            last_line = line
    if not final_event:
        print("Pipeline produced no events.", file=sys.stderr)
        return 1
    return 0 if final_event.get("success", True) else 1


def cmd_infer(args: argparse.Namespace) -> int:
    from finetune_studio.inference_ops import synthesize_single

    try:
        wav_path, status = synthesize_single(
            checkpoint_path=args.checkpoint,
            device=args.device,
            speaker_name=args.speaker_name,
            text=args.text,
            params={
                "temperature": args.temperature,
                "top_k": args.top_k,
                "top_p": args.top_p,
                "repetition_penalty": args.repetition_penalty,
                "max_new_tokens": args.max_new_tokens,
                "subtalker_temperature": args.subtalker_temperature,
                "subtalker_top_k": args.subtalker_top_k,
                "subtalker_top_p": args.subtalker_top_p,
            },
            language=args.language,
            instruct=args.instruct,
            seed=args.seed,
        )
    except Exception as e:
        print(f"Inference failed: {e}", file=sys.stderr)
        return 1
    print(status)
    final_wav_path = str(wav_path)
    if args.output_path:
        target = Path(args.output_path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(wav_path), target)
        print(f"Copied output: {target}")
        final_wav_path = str(target)

    if bool(args.review_after_generation):
        from finetune_studio.quality import (
            format_generation_review,
            run_generation_review,
            save_generation_review,
        )

        report = run_generation_review(
            generated_audio_path=final_wav_path,
            target_text=args.text,
            reference_audio_path=args.review_reference_audio,
            profile_raw_jsonl=args.review_profile_raw_jsonl,
            base_speaker_model=args.review_base_speaker_model,
            whisper_model=args.review_whisper_model,
        )
        print("")
        print(format_generation_review(report))
        if args.review_output_report:
            out = save_generation_review(report, args.review_output_report)
            print(f"\nSaved generation review report: {out}")

        decision = str(report.get("decision", "")).lower()
        if decision == "fail":
            return 2
        if decision == "warn":
            return 1
    return 0


def cmd_review_generation(args: argparse.Namespace) -> int:
    from finetune_studio.quality import (
        format_generation_review,
        run_generation_review,
        save_generation_review,
    )

    report = run_generation_review(
        generated_audio_path=args.generated_wav,
        target_text=args.target_text,
        reference_audio_path=args.reference_audio,
        profile_raw_jsonl=args.profile_raw_jsonl,
        base_speaker_model=args.base_speaker_model,
        whisper_model=args.whisper_model,
    )
    print(format_generation_review(report))
    if args.output_report:
        out = save_generation_review(report, args.output_report)
        print(f"\nSaved generation review report: {out}")

    decision = str(report.get("decision", "")).lower()
    if decision == "fail":
        return 2
    if decision == "warn":
        return 1
    return 0


def _add_preflight_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--raw-jsonl", required=True)
    parser.add_argument("--init-model-path", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--prepare-device", default="auto")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--output-report", default="")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qwen3-TTS Finetune Studio CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("validate", help="Validate raw dataset JSONL")
    p.add_argument("--raw-jsonl", required=True)
    p.add_argument("--output-report", default="")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("preflight", help="Run pre-training risk review")
    _add_preflight_args(p)
    p.set_defaults(func=cmd_preflight)

    p = sub.add_parser("precheck", help="Run pre-training go/no-go requirements check")
    _add_preflight_args(p)
    p.set_defaults(func=cmd_precheck)

    p = sub.add_parser("normalize", help="Normalize dataset audio to 24k mono")
    p.add_argument("--raw-jsonl", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--target-sr", type=int, default=24000)
    p.add_argument("--peak-normalize", action="store_true", default=False)
    p.set_defaults(func=cmd_normalize)

    p = sub.add_parser("prepare", help="Run prepare_data.py")
    p.add_argument("--device", default="auto")
    p.add_argument("--tokenizer-model-path", default="Qwen/Qwen3-TTS-Tokenizer-12Hz")
    p.add_argument("--input-jsonl", required=True)
    p.add_argument("--output-filename", default="train_with_codes.jsonl")
    p.add_argument("--batch-infer-num", type=int, default=32)
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("train", help="Run sft_12hz.py")
    p.add_argument("--init-model-path", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    p.add_argument("--train-jsonl", required=True)
    p.add_argument("--run-name", default="")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--num-epochs", type=int, default=3)
    p.add_argument("--speaker-name", required=True)
    p.add_argument("--speaker-id", type=int, default=3000)
    p.add_argument("--gradient-accumulation-steps", type=int, default=4)
    p.add_argument("--mixed-precision", default="auto", choices=["auto", "no", "fp16", "bf16"])
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--subtalker-loss-weight", type=float, default=0.3)
    p.add_argument("--attn-implementation", default="auto")
    p.add_argument("--torch-dtype", default="auto")
    p.add_argument("--log-every-n-steps", type=int, default=10)
    p.add_argument("--save-every-n-epochs", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("pipeline", help="Run prepare + train")
    p.add_argument("--raw-jsonl", required=True)
    p.add_argument("--prepare-device", default="auto")
    p.add_argument("--tokenizer-model-path", default="Qwen/Qwen3-TTS-Tokenizer-12Hz")
    p.add_argument("--prepare-output-filename", default="train_with_codes.jsonl")
    p.add_argument("--prepare-batch-infer-num", type=int, default=32)
    p.add_argument("--init-model-path", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    p.add_argument("--run-name", default="")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--num-epochs", type=int, default=3)
    p.add_argument("--speaker-name", required=True)
    p.add_argument("--speaker-id", type=int, default=3000)
    p.add_argument("--gradient-accumulation-steps", type=int, default=4)
    p.add_argument("--mixed-precision", default="auto", choices=["auto", "no", "fp16", "bf16"])
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--subtalker-loss-weight", type=float, default=0.3)
    p.add_argument("--attn-implementation", default="auto")
    p.add_argument("--torch-dtype", default="auto")
    p.add_argument("--log-every-n-steps", type=int, default=10)
    p.add_argument("--save-every-n-epochs", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_pipeline)

    p = sub.add_parser("infer", help="Single inference from checkpoint")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--speaker-name", required=True)
    p.add_argument("--language", default="auto")
    p.add_argument("--instruct", default="")
    p.add_argument("--text", required=True)
    p.add_argument("--output-path", default="")
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--repetition-penalty", type=float, default=1.05)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--subtalker-temperature", type=float, default=0.9)
    p.add_argument("--subtalker-top-k", type=int, default=50)
    p.add_argument("--subtalker-top-p", type=float, default=1.0)
    p.add_argument(
        "--review-after-generation",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--review-reference-audio", default="")
    p.add_argument("--review-profile-raw-jsonl", default="")
    p.add_argument("--review-base-speaker-model", default="Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    p.add_argument("--review-whisper-model", default="base")
    p.add_argument("--review-output-report", default="")
    p.set_defaults(func=cmd_infer)

    p = sub.add_parser("review-generation", help="Run post-generation quality checks")
    p.add_argument("--generated-wav", required=True)
    p.add_argument("--target-text", required=True)
    p.add_argument("--reference-audio", default="")
    p.add_argument("--profile-raw-jsonl", default="")
    p.add_argument("--base-speaker-model", default="Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    p.add_argument("--whisper-model", default="base")
    p.add_argument("--output-report", default="")
    p.set_defaults(func=cmd_review_generation)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
