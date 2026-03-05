#!/usr/bin/env python3
"""Container smoke test — validates all required imports and system dependencies.

Run during Docker build to catch missing packages early.
"""

from __future__ import annotations

import shutil
import sys


def main() -> int:
    errors: list[str] = []

    # Core modules needed by training handler and scripts
    required_modules = [
        "torch",
        "transformers",
        "accelerate",
        "safetensors",
        "huggingface_hub",
        "soundfile",
        "numpy",
        "boto3",
        "botocore",
        "einops",
        "librosa",
        "torchaudio",
    ]

    # Optional: nice-to-have but not strictly required for training
    optional_modules = [
        "runpod",
        "faster_whisper",
    ]

    for module in required_modules:
        try:
            __import__(module)
        except ImportError as e:
            errors.append(f"Missing module: {module} ({e})")

    for module in optional_modules:
        try:
            __import__(module)
        except ImportError:
            print(f"INFO: Optional module not installed: {module}")

    # Verify qwen_tts and the specific submodules used by training scripts
    # These MUST succeed — training will crash immediately without them.
    qwen_required = [
        ("qwen_tts", "top-level package"),
        ("qwen_tts.core.models.configuration_qwen3_tts", "TTSConfig for dataset.py"),
        ("qwen_tts.core.models.modeling_qwen3_tts", "mel_spectrogram for dataset.py"),
    ]
    for submod, reason in qwen_required:
        try:
            __import__(submod)
        except ImportError as e:
            errors.append(
                f"Missing qwen_tts module: {submod} ({e}) — needed for: {reason}"
            )
        except Exception as e:
            # May fail at build time due to missing CUDA — warn only
            print(f"WARNING: {submod} import raised {type(e).__name__}: {e}")

    # System binaries
    for binary in ["ffmpeg", "python3", "sox"]:
        if not shutil.which(binary):
            errors.append(f"Missing binary: {binary}")

    # CUDA availability (warning only — not an error during build)
    try:
        import torch

        if not torch.cuda.is_available():
            print("WARNING: CUDA not available (expected in GPU container at runtime)")
    except ImportError:
        pass  # already captured above

    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        return 1

    print("All smoke tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
