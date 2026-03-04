#!/usr/bin/env python3
"""Container smoke test — validates all required imports and system dependencies.

Run during Docker build to catch missing packages early.
"""

from __future__ import annotations

import shutil
import sys


def main() -> int:
    errors: list[str] = []

    # Core Python modules required by training and inference handlers
    required_modules = [
        "torch",
        "transformers",
        "accelerate",
        "safetensors",
        "huggingface_hub",
        "soundfile",
        "numpy",
        "librosa",
        "boto3",
        "botocore",
        "runpod",
        "qwen_tts",
    ]

    for module in required_modules:
        try:
            __import__(module)
        except ImportError as e:
            errors.append(f"Missing module: {module} ({e})")

    # System binaries
    for binary in ["ffmpeg", "python3"]:
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
