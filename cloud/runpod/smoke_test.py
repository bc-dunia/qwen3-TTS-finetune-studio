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
    ]

    # Optional: installed but may have missing transitive deps (--no-deps)
    optional_modules = [
        "qwen_tts",  # installed --no-deps, top-level import needs librosa
        "runpod",
        "librosa",
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

    # Verify qwen_tts submodules used by training scripts work
    qwen_submodules = [
        "qwen_tts.core.models.configuration_qwen3_tts",
        "qwen_tts.core.models.modeling_qwen3_tts",
    ]
    for submod in qwen_submodules:
        try:
            __import__(submod)
        except ImportError as e:
            # May fail at build time due to missing CUDA — warn only
            print(f"WARNING: {submod} import failed: {e}")
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
