from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from .paths import EXPORTS_DIR, ensure_workspace_dirs
from .run_registry import read_run_summary


def package_checkpoint(
    checkpoint_path: str,
    *,
    include_optimizer_files: bool = True,
) -> tuple[str, str]:
    ckpt = Path(checkpoint_path).resolve()
    if not ckpt.exists() or not ckpt.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt}")

    ensure_workspace_dirs()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_dir = (EXPORTS_DIR / f"checkpoint_package_{ts}").resolve()
    package_root = export_dir / ckpt.name
    package_root.mkdir(parents=True, exist_ok=True)

    ignore_patterns = []
    if not include_optimizer_files:
        ignore_patterns.extend(["optimizer*", "scheduler*", "rng_state*"])
    ignore = shutil.ignore_patterns(*ignore_patterns) if ignore_patterns else None
    shutil.copytree(ckpt, package_root / "checkpoint", dirs_exist_ok=True, ignore=ignore)

    run_dir = ckpt.parent
    manifest: dict[str, Any] = {
        "created_at": datetime.now().isoformat(),
        "checkpoint_path": str(ckpt),
        "run_dir": str(run_dir.resolve()),
        "checkpoint_name": ckpt.name,
    }

    run_config_path = run_dir / "run_config.json"
    if run_config_path.exists():
        manifest["run_config_path"] = str(run_config_path.resolve())
        with run_config_path.open("r", encoding="utf-8") as f:
            manifest["run_config"] = json.load(f)
        shutil.copy2(run_config_path, package_root / "run_config.json")

    run_summary = read_run_summary(run_dir)
    if run_summary:
        manifest["run_summary"] = run_summary
        with (package_root / "run_summary.json").open("w", encoding="utf-8") as f:
            json.dump(run_summary, f, indent=2, ensure_ascii=False)

    with (package_root / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    archive_base = export_dir / f"{ckpt.name}"
    zip_path = shutil.make_archive(str(archive_base), "zip", root_dir=package_root)
    message = f"Checkpoint package created: {zip_path}"
    return zip_path, message

