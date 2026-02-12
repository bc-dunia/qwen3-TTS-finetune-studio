from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from .paths import RUNS_DIR, ensure_workspace_dirs


def run_summary_path(run_dir: str | Path) -> Path:
    return Path(run_dir).resolve() / "run_summary.json"


def update_run_summary(run_dir: str | Path, patch: dict[str, Any]) -> str:
    path = run_summary_path(run_dir)
    path.parent.mkdir(parents=True, exist_ok=True)

    data: dict[str, Any] = {}
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}

    data.update(patch)
    data["updated_at"] = datetime.now().isoformat()

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return str(path)


def read_run_summary(run_dir: str | Path) -> dict[str, Any]:
    path = run_summary_path(run_dir)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _pid_is_alive(pid: int | None) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _pid_matches_script(pid: int | None, script: str | None) -> bool:
    if not script or not isinstance(pid, int) or pid <= 0:
        return True
    try:
        cmdline = subprocess.check_output(
            ["ps", "-p", str(pid), "-o", "command="],
            text=True,
        ).strip()
    except Exception:
        return False
    return str(script) in cmdline


def list_run_summaries(limit: int = 200) -> list[dict[str, Any]]:
    ensure_workspace_dirs()
    summaries: list[dict[str, Any]] = []
    for run_dir in RUNS_DIR.iterdir():
        if not run_dir.is_dir():
            continue
        summary = read_run_summary(run_dir)
        if not summary:
            summary = {
                "run_dir": str(run_dir.resolve()),
                "status": "unknown",
                "created_at": datetime.fromtimestamp(run_dir.stat().st_ctime).isoformat(),
            }
        else:
            summary.setdefault("run_dir", str(run_dir.resolve()))
            if str(summary.get("status", "")).lower() == "running":
                pid_raw = summary.get("process_pid")
                pid: int | None = None
                try:
                    if pid_raw is not None:
                        pid = int(pid_raw)
                except Exception:
                    pid = None
                script = summary.get("process_script")
                is_stale = (
                    pid is None
                    or not _pid_is_alive(pid)
                    or not _pid_matches_script(pid, str(script) if script else None)
                )
                if is_stale:
                    summary["status"] = "interrupted"
                    summary["stale_process"] = True
                    summary["updated_at"] = datetime.now().isoformat()
                    with run_summary_path(run_dir).open("w", encoding="utf-8") as f:
                        json.dump(summary, f, indent=2, ensure_ascii=False)
        summaries.append(summary)

    summaries.sort(key=lambda x: x.get("updated_at", x.get("created_at", "")), reverse=True)
    return summaries[:limit]


def run_summaries_table(limit: int = 100) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for item in list_run_summaries(limit=limit):
        rows.append(
            [
                item.get("status", ""),
                item.get("run_name", ""),
                item.get("speaker_name", ""),
                item.get("train_jsonl", ""),
                item.get("last_loss", ""),
                item.get("epochs_done", ""),
                item.get("checkpoints", 0),
                item.get("created_at", ""),
                item.get("run_dir", ""),
            ]
        )
    return rows
