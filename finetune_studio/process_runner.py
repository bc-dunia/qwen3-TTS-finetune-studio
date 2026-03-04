from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from typing import Iterable

from .paths import WORKSPACE_ROOT

_LOCK = threading.Lock()
_ACTIVE: dict[str, subprocess.Popen[str]] = {}
_LOCK_DIR = WORKSPACE_ROOT / ".locks"


class AlreadyRunningError(RuntimeError):
    pass


def pid_alive(pid: int | None) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True

def pid_matches_script(pid: int | None, script: str | None) -> bool:
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


def _lock_path(key: str) -> Path:
    _LOCK_DIR.mkdir(parents=True, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in {"_", "-"} else "_" for c in key)
    return (_LOCK_DIR / f"{safe}.pid").resolve()


def _read_lock_pid(key: str) -> int | None:
    info = _read_lock_info(key)
    if info is None:
        return None
    pid_raw = info.get("pid")
    if pid_raw is None:
        return None
    try:
        pid = int(pid_raw)
    except Exception:
        return None
    return pid if pid > 0 else None


def _read_lock_info(key: str) -> dict[str, object] | None:
    path = _lock_path(key)
    if not path.exists():
        return None
    try:
        raw = path.read_text(encoding="utf-8").strip()
        if not raw:
            return None
        if raw.startswith("{"):
            data = json.loads(raw)
            return data if isinstance(data, dict) else None
        # Backward compatibility: old format was plain pid text.
        return {"pid": int(raw)}
    except Exception:
        return None


def _write_lock_pid(key: str, pid: int, command: list[str]) -> None:
    path = _lock_path(key)
    script = None
    if len(command) >= 2:
        candidate = str(command[1])
        if candidate.endswith(".py"):
            script = candidate
    payload = {
        "pid": int(pid),
        "script": script,
        "started_at": time.time(),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _clear_lock(key: str, *, expected_pid: int | None = None) -> None:
    path = _lock_path(key)
    if not path.exists():
        return
    if expected_pid is None:
        path.unlink(missing_ok=True)
        return
    lock_pid = _read_lock_pid(key)
    if lock_pid == int(expected_pid):
        path.unlink(missing_ok=True)


def start_process(
    key: str,
    command: list[str],
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.Popen[str]:
    with _LOCK:
        current = _ACTIVE.get(key)
        if current and current.poll() is None:
            raise AlreadyRunningError(f"A process is already running for key='{key}'.")
        lock_pid = _read_lock_pid(key)
        if pid_alive(lock_pid):
            raise AlreadyRunningError(f"A process is already running for key='{key}' (pid={lock_pid}).")
        if lock_pid is not None:
            _clear_lock(key)

        proc = subprocess.Popen(
            command,
            cwd=str(cwd) if cwd else None,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        _ACTIVE[key] = proc
        _write_lock_pid(key, int(proc.pid), command)
    return proc


def clear_process(key: str, proc: subprocess.Popen[str]) -> None:
    with _LOCK:
        current = _ACTIVE.get(key)
        if current is proc:
            _ACTIVE.pop(key, None)
        _clear_lock(key, expected_pid=int(proc.pid))


def stop_process(key: str) -> str:
    with _LOCK:
        proc = _ACTIVE.get(key)
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
            return "Process terminated."
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
            return "Process killed after timeout."
        finally:
            with _LOCK:
                _ACTIVE.pop(key, None)
            _clear_lock(key, expected_pid=int(proc.pid))

    lock_pid = _read_lock_pid(key)
    if not pid_alive(lock_pid):
        _clear_lock(key)
        return "No running process."

    # Cross-process stop path: send signal to tracked pid.
    assert lock_pid is not None
    info = _read_lock_info(key) or {}
    script = info.get("script")
    if not pid_matches_script(lock_pid, script):
        _clear_lock(key)
        return f"Refusing to stop pid={lock_pid} due to lock mismatch."

    os.kill(lock_pid, signal.SIGTERM)
    deadline = time.time() + 10.0
    while time.time() < deadline:
        if not pid_alive(lock_pid):
            _clear_lock(key, expected_pid=lock_pid)
            return f"Process terminated (pid={lock_pid})."
        time.sleep(0.2)

    os.kill(lock_pid, signal.SIGKILL)
    deadline = time.time() + 5.0
    while time.time() < deadline:
        if not pid_alive(lock_pid):
            _clear_lock(key, expected_pid=lock_pid)
            return f"Process killed after timeout (pid={lock_pid})."
        time.sleep(0.2)

    return f"Failed to stop process (pid={lock_pid})."


def is_running(key: str) -> bool:
    with _LOCK:
        proc = _ACTIVE.get(key)
    if proc and proc.poll() is None:
        return True
    pid = _read_lock_pid(key)
    alive = pid_alive(pid)
    if not alive and pid is not None:
        _clear_lock(key)
        return False

    if alive and pid is not None:
        info = _read_lock_info(key) or {}
        script = info.get("script")
        if not pid_matches_script(pid, script):
            _clear_lock(key)
            return False

    return alive


def tail_logs(lines: Iterable[str], max_lines: int = 600) -> str:
    q = deque(lines, maxlen=max_lines)
    return "\n".join(q)
