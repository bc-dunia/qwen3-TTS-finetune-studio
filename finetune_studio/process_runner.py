from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from typing import Iterable, Literal

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


def pid_script_match_state(
    pid: int | None, script: str | None
) -> Literal["match", "mismatch", "unknown"]:
    if not script or not isinstance(pid, int) or pid <= 0:
        return "match"
    cmdline = ""
    try:
        psutil = __import__("psutil")
        cmdline = " ".join(psutil.Process(pid).cmdline()).strip()
    except Exception:
        cmdline = ""
    if not cmdline:
        proc_cmdline_path = Path(f"/proc/{pid}/cmdline")
        if proc_cmdline_path.exists():
            try:
                raw = proc_cmdline_path.read_bytes().replace(b"\x00", b" ")
                cmdline = raw.decode("utf-8").strip()
            except Exception:
                cmdline = ""
    if not cmdline:
        try:
            cmdline = subprocess.check_output(
                ["ps", "-p", str(pid), "-o", "command="],
                text=True,
            ).strip()
        except Exception:
            return "unknown"
    try:
        tokens = shlex.split(cmdline)
    except ValueError:
        return "unknown"
    return "match" if script in tokens else "mismatch"


def pid_matches_script(pid: int | None, script: str | None) -> bool:
    return pid_script_match_state(pid, script) == "match"


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
    if isinstance(pid_raw, bool):
        return None
    if isinstance(pid_raw, int):
        pid = pid_raw
    elif isinstance(pid_raw, str):
        try:
            pid = int(pid_raw)
        except ValueError:
            return None
    else:
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
        return {
            "pid": int(raw),
            "started_at": path.stat().st_mtime,
        }
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
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp_path.write_text(json.dumps(payload), encoding="utf-8")
    tmp_path.replace(path)


def _clear_lock(
    key: str,
    *,
    expected_pid: int | None = None,
    expected_started_at: float | None = None,
) -> None:
    path = _lock_path(key)
    if not path.exists():
        return
    if expected_pid is None and expected_started_at is None:
        path.unlink(missing_ok=True)
        return
    info = _read_lock_info(key) or {}
    lock_pid = _read_lock_pid(key)
    if expected_pid is not None and lock_pid != int(expected_pid):
        return
    if expected_started_at is not None:
        started_at = info.get("started_at")
        if not isinstance(started_at, (int, float)):
            return
        if abs(float(started_at) - float(expected_started_at)) > 1e-6:
            return
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
        started_at: float | None = None
        if pid_alive(lock_pid):
            info = _read_lock_info(key) or {}
            script_raw = info.get("script")
            script = script_raw if isinstance(script_raw, str) else None
            started_at_raw = info.get("started_at")
            started_at = (
                float(started_at_raw)
                if isinstance(started_at_raw, (int, float))
                else None
            )
            if script is None:
                _clear_lock(key, expected_pid=lock_pid, expected_started_at=started_at)
            else:
                match_state = pid_script_match_state(lock_pid, script)
                if match_state == "mismatch":
                    _clear_lock(
                        key, expected_pid=lock_pid, expected_started_at=started_at
                    )
                else:
                    raise AlreadyRunningError(
                        f"A process is already running for key='{key}' (pid={lock_pid})."
                    )
        if lock_pid is not None:
            _clear_lock(key, expected_pid=lock_pid, expected_started_at=started_at)

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
        info = _read_lock_info(key) or {}
        started_at_raw = info.get("started_at")
        started_at = (
            float(started_at_raw) if isinstance(started_at_raw, (int, float)) else None
        )
        _clear_lock(key, expected_pid=int(proc.pid), expected_started_at=started_at)


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
            info = _read_lock_info(key) or {}
            started_at_raw = info.get("started_at")
            started_at = (
                float(started_at_raw)
                if isinstance(started_at_raw, (int, float))
                else None
            )
            _clear_lock(key, expected_pid=int(proc.pid), expected_started_at=started_at)

    lock_pid = _read_lock_pid(key)
    if not pid_alive(lock_pid):
        info = _read_lock_info(key) or {}
        started_at_raw = info.get("started_at")
        started_at = (
            float(started_at_raw) if isinstance(started_at_raw, (int, float)) else None
        )
        _clear_lock(key, expected_pid=lock_pid, expected_started_at=started_at)
        return "No running process."

    # Cross-process stop path: send signal to tracked pid.
    assert lock_pid is not None
    info = _read_lock_info(key) or {}
    script_raw = info.get("script")
    script = script_raw if isinstance(script_raw, str) else None
    started_at_raw = info.get("started_at")
    started_at = (
        float(started_at_raw) if isinstance(started_at_raw, (int, float)) else None
    )
    if script is None:
        _clear_lock(key, expected_pid=lock_pid, expected_started_at=started_at)
        return "No running process."
    match_state = pid_script_match_state(lock_pid, script)
    if match_state == "mismatch":
        _clear_lock(key, expected_pid=lock_pid, expected_started_at=started_at)
        return f"Refusing to stop pid={lock_pid} due to lock mismatch."
    if match_state == "unknown":
        return f"Unable to verify pid={lock_pid} ownership from lock metadata."

    try:
        os.kill(lock_pid, signal.SIGTERM)
    except ProcessLookupError:
        _clear_lock(key, expected_pid=lock_pid, expected_started_at=started_at)
        return f"Process terminated (pid={lock_pid})."
    except PermissionError:
        return f"Permission denied while stopping pid={lock_pid}."
    deadline = time.time() + 10.0
    while time.time() < deadline:
        if not pid_alive(lock_pid):
            _clear_lock(key, expected_pid=lock_pid, expected_started_at=started_at)
            return f"Process terminated (pid={lock_pid})."
        time.sleep(0.2)

    try:
        os.kill(lock_pid, signal.SIGKILL)
    except ProcessLookupError:
        _clear_lock(key, expected_pid=lock_pid, expected_started_at=started_at)
        return f"Process terminated (pid={lock_pid})."
    except PermissionError:
        return f"Permission denied while killing pid={lock_pid}."
    deadline = time.time() + 5.0
    while time.time() < deadline:
        if not pid_alive(lock_pid):
            _clear_lock(key, expected_pid=lock_pid, expected_started_at=started_at)
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
        info = _read_lock_info(key) or {}
        started_at_raw = info.get("started_at")
        started_at = (
            float(started_at_raw) if isinstance(started_at_raw, (int, float)) else None
        )
        _clear_lock(key, expected_pid=pid, expected_started_at=started_at)
        return False

    if alive and pid is not None:
        info = _read_lock_info(key) or {}
        script_raw = info.get("script")
        script = script_raw if isinstance(script_raw, str) else None
        started_at_raw = info.get("started_at")
        started_at = (
            float(started_at_raw) if isinstance(started_at_raw, (int, float)) else None
        )
        if script is None:
            _clear_lock(key, expected_pid=pid, expected_started_at=started_at)
            return False
        match_state = pid_script_match_state(pid, script)
        if match_state == "unknown":
            return True
        if match_state == "mismatch":
            _clear_lock(key, expected_pid=pid, expected_started_at=started_at)
            return False

    return alive


def tail_logs(lines: Iterable[str], max_lines: int = 600) -> str:
    q = deque(lines, maxlen=max_lines)
    return "\n".join(q)
