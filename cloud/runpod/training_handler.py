from __future__ import annotations

# pyright: reportMissingImports=false
import json
import importlib
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .r2_storage import R2Storage
else:
    try:
        from .r2_storage import R2Storage
    except ImportError:
        from r2_storage import R2Storage

LOGGER = logging.getLogger("training_handler")
PREPARE_PROGRESS_RE = re.compile(
    r"(?:Done\.\s*)?Encoded\s+(\d+)\s+audio files", re.IGNORECASE
)
TRAINING_PROGRESS_RE = re.compile(
    r"Epoch\s+(\d+)\s*(?:\||,)\s*Step\s+(\d+)\s*(?:\||,)\s*Loss:\s*([0-9]*\.?[0-9]+)",
    re.IGNORECASE,
)
CHECKPOINT_DIR_RE = re.compile(r"checkpoint-epoch-(\d+)$")
SAFE_ID_RE = re.compile(r"^[a-zA-Z0-9._-]{1,128}$")
NORMALIZED_TEXT_RE = re.compile(r"[^0-9a-zA-Z가-힣]+")

JOB_ID_ENV = "JOB_ID"
JOB_TOKEN_ENV = "JOB_TOKEN"
WORKER_API_URL_ENV = "WORKER_API_URL"
DATASET_DIR = Path("/tmp/dataset")
OUTPUT_ROOT = Path("/tmp/output")
APP_FINETUNE_DIR = Path("/app/finetuning")
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[2] if len(_THIS_FILE.parents) > 2 else _THIS_FILE.parent
REPO_FINETUNE_DIR = _REPO_ROOT / "third_party" / "Qwen3-TTS" / "finetuning"

TARGET_SAMPLE_RATE = 24000
MIN_SEGMENT_SEC = 3.0
MAX_SEGMENT_SEC = 15.0
IDEAL_REF_MIN_SEC = 4.0
IDEAL_REF_MAX_SEC = 8.0
MIN_ACCEPTED_SEGMENTS = 25
MIN_ACCEPTED_AUDIO_MIN = 6.0
MIN_TRANSCRIPT_CHARS = 6
MIN_CONFIDENCE = 0.55
MAX_DUPLICATES_PER_TEXT = 2
# Small runs often peak early, so keep a wider checkpoint history available for
# validation and manual compare instead of only the latest few epochs.
MAX_UPLOADED_CHECKPOINTS = 10
HEARTBEAT_INTERVAL_SEC = 30.0


@dataclass
class JobConfig:
    voice_id: str
    dataset_r2_prefix: str
    speaker_name: str
    model_size: str
    batch_size: int
    learning_rate: float
    num_epochs: int
    run_name: str
    gradient_accumulation_steps: int
    speaker_id: int
    mixed_precision: str
    torch_dtype: str
    attn_implementation: str
    weight_decay: float
    max_grad_norm: float
    subtalker_loss_weight: float
    log_every_n_steps: int
    save_every_n_epochs: int
    max_steps: int
    seed: int
    worker_api_url: str | None = None
    job_token: str | None = None
    whisper_language: str | None = None  # None = auto-detect; e.g. "ko", "en", "zh", "ja"
    dataset_signature: str | None = None
    preprocess_cache_r2_prefix: str | None = None


class UnrecoverableError(Exception):
    pass


class WorkerReporter:
    def __init__(self, worker_url: str, job_id: str, job_token: str) -> None:
        self.worker_url = worker_url.rstrip("/")
        self.job_id = job_id
        self.job_token = job_token
        self._requests = importlib.import_module("requests")

    def _post(self, path: str, payload: dict[str, Any]) -> None:
        url = f"{self.worker_url}{path}"
        headers = {
            "Authorization": f"Bearer {self.job_token}",
            "Content-Type": "application/json",
        }

        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                response = self._requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=10,
                )
                if 200 <= response.status_code < 300:
                    return
                raise RuntimeError(
                    f"Worker callback failed ({response.status_code}): {response.text}"
                )
            except Exception as exc:
                last_exc = exc
                if attempt < 2:
                    time.sleep(0.7 * (attempt + 1))
                    continue
                LOGGER.error(
                    "Worker callback error path=%s payload=%s err=%s",
                    path,
                    payload,
                    exc,
                )

        if last_exc is not None:
            LOGGER.debug("Worker callback giving up: %s", last_exc)

    def heartbeat(
        self, progress: dict[str, Any] | None = None, message: str | None = None
    ) -> None:
        payload: dict[str, Any] = {}
        if progress is not None:
            payload["progress"] = progress
        if message:
            payload["message"] = message
        self._post(f"/v1/internal/training/{self.job_id}/heartbeat", payload)

    def report(
        self,
        *,
        status: str,
        progress: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        message: str | None = None,
        error: str | None = None,
        checkpoints: list[dict[str, Any]] | None = None,
    ) -> None:
        payload: dict[str, Any] = {"status": status}
        if progress is not None:
            payload["progress"] = progress
        if metrics is not None:
            payload["metrics"] = metrics
        if message:
            payload["message"] = message
        if error:
            payload["error"] = error
        if checkpoints is not None:
            payload["checkpoints"] = checkpoints
        self._post(f"/v1/internal/training/{self.job_id}/report", payload)

    def upload_log_chunk(
        self, seq: int, r2_key: str, bytes_count: int, lines: int
    ) -> None:
        self._post(
            f"/v1/internal/training/{self.job_id}/log",
            {
                "seq": seq,
                "r2_key": r2_key,
                "bytes": bytes_count,
                "lines": lines,
            },
        )

    def report_preprocess_cache(
        self,
        *,
        dataset_signature: str,
        cache_r2_prefix: str,
        train_raw_r2_key: str,
        ref_audio_r2_key: str | None,
        reference_profile_r2_key: str | None,
        source_file_count: int | None,
        segments_created: int | None,
        segments_accepted: int | None,
        accepted_duration_min: float | None,
    ) -> None:
        payload: dict[str, Any] = {
            "dataset_signature": dataset_signature,
            "cache_r2_prefix": cache_r2_prefix,
            "train_raw_r2_key": train_raw_r2_key,
            "ref_audio_r2_key": ref_audio_r2_key,
            "reference_profile_r2_key": reference_profile_r2_key,
            "source_file_count": source_file_count,
            "segments_created": segments_created,
            "segments_accepted": segments_accepted,
            "accepted_duration_min": accepted_duration_min,
        }
        self._post(f"/v1/internal/training/{self.job_id}/preprocess-cache", payload)


class LogBuffer:
    def __init__(
        self,
        r2: R2Storage,
        job_id: str,
        reporter: WorkerReporter | None,
        flush_line_count: int = 200,
        flush_interval_sec: float = 30.0,
    ) -> None:
        self.r2 = r2
        self.job_id = job_id
        self.reporter = reporter
        self.flush_line_count = flush_line_count
        self.flush_interval_sec = flush_interval_sec
        self._lines: list[dict[str, Any]] = []
        self._seq = 0
        self._last_flush_ts = time.time()

    def append(self, line: str) -> None:
        now = time.time()
        self._lines.append({"ts": now, "msg": line})
        if (
            len(self._lines) >= self.flush_line_count
            or (now - self._last_flush_ts) >= self.flush_interval_sec
        ):
            self.flush()

    def flush(self) -> None:
        if not self._lines:
            return

        r2_key = f"jobs/{self.job_id}/logs/{self._seq:08d}.jsonl"
        payload_lines = [json.dumps(item, ensure_ascii=False) for item in self._lines]
        payload = ("\n".join(payload_lines) + "\n").encode("utf-8")
        line_count = len(self._lines)
        self.r2.upload_bytes(payload, r2_key, content_type="application/jsonl")

        if self.reporter is not None:
            self.reporter.upload_log_chunk(self._seq, r2_key, len(payload), line_count)

        self._seq += 1
        self._lines = []
        self._last_flush_ts = time.time()

    def close(self) -> None:
        self.flush()


class StatusWriter:
    def __init__(
        self, r2: R2Storage, job_id: str, reporter: WorkerReporter | None = None
    ) -> None:
        self.r2 = r2
        self.job_id = job_id
        self.reporter = reporter

    def write(self, payload: dict[str, Any]) -> None:
        data = dict(payload)
        data["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.r2.write_job_status(self.job_id, data)
        LOGGER.info("Status update: %s", json.dumps(data, ensure_ascii=False))

        if self.reporter is not None and isinstance(data.get("status"), str):
            self.reporter.report(
                status=str(data["status"]),
                progress=data.get("progress")
                if isinstance(data.get("progress"), dict)
                else None,
                metrics=data.get("metrics")
                if isinstance(data.get("metrics"), dict)
                else None,
                checkpoints=data.get("checkpoints")
                if isinstance(data.get("checkpoints"), list)
                else None,
                message=str(data["message"])
                if isinstance(data.get("message"), str)
                else None,
                error=str(data["error"])
                if isinstance(data.get("error"), str)
                else None,
            )

    def heartbeat(
        self,
        progress: dict[str, Any] | None = None,
        message: str | None = None,
    ) -> None:
        if self.reporter is None:
            return
        self.reporter.heartbeat(progress=progress, message=message)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def resolve_finetune_dir() -> Path:
    if APP_FINETUNE_DIR.exists():
        return APP_FINETUNE_DIR
    if REPO_FINETUNE_DIR.exists():
        return REPO_FINETUNE_DIR
    raise FileNotFoundError(
        f"Finetuning dir not found: {APP_FINETUNE_DIR} or {REPO_FINETUNE_DIR}"
    )


def parse_job_config(raw: dict[str, Any]) -> JobConfig:
    required = [
        "voice_id",
        "dataset_r2_prefix",
        "speaker_name",
        "model_size",
        "batch_size",
        "learning_rate",
        "num_epochs",
        "run_name",
    ]
    missing = [k for k in required if k not in raw]
    if missing:
        raise ValueError(f"Config missing required keys: {', '.join(missing)}")
    cfg = JobConfig(
        voice_id=str(raw["voice_id"]),
        dataset_r2_prefix=str(raw["dataset_r2_prefix"]),
        speaker_name=str(raw["speaker_name"]),
        model_size=str(raw["model_size"]),
        batch_size=int(raw["batch_size"]),
        learning_rate=float(raw["learning_rate"]),
        num_epochs=int(raw["num_epochs"]),
        run_name=str(raw["run_name"]),
        gradient_accumulation_steps=int(raw.get("gradient_accumulation_steps", 4)),
        speaker_id=int(raw.get("speaker_id", 3000)),
        mixed_precision=str(raw.get("mixed_precision", "bf16")),
        torch_dtype=str(raw.get("torch_dtype", "bfloat16")),
        attn_implementation=str(raw.get("attn_implementation", "flash_attention_2")),
        weight_decay=float(raw.get("weight_decay", 0.01)),
        max_grad_norm=float(raw.get("max_grad_norm", 1.0)),
        subtalker_loss_weight=float(raw.get("subtalker_loss_weight", 0.3)),
        log_every_n_steps=int(raw.get("log_every_n_steps", 10)),
        save_every_n_epochs=int(raw.get("save_every_n_epochs", 1)),
        max_steps=int(raw.get("max_steps", 0)),
        seed=int(raw.get("seed", 42)),
        worker_api_url=(str(raw.get("worker_api_url", "")).strip() or None),
        job_token=(str(raw.get("job_token", "")).strip() or None),
        whisper_language=(str(raw.get("whisper_language", "")).strip() or None),
        dataset_signature=(str(raw.get("dataset_signature", "")).strip() or None),
        preprocess_cache_r2_prefix=(
            str(raw.get("preprocess_cache_r2_prefix", "")).strip() or None
        ),
    )
    # Normalize whisper_language: "auto" or empty → None (auto-detect)
    if cfg.whisper_language and cfg.whisper_language.lower() == "auto":
        cfg.whisper_language = None
    for field_name, field_value in [
        ("voice_id", cfg.voice_id),
        ("run_name", cfg.run_name),
        ("speaker_name", cfg.speaker_name),
    ]:
        if not SAFE_ID_RE.match(field_value):
            raise ValueError(
                f"Invalid {field_name}: must be alphanumeric/dash/dot/underscore, got: {field_value!r}"
            )
    return cfg


def parse_dataset_prefix(prefix: str) -> tuple[str, str]:
    """Parse dataset_r2_prefix into (voice_id, dataset_name).

    Accepts both:
      datasets/{voice_id}/{name} -> (voice_id, name)
      datasets/{voice_id}         -> (voice_id, "")  # root-level dataset
    """
    parts = [p for p in prefix.strip("/").split("/") if p]
    if len(parts) < 2 or parts[0] != "datasets":
        raise ValueError(
            f"dataset_r2_prefix must start with datasets/{{voice_id}}, got: {prefix}"
        )
    voice_id = parts[1]
    dataset_name = "/".join(parts[2:]) if len(parts) > 2 else ""
    return voice_id, dataset_name


def _resolve_hf_model(model_name: str, label: str) -> str:
    """Check local paths then download from HuggingFace."""
    for candidate in [
        Path("/models") / model_name,
        Path("/runpod-vol/models") / model_name,
    ]:
        if candidate.is_dir():
            resolved = str(candidate.resolve())
            LOGGER.info("Using local %s: %s", label, resolved)
            return resolved
    repo_id = f"Qwen/{model_name}"
    hf_token = os.environ.get("HF_TOKEN", "").strip() or None
    LOGGER.info("Downloading %s from HuggingFace: %s", label, repo_id)
    snapshot_download = importlib.import_module("huggingface_hub").snapshot_download
    try:
        downloaded = snapshot_download(repo_id=repo_id, token=hf_token)
    except Exception:
        if hf_token:
            LOGGER.warning("HF token failed; retrying anonymous download")
            downloaded = snapshot_download(repo_id=repo_id, token=False)
        else:
            raise
    LOGGER.info("Downloaded %s path: %s", label, downloaded)
    return downloaded


def resolve_tokenizer_model() -> str:
    """Resolve tokenizer model path (separate from base model)."""
    return _resolve_hf_model("Qwen3-TTS-Tokenizer-12Hz", "tokenizer")


def count_jsonl_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def resolve_base_model(model_size: str) -> str:
    model_name = f"Qwen3-TTS-12Hz-{model_size.strip()}-Base"
    return _resolve_hf_model(model_name, "base model")


def stream_subprocess(
    command: list[str],
    *,
    cwd: Path,
    callback: Callable[[str], None],
    heartbeat: Callable[[], None] | None = None,
) -> int:
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        command,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    stop_heartbeat = threading.Event()
    heartbeat_thread: threading.Thread | None = None
    if heartbeat is not None:
        def heartbeat_loop() -> None:
            while not stop_heartbeat.wait(HEARTBEAT_INTERVAL_SEC):
                if proc.poll() is not None:
                    return
                try:
                    heartbeat()
                except Exception:
                    LOGGER.exception("Subprocess heartbeat callback failed")

        heartbeat_thread = threading.Thread(
            target=heartbeat_loop,
            name="subprocess-heartbeat",
            daemon=True,
        )
        heartbeat_thread.start()

    try:
        for raw in proc.stdout:
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            LOGGER.info("[subprocess] %s", line)
            callback(line)
        return proc.wait()
    finally:
        stop_heartbeat.set()
        if heartbeat_thread is not None:
            heartbeat_thread.join(timeout=1.0)


def rewrite_jsonl_paths(jsonl_path: Path, dataset_dir: Path) -> None:
    """Rewrite audio/ref_audio paths in JSONL to be absolute under dataset_dir.

    Builds a filename -> absolute-path lookup by scanning dataset_dir recursively,
    then resolves each audio/ref_audio field by filename match. This handles files
    in subdirectories (e.g., segments/seg_000001.wav).
    """
    # Build filename -> path lookup (last one wins if duplicates)
    file_lookup: dict[str, str] = {}
    for fpath in dataset_dir.rglob("*"):
        if fpath.is_file():
            file_lookup[fpath.name] = str(fpath)

    rows: list[dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    rewritten = 0
    for row in rows:
        for key in ("audio", "ref_audio"):
            if key in row and row[key]:
                original = str(row[key])
                filename = Path(original).name
                # Try lookup first, fall back to dataset_dir / filename
                resolved = file_lookup.get(filename, str(dataset_dir / filename))
                if resolved != original:
                    rewritten += 1
                row[key] = resolved
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    LOGGER.info(
        "Rewrote %d paths in %s (total %d rows)", rewritten, jsonl_path, len(rows)
    )


def validate_dataset(jsonl_path: Path) -> None:
    """Validate that all JSONL entries have required keys and referenced files exist."""
    errors: list[str] = []
    audio_files: set[str] = set()
    ref_audio_files: set[str] = set()
    item_count = 0

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, start=1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            item_count += 1
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as e:
                errors.append(f"Line {line_num}: invalid JSON: {e}")
                continue

            for key in ("audio", "text", "ref_audio"):
                if key not in row:
                    errors.append(f"Line {line_num}: missing required key '{key}'")

            text = str(row.get("text", "")).strip()
            if not text:
                errors.append(f"Line {line_num}: 'text' is empty")

            audio_path = str(row.get("audio", ""))
            if audio_path:
                audio_files.add(audio_path)
                if not Path(audio_path).exists():
                    errors.append(
                        f"Line {line_num}: audio file not found: {audio_path}"
                    )

            ref_path = str(row.get("ref_audio", ""))
            if ref_path:
                ref_audio_files.add(ref_path)
                if not Path(ref_path).exists():
                    errors.append(
                        f"Line {line_num}: ref_audio file not found: {ref_path}"
                    )

    if errors:
        raise ValueError(
            f"Dataset validation failed with {len(errors)} error(s):\n"
            + "\n".join(errors[:20])
        )

    LOGGER.info(
        "Dataset validated: %d items, %d unique audio files, %d unique ref_audio files",
        item_count,
        len(audio_files),
        len(ref_audio_files),
    )


def validate_audio_format(jsonl_path: Path) -> None:
    """Validate audio files are 24kHz. Resample if needed (safety net)."""
    import soundfile as sf

    all_audio: set[str] = set()
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Malformed JSONL at line {line_no} in {jsonl_path}: {exc}"
                ) from exc
            for key in ("audio", "ref_audio"):
                val = str(row.get(key, "")).strip()
                if val:
                    all_audio.add(val)

    resampled = 0
    for audio_path in sorted(all_audio):
        p = Path(audio_path)
        if not p.exists():
            continue
        try:
            info = sf.info(str(p))
        except Exception as exc:
            LOGGER.warning("Cannot read audio file %s: %s", p, exc)
            continue

        if info.samplerate != 24000:
            LOGGER.warning(
                "Audio %s has sr=%d, resampling to 24000", p, info.samplerate
            )
            try:
                import librosa
                import numpy as np

                audio_data, _ = librosa.load(str(p), sr=24000, mono=True)
                sf.write(str(p), audio_data.astype(np.float32), 24000)
                resampled += 1
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to resample {p} from {info.samplerate}Hz to 24000Hz: {exc}"
                ) from exc

    if resampled > 0:
        LOGGER.info("Resampled %d audio file(s) to 24kHz", resampled)
    else:
        LOGGER.info("All %d audio file(s) are 24kHz", len(all_audio))


def _probe_duration_seconds(audio_path: Path) -> float:
    import soundfile as sf

    return float(sf.info(str(audio_path)).duration)


def _run_ffmpeg_capture(
    command: list[str], *, heartbeat: Callable[[], None] | None = None
) -> str:
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    while True:
        try:
            stdout, stderr = proc.communicate(timeout=HEARTBEAT_INTERVAL_SEC)
            break
        except subprocess.TimeoutExpired:
            if heartbeat is not None:
                heartbeat()

    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed (exit={proc.returncode}): {' '.join(command)}\n{stderr}"
        )
    return (stdout or "") + (stderr or "")


def _detect_silence_intervals(
    audio_path: Path,
    noise_db: int = -30,
    min_silence: float = 0.5,
    heartbeat: Callable[[], None] | None = None,
) -> list[tuple[float, float]]:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-i",
        str(audio_path),
        "-af",
        f"silencedetect=noise={noise_db}dB:d={min_silence}",
        "-f",
        "null",
        "-",
    ]
    output = _run_ffmpeg_capture(cmd, heartbeat=heartbeat)
    starts: list[float] = []
    ends: list[float] = []
    for line in output.splitlines():
        s_match = re.search(r"silence_start:\s*([0-9]*\.?[0-9]+)", line)
        if s_match:
            starts.append(float(s_match.group(1)))
            continue
        e_match = re.search(r"silence_end:\s*([0-9]*\.?[0-9]+)", line)
        if e_match:
            ends.append(float(e_match.group(1)))

    intervals: list[tuple[float, float]] = []
    pair_count = min(len(starts), len(ends))
    for idx in range(pair_count):
        start = max(0.0, starts[idx])
        end = max(start, ends[idx])
        intervals.append((start, end))
    return intervals


def _build_speech_segments(
    audio_path: Path, heartbeat: Callable[[], None] | None = None
) -> list[tuple[float, float]]:
    duration = _probe_duration_seconds(audio_path)
    silences = _detect_silence_intervals(audio_path, heartbeat=heartbeat)

    speech_intervals: list[tuple[float, float]] = []
    cursor = 0.0
    for silence_start, silence_end in silences:
        if silence_start > cursor:
            speech_intervals.append((cursor, silence_start))
        cursor = max(cursor, silence_end)
    if cursor < duration:
        speech_intervals.append((cursor, duration))

    split_intervals: list[tuple[float, float]] = []
    for seg_start, seg_end in speech_intervals:
        start = max(0.0, seg_start)
        end = min(duration, seg_end)
        while end - start > MAX_SEGMENT_SEC:
            split_intervals.append((start, start + MAX_SEGMENT_SEC))
            start += MAX_SEGMENT_SEC
        if end - start >= MIN_SEGMENT_SEC:
            split_intervals.append((start, end))

    merged: list[tuple[float, float]] = []
    idx = 0
    while idx < len(split_intervals):
        start, end = split_intervals[idx]
        if (end - start) >= 3.0:
            merged.append((start, end))
            idx += 1
            continue

        while idx + 1 < len(split_intervals):
            n_start, n_end = split_intervals[idx + 1]
            gap = max(0.0, n_start - end)
            candidate_end = n_end
            if gap > 0.6:
                break
            if candidate_end - start > MAX_SEGMENT_SEC:
                break
            end = candidate_end
            idx += 1
            if end - start >= MIN_SEGMENT_SEC:
                break

        if end - start >= MIN_SEGMENT_SEC:
            merged.append((start, end))
        idx += 1

    return [
        (start, end - start)
        for start, end in merged
        if MIN_SEGMENT_SEC <= (end - start) <= MAX_SEGMENT_SEC
    ]


def _normalize_text(text: str) -> str:
    return NORMALIZED_TEXT_RE.sub("", (text or "").strip().lower())


def _compute_signal_metrics(audio_path: Path) -> dict[str, float]:
    import numpy as np
    import soundfile as sf

    samples, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
    audio = np.asarray(samples, dtype=np.float32)
    if audio.size == 0:
        return {
            "rms": 0.0,
            "peak": 0.0,
            "silence_ratio": 1.0,
            "clipping_ratio": 0.0,
            "duration_sec": 0.0,
        }
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    rms = float(np.sqrt(np.mean(np.square(audio)) + 1e-12))
    peak = float(np.max(np.abs(audio)))
    silence_ratio = float(np.mean(np.abs(audio) < 1e-4))
    clipping_ratio = float(np.mean(np.abs(audio) >= 0.999))
    duration_sec = float(len(audio) / max(int(sr), 1))
    return {
        "rms": rms,
        "peak": peak,
        "silence_ratio": silence_ratio,
        "clipping_ratio": clipping_ratio,
        "duration_sec": duration_sec,
    }


def _compute_rms(audio_path: Path) -> float:
    import numpy as np
    import soundfile as sf

    samples, _ = sf.read(str(audio_path), always_2d=False)
    arr = np.asarray(samples, dtype=np.float32)
    if arr.size == 0:
        return 0.0
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    return float(np.sqrt(np.mean(np.square(arr))))


def preprocess_raw_audio(
    dataset_dir: Path, output_jsonl: Path, status: StatusWriter,
    whisper_language: str | None = None,
) -> dict[str, Any]:
    import torch

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise UnrecoverableError(
            "faster-whisper is not installed in this image. "
            "Preprocessing requires faster-whisper. Please provide a pre-built train_raw.jsonl in the dataset."
        )

    status.write(
        {
            "status": "preprocessing",
            "message": "Scanning dataset for raw audio files...",
        }
    )

    allowed_ext = {".wav", ".mp4", ".mp3", ".flac", ".m4a"}
    source_audio = [
        p
        for p in sorted(dataset_dir.rglob("*"))
        if p.is_file() and p.suffix.lower() in allowed_ext and "segments" not in p.parts
    ]
    if not source_audio:
        raise FileNotFoundError(f"No raw audio files found under {dataset_dir}")

    converted_dir = dataset_dir / "converted"
    converted_dir.mkdir(parents=True, exist_ok=True)
    wav_inputs: list[Path] = []

    status.write(
        {
            "status": "preprocessing",
            "message": "Converting audio files to 24kHz mono WAV...",
            "progress": {"step": 0, "total": len(source_audio)},
        }
    )
    for idx, src in enumerate(source_audio, start=1):
        converted_path = converted_dir / f"src_{idx:04d}.wav"
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-y",
            "-i",
            str(src),
            "-ac",
            "1",
            "-ar",
            str(TARGET_SAMPLE_RATE),
            "-c:a",
            "pcm_s16le",
            str(converted_path),
        ]
        _run_ffmpeg_capture(
            cmd,
            heartbeat=lambda idx=idx: status.heartbeat(
                progress={"step": idx - 1, "total": len(source_audio)},
                message=f"ffmpeg conversion still running ({idx}/{len(source_audio)})",
            ),
        )
        wav_inputs.append(converted_path)

        status.write(
            {
                "status": "preprocessing",
                "message": "Converting audio files to 24kHz mono WAV...",
                "progress": {"step": idx, "total": len(source_audio)},
            }
        )

    segments_dir = dataset_dir / "segments"
    if segments_dir.exists():
        shutil.rmtree(segments_dir)
    segments_dir.mkdir(parents=True, exist_ok=True)

    all_segments: list[Path] = []
    seg_index = 1
    status.write(
        {
            "status": "preprocessing",
            "message": "Running VAD segmentation...",
            "progress": {"step": 0, "total": len(wav_inputs)},
        }
    )
    for file_idx, wav_path in enumerate(wav_inputs, start=1):
        segments = _build_speech_segments(
            wav_path,
            heartbeat=lambda file_idx=file_idx: status.heartbeat(
                progress={"step": file_idx - 1, "total": len(wav_inputs)},
                message=f"VAD segmentation still running ({file_idx}/{len(wav_inputs)})",
            ),
        )
        for start_sec, dur_sec in segments:
            if dur_sec < 1.0:
                continue
            out_path = segments_dir / f"seg_{seg_index:06d}.wav"
            seg_index += 1
            cmd = [
                "ffmpeg",
                "-hide_banner",
                "-y",
                "-i",
                str(wav_path),
                "-ss",
                f"{start_sec:.3f}",
                "-t",
                f"{dur_sec:.3f}",
                "-ac",
                "1",
                "-ar",
                "24000",
                "-c:a",
                "pcm_s16le",
                str(out_path),
            ]
            _run_ffmpeg_capture(
                cmd,
                heartbeat=lambda file_idx=file_idx: status.heartbeat(
                    progress={"step": file_idx - 1, "total": len(wav_inputs)},
                    message=f"Segment export still running ({file_idx}/{len(wav_inputs)})",
                ),
            )
            all_segments.append(out_path)

        status.write(
            {
                "status": "preprocessing",
                "message": "Running VAD segmentation...",
                "progress": {"step": file_idx, "total": len(wav_inputs)},
            }
        )

    if not all_segments:
        raise RuntimeError("No valid speech segments created during preprocessing")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    LOGGER.info(
        "Loading faster-whisper model large-v3 on %s (%s)", device, compute_type
    )
    model = WhisperModel("large-v3", device=device, compute_type=compute_type)

    raw_rows: list[dict[str, Any]] = []
    ref_candidates: list[tuple[float, Path]] = []
    skipped_short = 0
    skipped_low_rms = 0
    skipped_low_conf = 0
    skipped_short_text = 0
    status.write(
        {
            "status": "preprocessing",
            "message": "Transcribing and filtering segments...",
            "progress": {"step": 0, "total": len(all_segments)},
        }
    )

    for idx, seg_path in enumerate(all_segments, start=1):
        duration = _probe_duration_seconds(seg_path)
        metrics = _compute_signal_metrics(seg_path)
        rms = metrics.get("rms", 0.0)
        if duration < MIN_SEGMENT_SEC or duration > MAX_SEGMENT_SEC:
            skipped_short += 1
            status.write(
                {
                    "status": "preprocessing",
                    "message": "Transcribing and filtering segments...",
                    "progress": {"step": idx, "total": len(all_segments)},
                }
            )
            continue
        if rms < 0.005:
            skipped_low_rms += 1
            status.write(
                {
                    "status": "preprocessing",
                    "message": "Transcribing and filtering segments...",
                    "progress": {"step": idx, "total": len(all_segments)},
                }
            )
            continue

        whisper_segments, _ = model.transcribe(
            str(seg_path),
            language=whisper_language,
            beam_size=5,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        text_parts: list[str] = []
        confidence_values: list[float] = []
        for wseg in whisper_segments:
            seg_text = str(wseg.text).strip()
            if seg_text:
                text_parts.append(seg_text)
                confidence_values.append(
                    max(0.0, min(1.0, math.exp(float(wseg.avg_logprob))))
                )

        text = " ".join(text_parts).strip()
        confidence = (
            sum(confidence_values) / len(confidence_values)
            if confidence_values
            else 0.0
        )

        if not text or confidence < MIN_CONFIDENCE:
            skipped_low_conf += 1
            status.write(
                {
                    "status": "preprocessing",
                    "message": "Transcribing and filtering segments...",
                    "progress": {"step": idx, "total": len(all_segments)},
                }
            )
            continue

        if len(_normalize_text(text)) < MIN_TRANSCRIPT_CHARS:
            skipped_short_text += 1
            status.write(
                {
                    "status": "preprocessing",
                    "message": "Transcribing and filtering segments...",
                    "progress": {"step": idx, "total": len(all_segments)},
                }
            )
            continue

        ref_duration_bonus = 1.0 - min(1.0, abs(duration - 6.0) / 4.0)
        ref_score = (
            (confidence * 0.55)
            + (ref_duration_bonus * 0.2)
            + (min(1.0, rms / 0.04) * 0.15)
            + ((1.0 - min(1.0, metrics.get("silence_ratio", 1.0))) * 0.1)
        )

        raw_rows.append(
            {
                "audio": str(seg_path),
                "text": text,
                "_duration": duration,
                "_confidence": confidence,
                "_metrics": metrics,
                "_norm_text": _normalize_text(text),
                "_ref_score": ref_score,
            }
        )

        if IDEAL_REF_MIN_SEC <= duration <= IDEAL_REF_MAX_SEC:
            ref_candidates.append((ref_score, seg_path))

        status.write(
            {
                "status": "preprocessing",
                "message": "Transcribing and filtering segments...",
                "progress": {"step": idx, "total": len(all_segments)},
            }
        )

    if not raw_rows:
        raise RuntimeError(
            "No usable segments left after transcription and quality filtering"
        )

    raw_rows.sort(
        key=lambda row: (
            -float(row.get("_confidence", 0.0)),
            -float(row.get("_duration", 0.0)),
            str(row.get("audio", "")),
        )
    )

    accepted_rows: list[dict[str, Any]] = []
    text_counts: dict[str, int] = {}
    duplicate_dropped = 0
    for row in raw_rows:
        norm = str(row.get("_norm_text", "")).strip()
        count = text_counts.get(norm, 0)
        if norm and count >= MAX_DUPLICATES_PER_TEXT:
            duplicate_dropped += 1
            continue
        if norm:
            text_counts[norm] = count + 1
        accepted_rows.append(row)

    total_duration_min = sum(float(row.get("_duration", 0.0)) for row in accepted_rows) / 60.0
    if len(accepted_rows) < MIN_ACCEPTED_SEGMENTS or total_duration_min < MIN_ACCEPTED_AUDIO_MIN:
        raise RuntimeError(
            "Dataset quality gate failed after preprocessing: "
            f"accepted_segments={len(accepted_rows)} (required >= {MIN_ACCEPTED_SEGMENTS}), "
            f"accepted_minutes={total_duration_min:.2f} (required >= {MIN_ACCEPTED_AUDIO_MIN:.2f}). "
            "Upload more clean speech from the same speaker before training."
        )

    if ref_candidates:
        ref_candidates.sort(key=lambda item: item[0], reverse=True)
        ref_audio_source = ref_candidates[0][1]
    else:
        accepted_rows.sort(key=lambda row: row["_confidence"], reverse=True)
        ref_audio_source = Path(str(accepted_rows[0]["audio"]))
    reference_text = next(
        (
            str(row.get("text", "")).strip()
            for row in accepted_rows
            if str(row.get("audio", "")).strip() == str(ref_audio_source)
        ),
        "",
    )

    ref_audio_path = dataset_dir / "ref_audio.wav"
    shutil.copyfile(ref_audio_source, ref_audio_path)
    LOGGER.info("Selected reference audio: %s", ref_audio_path)

    with output_jsonl.open("w", encoding="utf-8") as f:
        for row in accepted_rows:
            f.write(
                json.dumps(
                    {
                        "audio": row["audio"],
                        "text": row["text"],
                        "ref_audio": str(ref_audio_path),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    dataset_report = {
        "source_audio_files": len(source_audio),
        "segments_created": len(all_segments),
        "segments_accepted": len(accepted_rows),
        "accepted_duration_min": round(total_duration_min, 2),
        "avg_confidence": round(
            sum(float(row.get("_confidence", 0.0)) for row in accepted_rows)
            / max(len(accepted_rows), 1),
            4,
        ),
        "duplicate_dropped": duplicate_dropped,
        "skipped_short_or_long": skipped_short,
        "skipped_low_rms": skipped_low_rms,
        "skipped_low_confidence": skipped_low_conf,
        "skipped_short_text": skipped_short_text,
        "reference_audio": str(ref_audio_path),
        "reference_text": reference_text,
    }
    (dataset_dir / "preprocess_report.json").write_text(
        json.dumps(dataset_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    LOGGER.info(
        "Generated %s with %d samples (%d raw files -> %d segments)",
        output_jsonl,
        len(accepted_rows),
        len(source_audio),
        len(all_segments),
    )
    LOGGER.info("Preprocess report: %s", json.dumps(dataset_report, ensure_ascii=False))
    return dataset_report


def restore_preprocess_cache(
    *,
    r2: R2Storage,
    cache_prefix: str,
    dataset_dir: Path,
    status: StatusWriter,
) -> bool:
    cache_prefix = cache_prefix.strip().rstrip("/")
    if not cache_prefix:
        return False

    objects = r2.list_prefix(f"{cache_prefix}/")
    if not objects:
        LOGGER.info("No preprocess cache objects found under %s", cache_prefix)
        return False

    status.write(
        {
            "status": "preprocessing",
            "message": "Restoring cached transcripts and segments...",
            "progress": {"step": 0, "total": 1},
        }
    )
    r2.download_prefix(cache_prefix, dataset_dir)
    restored = (dataset_dir / "train_raw.jsonl").exists()
    status.write(
        {
            "status": "preprocessing",
            "message": (
                "Restored cached transcripts and segments"
                if restored
                else "Cached preprocess artifacts incomplete; rebuilding transcripts"
            ),
            "progress": {"step": 1, "total": 1},
        }
    )
    return restored


def upload_preprocess_cache(
    *,
    r2: R2Storage,
    cfg: JobConfig,
    dataset_dir: Path,
    preprocess_report: dict[str, Any] | None,
    reporter: WorkerReporter | None,
) -> str | None:
    dataset_signature = (cfg.dataset_signature or "").strip()
    if not dataset_signature:
      return None

    cache_prefix = f"{PREFIX_PREPROCESS_CACHE}/{cfg.voice_id}/{dataset_signature}"
    train_raw_jsonl = dataset_dir / "train_raw.jsonl"
    if not train_raw_jsonl.exists():
        return None

    r2.upload_file(train_raw_jsonl, f"{cache_prefix}/train_raw.jsonl", content_type="application/jsonl")
    ref_audio_path = dataset_dir / "ref_audio.wav"
    ref_audio_key: str | None = None
    if ref_audio_path.exists():
        ref_audio_key = f"{cache_prefix}/ref_audio.wav"
        r2.upload_file(ref_audio_path, ref_audio_key, content_type="audio/wav")

    reference_profile_key: str | None = None
    reference_profile_path = dataset_dir / "reference_profile.json"
    if reference_profile_path.exists():
        reference_profile_key = f"{cache_prefix}/reference_profile.json"
        r2.upload_file(reference_profile_path, reference_profile_key, content_type="application/json")

    preprocess_report_path = dataset_dir / "preprocess_report.json"
    if preprocess_report_path.exists():
        r2.upload_file(
            preprocess_report_path,
            f"{cache_prefix}/preprocess_report.json",
            content_type="application/json",
        )

    segments_dir = dataset_dir / "segments"
    if segments_dir.exists():
        for seg_path in segments_dir.rglob("*"):
            if seg_path.is_file():
                relative = seg_path.relative_to(dataset_dir)
                r2.upload_file(seg_path, f"{cache_prefix}/{relative}", content_type="audio/wav")

    if reporter is not None:
        reporter.report_preprocess_cache(
            dataset_signature=dataset_signature,
            cache_r2_prefix=cache_prefix,
            train_raw_r2_key=f"{cache_prefix}/train_raw.jsonl",
            ref_audio_r2_key=ref_audio_key,
            reference_profile_r2_key=reference_profile_key,
            source_file_count=(
                int(preprocess_report.get("source_audio_files"))
                if preprocess_report and preprocess_report.get("source_audio_files") is not None
                else None
            ),
            segments_created=(
                int(preprocess_report.get("segments_created"))
                if preprocess_report and preprocess_report.get("segments_created") is not None
                else None
            ),
            segments_accepted=(
                int(preprocess_report.get("segments_accepted"))
                if preprocess_report and preprocess_report.get("segments_accepted") is not None
                else None
            ),
            accepted_duration_min=(
                float(preprocess_report.get("accepted_duration_min"))
                if preprocess_report and preprocess_report.get("accepted_duration_min") is not None
                else None
            ),
        )

    LOGGER.info("Uploaded preprocess cache to %s", cache_prefix)
    return cache_prefix


def run_prepare(
    *,
    finetune_dir: Path,
    base_model: str,
    tokenizer_model: str,
    input_jsonl: Path,
    output_jsonl: Path,
    status: StatusWriter,
    log_buffer: LogBuffer,
) -> None:
    total = max(1, count_jsonl_rows(input_jsonl))
    status.write(
        {
            "status": "preparing",
            "message": "Extracting audio codes...",
            "progress": {"step": 0, "total": total},
        }
    )
    command = [
        "python3",
        str((finetune_dir / "prepare_data.py").resolve()),
        "--device",
        "cuda:0",
        "--tokenizer_model_path",
        tokenizer_model,
        "--input_jsonl",
        str(input_jsonl),
        "--output_jsonl",
        str(output_jsonl),
        "--batch_infer_num",
        "16",
    ]

    def on_line(line: str) -> None:
        log_buffer.append(line)
        m = PREPARE_PROGRESS_RE.search(line)
        if not m:
            return
        step = min(total, int(m.group(1)))
        status.write(
            {
                "status": "preparing",
                "message": "Extracting audio codes...",
                "progress": {"step": step, "total": total},
            }
        )

    rc = stream_subprocess(
        command,
        cwd=finetune_dir,
        callback=on_line,
        heartbeat=lambda: status.heartbeat(message="prepare_data.py still running"),
    )
    if rc != 0:
        raise RuntimeError(f"prepare_data.py failed with exit code {rc}")
    status.write(
        {
            "status": "preparing",
            "message": "Extracting audio codes...",
            "progress": {"step": total, "total": total},
        }
    )


def run_training(
    *,
    finetune_dir: Path,
    base_model: str,
    data_jsonl: Path,
    output_dir: Path,
    cfg: JobConfig,
    status: StatusWriter,
    log_buffer: LogBuffer,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    status.write(
        {
            "status": "training",
            "message": "Training started",
            "progress": {
                "epoch": 0,
                "step": 0,
                "loss": 0.0,
                "total_epochs": cfg.num_epochs,
            },
        }
    )
    command = [
        "python3",
        str((finetune_dir / "sft_12hz.py").resolve()),
        "--init_model_path",
        base_model,
        "--train_jsonl",
        str(data_jsonl),
        "--output_model_path",
        str(output_dir),
        "--speaker_name",
        cfg.speaker_name,
        "--batch_size",
        str(cfg.batch_size),
        "--lr",
        str(cfg.learning_rate),
        "--num_epochs",
        str(cfg.num_epochs),
        "--gradient_accumulation_steps",
        str(cfg.gradient_accumulation_steps),
        "--speaker_id",
        str(cfg.speaker_id),
        "--mixed_precision",
        cfg.mixed_precision,
        "--torch_dtype",
        cfg.torch_dtype,
        "--attn_implementation",
        cfg.attn_implementation,
        "--weight_decay",
        str(cfg.weight_decay),
        "--max_grad_norm",
        str(cfg.max_grad_norm),
        "--subtalker_loss_weight",
        str(cfg.subtalker_loss_weight),
        "--log_every_n_steps",
        str(cfg.log_every_n_steps),
        "--save_every_n_epochs",
        str(cfg.save_every_n_epochs),
        "--max_steps",
        str(cfg.max_steps),
        "--seed",
        str(cfg.seed),
    ]

    def on_line(line: str) -> None:
        log_buffer.append(line)
        m = TRAINING_PROGRESS_RE.search(line)
        if not m:
            return
        status.write(
            {
                "status": "training",
                "progress": {
                    "epoch": int(m.group(1)) + 1,
                    "step": int(m.group(2)) + 1,
                    "loss": float(m.group(3)),
                    "total_epochs": cfg.num_epochs,
                },
            }
        )

    rc = stream_subprocess(
        command,
        cwd=finetune_dir,
        callback=on_line,
        heartbeat=lambda: status.heartbeat(message="sft_12hz.py still running"),
    )
    if rc != 0:
        raise RuntimeError(f"sft_12hz.py failed with exit code {rc}")


def find_checkpoints(output_dir: Path) -> list[tuple[int, Path]]:
    checkpoints: list[tuple[int, Path]] = []
    for path in output_dir.glob("checkpoint-epoch-*"):
        if not path.is_dir():
            continue
        m = CHECKPOINT_DIR_RE.search(path.name)
        if m:
            checkpoints.append((int(m.group(1)), path))
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints


def upload_checkpoints(
    *, r2: R2Storage, cfg: JobConfig, output_dir: Path, status: StatusWriter
) -> list[dict[str, Any]]:
    """Upload all checkpoints to R2 using full recursive upload.

    Uses upload_checkpoint_full() (NOT delta) because the inference Docker
    image is 'lite' — no base models baked in. The inference handler uses
    checkpoint_type='full' and calls from_pretrained() directly on the
    downloaded directory, so it needs ALL files including speech_tokenizer/.
    """
    status.write({"status": "uploading", "message": "Uploading checkpoints to R2..."})
    checkpoints = find_checkpoints(output_dir)
    if not checkpoints:
        raise RuntimeError(f"No checkpoint-epoch-* directories found in {output_dir}")

    if len(checkpoints) <= MAX_UPLOADED_CHECKPOINTS:
        selected_checkpoints = checkpoints
    else:
        selected_indexes: set[int] = set()
        for idx in range(MAX_UPLOADED_CHECKPOINTS):
            selected_indexes.add(
                round((idx * (len(checkpoints) - 1)) / max(1, MAX_UPLOADED_CHECKPOINTS - 1))
            )
        selected_checkpoints = [checkpoints[index] for index in sorted(selected_indexes)]

    selected_epochs = {epoch for epoch, _ in selected_checkpoints}
    skipped_epochs = [epoch for epoch, _ in checkpoints if epoch not in selected_epochs]
    if skipped_epochs:
        LOGGER.info(
            "Skipping upload of %d checkpoint(s) after spread selection: %s",
            len(skipped_epochs),
            skipped_epochs,
        )
    uploaded: list[dict[str, Any]] = []
    total_to_upload = len(selected_checkpoints)
    for index, (epoch, ckpt_dir) in enumerate(selected_checkpoints, start=1):
        progress = {
            "epoch": epoch,
            "total_epochs": cfg.num_epochs,
            "step": index,
            "total_steps": total_to_upload,
        }
        status.write(
            {
                "status": "uploading",
                "message": f"Uploading checkpoint {index}/{total_to_upload} (epoch {epoch}) to R2...",
                "progress": progress,
                "checkpoints": uploaded,
            }
        )
        prefix = r2.upload_checkpoint_full(
            checkpoint_dir=ckpt_dir,
            voice_id=cfg.voice_id,
            run_name=cfg.run_name,
            epoch=epoch,
        )
        status.heartbeat(progress=progress, message=f"Verifying checkpoint epoch {epoch}")
        _verify_checkpoint_upload(r2, prefix, ckpt_dir)
        uploaded.append({"epoch": epoch, "r2_prefix": prefix})
        status.write(
            {
                "status": "uploading",
                "message": f"Uploaded checkpoint {index}/{total_to_upload} (epoch {epoch})",
                "progress": progress,
                "checkpoints": uploaded,
            }
        )
        LOGGER.info("Checkpoint epoch %d uploaded and verified: %s", epoch, prefix)
    return uploaded


def _verify_checkpoint_upload(r2: R2Storage, r2_prefix: str, local_dir: Path) -> None:
    """Verify uploaded checkpoint is complete by comparing local vs R2 file counts."""
    local_files: dict[str, int] = {}
    for fpath in local_dir.rglob("*"):
        if fpath.is_file():
            rel = str(fpath.relative_to(local_dir))
            local_files[rel] = fpath.stat().st_size

    r2_objects = r2.list_prefix(f"{r2_prefix}/")
    r2_files: dict[str, int] = {}
    for obj in r2_objects:
        rel = obj["key"][len(r2_prefix) + 1 :]
        if rel:
            r2_files[rel] = obj["size"]

    # Check required files exist
    required = ["model.safetensors", "config.json"]
    missing_required = [f for f in required if f not in r2_files]
    if missing_required:
        raise RuntimeError(
            f"Checkpoint verification FAILED: required files missing on R2: {missing_required}"
        )

    # Check all local files were uploaded
    missing = set(local_files.keys()) - set(r2_files.keys())
    if missing:
        raise RuntimeError(
            f"Checkpoint verification FAILED: {len(missing)} file(s) missing on R2: "
            + ", ".join(sorted(missing)[:10])
        )

    # Check sizes match (catches partial uploads)
    size_mismatches: list[str] = []
    for rel_path, local_size in local_files.items():
        r2_size = r2_files.get(rel_path, -1)
        if r2_size != local_size:
            size_mismatches.append(f"{rel_path}: local={local_size} r2={r2_size}")
    if size_mismatches:
        raise RuntimeError(
            f"Checkpoint verification FAILED: {len(size_mismatches)} file(s) size mismatch: "
            + ", ".join(size_mismatches[:5])
        )

    LOGGER.info(
        "Checkpoint verified: %d/%d files match on R2 (%s)",
        len(r2_files),
        len(local_files),
        r2_prefix,
    )


def terminate_pod() -> None:
    pod_id = os.environ.get("RUNPOD_POD_ID", "").strip()
    api_key = os.environ.get("RUNPOD_API_KEY", "").strip()
    if not pod_id or not api_key:
        LOGGER.warning("Skipping termination: RUNPOD_POD_ID or RUNPOD_API_KEY missing")
        return
    query = f'mutation {{ podTerminate(input: {{podId: "{pod_id}"}}) }}'
    try:
        requests_module = importlib.import_module("requests")
        resp = requests_module.post(
            "https://api.runpod.io/graphql",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"query": query},
            timeout=30,
        )
        LOGGER.info(
            "Terminate response: status=%s body=%s", resp.status_code, resp.text
        )
    except Exception as exc:
        LOGGER.exception("Pod termination failed: %s", exc)


def main() -> int:
    setup_logging()
    LOGGER.info("RunPod training handler started")
    log_buffer: LogBuffer | None = None
    upload_verified = False
    training_started = False
    # --- Early init: R2 + status must be inside try/except so crashes are observable ---
    try:
        job_id = required_env(JOB_ID_ENV)
        r2 = R2Storage()
        status = StatusWriter(r2, job_id)
        status.write({"status": "starting", "message": "Handler initializing..."})
    except Exception as exc:
        LOGGER.exception(
            "Fatal: handler failed before status could be established: %s", exc
        )
        return 1
    try:
        raw_cfg = r2.read_job_config(job_id)
        if raw_cfg is None:
            raise UnrecoverableError(f"Job config not found for job_id={job_id}")
        try:
            cfg = parse_job_config(raw_cfg)
        except (ValueError, KeyError, TypeError) as exc:
            raise UnrecoverableError(f"Invalid job config: {exc}") from exc

        worker_api_url = (
            os.environ.get(WORKER_API_URL_ENV, "").strip() or cfg.worker_api_url or ""
        )
        job_token = os.environ.get(JOB_TOKEN_ENV, "").strip() or cfg.job_token or ""
        reporter: WorkerReporter | None = None
        if worker_api_url and job_token:
            reporter = WorkerReporter(worker_api_url, job_id, job_token)
            status.reporter = reporter
        else:
            LOGGER.warning(
                "Worker reporter disabled (WORKER_API_URL/JOB_TOKEN missing): url=%s token_present=%s",
                bool(worker_api_url),
                bool(job_token),
            )
        log_buffer = LogBuffer(r2, job_id, reporter)

        try:
            dataset_voice_id, dataset_name = parse_dataset_prefix(cfg.dataset_r2_prefix)
        except ValueError as exc:
            raise UnrecoverableError(f"Invalid dataset_r2_prefix: {exc}") from exc
        if dataset_voice_id != cfg.voice_id:
            LOGGER.warning(
                "voice_id mismatch: config=%s dataset=%s (using config voice_id for uploads)",
                cfg.voice_id,
                dataset_voice_id,
            )

        status.write(
            {"status": "downloading", "message": "Downloading dataset from R2..."}
        )
        status.heartbeat(message="Starting dataset download")
        if DATASET_DIR.exists():
            shutil.rmtree(DATASET_DIR)
        DATASET_DIR.mkdir(parents=True, exist_ok=True)
        try:
            r2.download_dataset(dataset_voice_id, dataset_name, DATASET_DIR)
        except Exception as exc:
            raise UnrecoverableError(f"Dataset download failed: {exc}") from exc

        train_raw_jsonl = DATASET_DIR / "train_raw.jsonl"
        generated_train_raw = False
        preprocess_report: dict[str, Any] | None = None
        if not train_raw_jsonl.exists():
            cache_restored = False
            if cfg.preprocess_cache_r2_prefix:
                try:
                    cache_restored = restore_preprocess_cache(
                        r2=r2,
                        cache_prefix=cfg.preprocess_cache_r2_prefix,
                        dataset_dir=DATASET_DIR,
                        status=status,
                    )
                except Exception as exc:
                    LOGGER.warning("Failed to restore preprocess cache: %s", exc)
            status.write(
                {
                    "status": "preprocessing",
                    "message": "Segmenting and transcribing audio...",
                }
            )
            if not cache_restored or not train_raw_jsonl.exists():
                try:
                    preprocess_report = preprocess_raw_audio(
                        DATASET_DIR,
                        train_raw_jsonl,
                        status,
                        whisper_language=cfg.whisper_language,
                    )
                except (FileNotFoundError, RuntimeError) as exc:
                    raise UnrecoverableError(f"Preprocessing failed: {exc}") from exc
                generated_train_raw = True
                try:
                    ref_audio_local = DATASET_DIR / "ref_audio.wav"
                    if ref_audio_local.exists():
                        r2.upload_file(
                            ref_audio_local,
                            f"{cfg.dataset_r2_prefix}/ref_audio.wav",
                            content_type="audio/wav",
                        )
                    if preprocess_report is not None:
                        ref_meta = dict(preprocess_report)
                        ref_meta["reference_audio_key"] = f"{cfg.dataset_r2_prefix}/ref_audio.wav"
                        r2.upload_json(ref_meta, f"{cfg.dataset_r2_prefix}/reference_profile.json")
                except Exception as exc:
                    LOGGER.warning("Failed to upload generated reference profile: %s", exc)
                try:
                    upload_preprocess_cache(
                        r2=r2,
                        cfg=cfg,
                        dataset_dir=DATASET_DIR,
                        preprocess_report=preprocess_report,
                        reporter=reporter,
                    )
                except Exception as exc:
                    LOGGER.warning("Failed to upload preprocess cache: %s", exc)

        # Rewrite JSONL paths to match downloaded file locations
        if not generated_train_raw:
            rewrite_jsonl_paths(train_raw_jsonl, DATASET_DIR)

        # Validate dataset integrity (all files exist, required keys present)
        status.write({"status": "validating", "message": "Validating dataset..."})
        try:
            validate_dataset(train_raw_jsonl)
        except ValueError as exc:
            raise UnrecoverableError(f"Dataset validation failed: {exc}") from exc

        # Ensure all audio is 24kHz (resample if needed)
        validate_audio_format(train_raw_jsonl)

        status.heartbeat(message="Resolving base model and tokenizer")
        try:
            base_model = resolve_base_model(cfg.model_size)
            tokenizer_model = resolve_tokenizer_model()
        except Exception as exc:
            raise UnrecoverableError(f"Model resolution failed: {exc}") from exc
        finetune_dir = resolve_finetune_dir()
        train_with_codes = DATASET_DIR / "train_with_codes.jsonl"
        run_prepare(
            finetune_dir=finetune_dir,
            base_model=base_model,
            tokenizer_model=tokenizer_model,
            input_jsonl=train_raw_jsonl,
            output_jsonl=train_with_codes,
            status=status,
            log_buffer=log_buffer,
        )

        output_dir = OUTPUT_ROOT / cfg.run_name
        if output_dir.exists():
            LOGGER.warning("Removing existing output directory: %s", output_dir)
            shutil.rmtree(output_dir)
        training_started = True
        run_training(
            finetune_dir=finetune_dir,
            base_model=base_model,
            data_jsonl=train_with_codes,
            output_dir=output_dir,
            cfg=cfg,
            status=status,
            log_buffer=log_buffer,
        )

        checkpoints = upload_checkpoints(
            r2=r2, cfg=cfg, output_dir=output_dir, status=status
        )
        upload_verified = True
        status.write({"status": "completed", "checkpoints": checkpoints})
        LOGGER.info("Job completed successfully — checkpoint uploaded and verified")
        return 0
    except UnrecoverableError as exc:
        LOGGER.error("Unrecoverable error (pod will terminate): %s", exc)
        try:
            status.write({"status": "failed", "error": str(exc)})
        except Exception:
            LOGGER.exception("Failed to write failed status")
        return 1
    except Exception as exc:
        LOGGER.exception("Job failed: %s", exc)
        try:
            status.write({"status": "failed", "error": str(exc)})
        except Exception:
            LOGGER.exception("Failed to write failed status")
        return 1
    finally:
        if log_buffer is not None:
            try:
                log_buffer.close()
            except Exception:
                LOGGER.exception("Failed to flush log buffer")
        if upload_verified:
            terminate_pod()
        elif not training_started:
            LOGGER.info(
                "Pre-training failure — no checkpoints to recover. Terminating pod."
            )
            terminate_pod()
        else:
            LOGGER.warning(
                "SKIPPING pod termination — training started but checkpoint upload was NOT verified. "
                "Pod will remain alive for manual recovery. "
                "Use RunPod dashboard or API to terminate manually after recovery."
            )


if __name__ == "__main__":
    sys.exit(main())
