"""Cloudflare R2 storage client for RunPod handlers.

Uses boto3 with S3-compatible API to interact with Cloudflare R2.
Handles checkpoint upload/download, dataset management, and audio file storage.

Required environment variables:
    R2_ENDPOINT_URL:      https://<account_id>.r2.cloudflarestorage.com
    R2_ACCESS_KEY_ID:     S3-compatible access key
    R2_SECRET_ACCESS_KEY: S3-compatible secret key
    R2_BUCKET:            Bucket name (default: qwen-tts-studio)
"""

from __future__ import annotations

import json
import logging
import mimetypes
import os
from pathlib import Path
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError as BotoClientError

logger = logging.getLogger(__name__)

# Files that constitute a checkpoint "delta" — everything needed to reconstruct
# the fine-tuned model when combined with the base model.
CHECKPOINT_DELTA_FILES = [
    "model.safetensors",
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
]

# R2 bucket key prefixes
PREFIX_CHECKPOINTS = "checkpoints"
PREFIX_DATASETS = "datasets"
PREFIX_AUDIO = "audio"
PREFIX_JOBS = "jobs"


class R2Storage:
    """S3-compatible client for Cloudflare R2."""

    def __init__(
        self,
        endpoint_url: str | None = None,
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
        bucket: str | None = None,
    ) -> None:
        self.endpoint_url = endpoint_url or os.environ["R2_ENDPOINT_URL"]
        self.access_key_id = access_key_id or os.environ["R2_ACCESS_KEY_ID"]
        self.secret_access_key = secret_access_key or os.environ["R2_SECRET_ACCESS_KEY"]
        self.bucket = bucket or os.environ.get("R2_BUCKET", "qwen-tts-studio")

        self.client = boto3.client(
            "s3",
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
            region_name="auto",
            config=Config(
                retries={"max_attempts": 3, "mode": "adaptive"},
                max_pool_connections=20,
            ),
        )

    # ── Core operations ──────────────────────────────────────────────

    def upload_file(
        self,
        local_path: str | Path,
        r2_key: str,
        content_type: str | None = None,
    ) -> str:
        """Upload a local file to R2. Returns the R2 key."""
        local_path = Path(local_path)
        if content_type is None:
            content_type = (
                mimetypes.guess_type(str(local_path))[0] or "application/octet-stream"
            )

        extra = {"ContentType": content_type}
        file_size = local_path.stat().st_size

        # Use multipart for files > 100MB
        if file_size > 100 * 1024 * 1024:
            transfer_config = boto3.s3.transfer.TransferConfig(
                multipart_threshold=100 * 1024 * 1024,
                multipart_chunksize=100 * 1024 * 1024,
                max_concurrency=4,
            )
            self.client.upload_file(
                str(local_path),
                self.bucket,
                r2_key,
                ExtraArgs=extra,
                Config=transfer_config,
            )
        else:
            self.client.upload_file(
                str(local_path), self.bucket, r2_key, ExtraArgs=extra
            )

        logger.info(
            "Uploaded %s -> r2://%s/%s (%d bytes)",
            local_path,
            self.bucket,
            r2_key,
            file_size,
        )
        return r2_key

    def download_file(self, r2_key: str, local_path: str | Path) -> Path:
        """Download an R2 object to a local file. Returns the local path."""
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(self.bucket, r2_key, str(local_path))
        logger.info("Downloaded r2://%s/%s -> %s", self.bucket, r2_key, local_path)
        return local_path

    def upload_bytes(
        self, data: bytes, r2_key: str, content_type: str = "application/octet-stream"
    ) -> str:
        """Upload raw bytes to R2. Returns the R2 key."""
        self.client.put_object(
            Bucket=self.bucket, Key=r2_key, Body=data, ContentType=content_type
        )
        logger.info("Uploaded %d bytes -> r2://%s/%s", len(data), self.bucket, r2_key)
        return r2_key

    def download_bytes(self, r2_key: str) -> bytes:
        """Download an R2 object as bytes."""
        response = self.client.get_object(Bucket=self.bucket, Key=r2_key)
        return response["Body"].read()

    def upload_json(self, data: Any, r2_key: str) -> str:
        """Upload a JSON-serializable object to R2."""
        body = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        return self.upload_bytes(body, r2_key, content_type="application/json")

    def download_json(self, r2_key: str) -> Any:
        """Download and parse a JSON object from R2."""
        return json.loads(self.download_bytes(r2_key))

    def list_prefix(self, prefix: str) -> list[dict[str, Any]]:
        """List all objects under a prefix."""
        results: list[dict[str, Any]] = []
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                results.append(
                    {
                        "key": obj["Key"],
                        "size": obj["Size"],
                        "last_modified": obj["LastModified"].isoformat(),
                    }
                )
        return results

    def delete_key(self, r2_key: str) -> None:
        """Delete a single object from R2."""
        self.client.delete_object(Bucket=self.bucket, Key=r2_key)
        logger.info("Deleted r2://%s/%s", self.bucket, r2_key)

    def delete_prefix(self, prefix: str) -> int:
        """Delete all objects under a prefix. Returns count deleted."""
        objects = self.list_prefix(prefix)
        if not objects:
            return 0
        # Delete in batches of 1000 (S3 limit)
        deleted = 0
        for i in range(0, len(objects), 1000):
            batch = objects[i : i + 1000]
            self.client.delete_objects(
                Bucket=self.bucket,
                Delete={"Objects": [{"Key": obj["key"]} for obj in batch]},
            )
            deleted += len(batch)
        logger.info("Deleted %d objects under r2://%s/%s", deleted, self.bucket, prefix)
        return deleted

    def head_object(self, r2_key: str) -> dict[str, Any] | None:
        """Get object metadata. Returns None if not found."""
        try:
            resp = self.client.head_object(Bucket=self.bucket, Key=r2_key)
            return {
                "size": resp["ContentLength"],
                "content_type": resp.get("ContentType"),
                "last_modified": resp["LastModified"].isoformat(),
                "etag": resp.get("ETag"),
            }
        except BotoClientError as e:
            if e.response.get("Error", {}).get("Code") in {
                "404",
                "NotFound",
                "NoSuchKey",
            }:
                return None
            raise

    # ── Checkpoint operations ────────────────────────────────────────

    def upload_checkpoint_delta(
        self,
        checkpoint_dir: str | Path,
        voice_id: str,
        run_name: str,
        epoch: int,
    ) -> str:
        """Upload only the delta files from a checkpoint directory.

        Per Oracle recommendation: store only fine-tuned weights + configs,
        not the full base model copy. The base model is baked into the
        inference Docker image.

        Returns the R2 prefix for the uploaded checkpoint.
        """
        checkpoint_dir = Path(checkpoint_dir)
        prefix = f"{PREFIX_CHECKPOINTS}/{voice_id}/{run_name}/checkpoint-epoch-{epoch}"

        required = ["model.safetensors", "config.json"]
        missing = [f for f in required if not (checkpoint_dir / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"Checkpoint missing required files: {missing} in {checkpoint_dir}"
            )

        uploaded = 0
        for fname in CHECKPOINT_DELTA_FILES:
            fpath = checkpoint_dir / fname
            if fpath.exists():
                self.upload_file(fpath, f"{prefix}/{fname}")
                uploaded += 1

        # Also upload any additional config files we might have missed
        for fpath in checkpoint_dir.glob("*.json"):
            if fpath.name not in CHECKPOINT_DELTA_FILES:
                self.upload_file(fpath, f"{prefix}/{fpath.name}")
                uploaded += 1

        logger.info(
            "Uploaded checkpoint delta (%d files) -> r2://%s/%s",
            uploaded,
            self.bucket,
            prefix,
        )
        return prefix

    def upload_checkpoint_full(
        self,
        checkpoint_dir: str | Path,
        voice_id: str,
        run_name: str,
        epoch: int,
    ) -> str:
        """Upload the entire checkpoint directory (all files)."""
        checkpoint_dir = Path(checkpoint_dir)
        prefix = f"{PREFIX_CHECKPOINTS}/{voice_id}/{run_name}/checkpoint-epoch-{epoch}"

        uploaded = 0
        for fpath in checkpoint_dir.rglob("*"):
            if fpath.is_file():
                relative = fpath.relative_to(checkpoint_dir)
                self.upload_file(fpath, f"{prefix}/{relative}")
                uploaded += 1

        logger.info(
            "Uploaded full checkpoint (%d files) -> r2://%s/%s",
            uploaded,
            self.bucket,
            prefix,
        )
        return prefix

    def download_checkpoint(
        self,
        voice_id: str,
        run_name: str,
        epoch: int,
        local_dir: str | Path,
    ) -> Path:
        """Download a checkpoint from R2 to a local directory."""
        prefix = f"{PREFIX_CHECKPOINTS}/{voice_id}/{run_name}/checkpoint-epoch-{epoch}/"
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)

        objects = self.list_prefix(prefix)
        for obj in objects:
            relative = obj["key"][len(prefix) :]
            self.download_file(obj["key"], local_dir / relative)

        logger.info(
            "Downloaded checkpoint (%d files) -> %s",
            len(objects),
            local_dir,
        )
        return local_dir

    def list_checkpoints(self, voice_id: str) -> list[dict[str, Any]]:
        """List all checkpoints for a voice."""
        prefix = f"{PREFIX_CHECKPOINTS}/{voice_id}/"
        objects = self.list_prefix(prefix)

        # Parse unique checkpoint paths
        checkpoints: dict[str, dict[str, Any]] = {}
        for obj in objects:
            # Extract run_name/checkpoint-epoch-N from the key
            parts = obj["key"][len(prefix) :].split("/")
            if len(parts) >= 2:
                run_name = parts[0]
                ckpt_name = parts[1]
                ckpt_key = f"{run_name}/{ckpt_name}"
                if ckpt_key not in checkpoints:
                    checkpoints[ckpt_key] = {
                        "voice_id": voice_id,
                        "run_name": run_name,
                        "checkpoint": ckpt_name,
                        "prefix": f"{prefix}{ckpt_key}",
                        "files": [],
                        "total_size": 0,
                    }
                checkpoints[ckpt_key]["files"].append(obj["key"])
                checkpoints[ckpt_key]["total_size"] += obj["size"]

        return list(checkpoints.values())

    # ── Dataset operations ───────────────────────────────────────────

    def upload_dataset(
        self,
        local_dir: str | Path,
        voice_id: str,
        dataset_name: str,
    ) -> str:
        """Upload a dataset directory to R2."""
        local_dir = Path(local_dir)
        prefix = f"{PREFIX_DATASETS}/{voice_id}/{dataset_name}"

        uploaded = 0
        for fpath in local_dir.rglob("*"):
            if fpath.is_file():
                relative = fpath.relative_to(local_dir)
                ct = "audio/wav" if fpath.suffix == ".wav" else None
                self.upload_file(fpath, f"{prefix}/{relative}", content_type=ct)
                uploaded += 1

        logger.info(
            "Uploaded dataset (%d files) -> r2://%s/%s", uploaded, self.bucket, prefix
        )
        return prefix

    def download_dataset(
        self,
        voice_id: str,
        dataset_name: str,
        local_dir: str | Path,
    ) -> Path:
        """Download a dataset from R2 to a local directory."""
        prefix = f"{PREFIX_DATASETS}/{voice_id}/{dataset_name}/"
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)

        objects = self.list_prefix(prefix)
        for obj in objects:
            relative = obj["key"][len(prefix) :]
            self.download_file(obj["key"], local_dir / relative)

        logger.info("Downloaded dataset (%d files) -> %s", len(objects), local_dir)
        return local_dir

    # ── Audio operations ─────────────────────────────────────────────

    def upload_audio(
        self,
        local_path: str | Path,
        voice_id: str,
        filename: str,
    ) -> str:
        """Upload a generated audio file to R2. Returns the R2 key."""
        r2_key = f"{PREFIX_AUDIO}/{voice_id}/{filename}"
        self.upload_file(local_path, r2_key, content_type="audio/wav")
        return r2_key

    def upload_audio_bytes(
        self,
        audio_data: bytes,
        voice_id: str,
        filename: str,
        content_type: str = "audio/wav",
    ) -> str:
        """Upload audio bytes to R2. Returns the R2 key."""
        r2_key = f"{PREFIX_AUDIO}/{voice_id}/{filename}"
        return self.upload_bytes(audio_data, r2_key, content_type=content_type)

    def download_audio(
        self, voice_id: str, filename: str, local_path: str | Path
    ) -> Path:
        """Download an audio file from R2."""
        r2_key = f"{PREFIX_AUDIO}/{voice_id}/{filename}"
        return self.download_file(r2_key, local_path)

    # ── Job status operations ────────────────────────────────────────

    def write_job_status(self, job_id: str, status: dict[str, Any]) -> str:
        """Write job status JSON to R2."""
        r2_key = f"{PREFIX_JOBS}/{job_id}/status.json"
        return self.upload_json(status, r2_key)

    def read_job_status(self, job_id: str) -> dict[str, Any] | None:
        """Read job status from R2. Returns None if not found."""
        r2_key = f"{PREFIX_JOBS}/{job_id}/status.json"
        if self.head_object(r2_key) is None:
            return None
        return self.download_json(r2_key)

    def write_job_config(self, job_id: str, config: dict[str, Any]) -> str:
        """Write job config JSON to R2."""
        r2_key = f"{PREFIX_JOBS}/{job_id}/config.json"
        return self.upload_json(config, r2_key)

    def read_job_config(self, job_id: str) -> dict[str, Any] | None:
        """Read job config from R2. Returns None if not found."""
        r2_key = f"{PREFIX_JOBS}/{job_id}/config.json"
        if self.head_object(r2_key) is None:
            return None
        return self.download_json(r2_key)

    def append_job_log(self, job_id: str, log_chunk: str) -> str:
        """Append a log chunk to the job's log in R2.

        Logs are stored as numbered chunks to avoid read-modify-write.
        """
        existing = self.list_prefix(f"{PREFIX_JOBS}/{job_id}/logs/")
        chunk_num = len(existing)
        r2_key = f"{PREFIX_JOBS}/{job_id}/logs/chunk_{chunk_num:04d}.txt"
        return self.upload_bytes(log_chunk.encode("utf-8"), r2_key, "text/plain")
