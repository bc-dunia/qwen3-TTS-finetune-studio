#!/usr/bin/env bash

set -u

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 LOG_FILE JOB_ID [JOB_ID ...]" >&2
  exit 1
fi

log_file=$1
shift

mkdir -p "$(dirname "$log_file")"
exec >>"$log_file" 2>&1

while true; do
  date "+=== %F %T %Z ==="

  for job_id in "$@"; do
    if ! curl -fsS "https://qwen-tts-api.brian-367.workers.dev/v1/training/$job_id" \
      | jq -c '{job_id,status,epoch:(.progress.epoch//null),total_epochs:(.progress.total_epochs//null),step:(.progress.step//null),loss:(.progress.loss//null),validation_checked:(.summary.validation_checked//null),validation_passed:(.summary.validation_passed//null),selected_checkpoint_epoch:(.summary.selected_checkpoint_epoch//null),msg:(.summary.validation_message // .summary.last_message // .error_message // null)}'
    then
      jq -nc --arg job_id "$job_id" --arg msg "status request failed" \
        '{job_id:$job_id,status:"monitor_error",msg:$msg}'
    fi
  done

  sleep 30
done
