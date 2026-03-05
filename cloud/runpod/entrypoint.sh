#!/bin/bash
# Entrypoint wrapper for RunPod training pods.
# Captures stdout/stderr and uploads diagnostic log to R2 on failure.
# NOTE: No "set -e" — we MUST reach the crash-log upload even if the handler fails.

LOG=/tmp/handler_output.log
JOB="${JOB_ID:-unknown}"

export HF_HUB_ENABLE_HF_TRANSFER=0

echo "=== ENTRYPOINT START $(date -u) ===" | tee "$LOG"
echo "Python: $(python3 --version 2>&1)" | tee -a "$LOG"
echo "JOB_ID: $JOB" | tee -a "$LOG"

# Run the handler, capturing all output.
# Do NOT use set -e/pipefail — we need to continue to the upload step.
python3 -u /app/training_handler.py 2>&1 | tee -a "$LOG"
EXIT_CODE=${PIPESTATUS[0]}

echo "=== HANDLER EXIT CODE: $EXIT_CODE ===" | tee -a "$LOG"

# ALWAYS upload the log to R2 for observability.
LOG_KEY="jobs/${JOB}/handler_log.txt"
if [ "$EXIT_CODE" -ne 0 ]; then
    LOG_KEY="jobs/${JOB}/crash_log.txt"
fi

echo "Uploading log to R2 key=$LOG_KEY ..." | tee -a "$LOG"
python3 -c "
import boto3, os, sys
from botocore.config import Config
try:
    s3 = boto3.client('s3',
        endpoint_url=os.environ.get('R2_ENDPOINT_URL', ''),
        aws_access_key_id=os.environ.get('R2_ACCESS_KEY_ID', ''),
        aws_secret_access_key=os.environ.get('R2_SECRET_ACCESS_KEY', ''),
        region_name='auto',
        config=Config(retries={'max_attempts': 3, 'mode': 'adaptive'})
    )
    with open('/tmp/handler_output.log', 'rb') as f:
        s3.put_object(Bucket=os.environ.get('R2_BUCKET', 'qwen-tts-studio'),
                      Key=sys.argv[1],
                      Body=f.read())
    print(f'Log uploaded to R2: {sys.argv[1]}')
except Exception as e:
    print(f'Failed to upload log: {e}')
" "$LOG_KEY" 2>&1 | tee -a "$LOG"

exit $EXIT_CODE
