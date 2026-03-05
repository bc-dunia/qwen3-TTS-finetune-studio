#!/bin/bash
# Entrypoint wrapper for RunPod training pods.
# Captures stdout/stderr and uploads diagnostic log to R2 on failure.
set -euo pipefail

LOG=/tmp/handler_output.log
JOB="${JOB_ID:-unknown}"

echo "=== ENTRYPOINT START $(date -u) ===" | tee "$LOG"
echo "Python: $(python3 --version 2>&1)" | tee -a "$LOG"
echo "JOB_ID: $JOB" | tee -a "$LOG"
echo "Image: $(cat /app/.image_sha 2>/dev/null || echo 'unknown')" | tee -a "$LOG"

# Run the handler, capturing all output
python3 -u /app/training_handler.py 2>&1 | tee -a "$LOG"
EXIT_CODE=${PIPESTATUS[0]}

echo "=== HANDLER EXIT CODE: $EXIT_CODE ===" | tee -a "$LOG"

# If handler failed, upload diagnostic log to R2
if [ "$EXIT_CODE" -ne 0 ]; then
    echo "Handler failed. Uploading diagnostic log to R2..."
    python3 -c "
import boto3, os
from botocore.config import Config
try:
    s3 = boto3.client('s3',
        endpoint_url=os.environ.get('R2_ENDPOINT_URL', ''),
        aws_access_key_id=os.environ.get('R2_ACCESS_KEY_ID', ''),
        aws_secret_access_key=os.environ.get('R2_SECRET_ACCESS_KEY', ''),
        region_name='auto',
        config=Config(retries={'max_attempts': 3, 'mode': 'adaptive'})
    )
    job_id = os.environ.get('JOB_ID', 'unknown')
    with open('/tmp/handler_output.log', 'rb') as f:
        s3.put_object(Bucket=os.environ.get('R2_BUCKET', 'qwen-tts-studio'),
                      Key=f'jobs/{job_id}/crash_log.txt',
                      Body=f.read())
    print('Crash log uploaded to R2')
except Exception as e:
    print(f'Failed to upload crash log: {e}')
" 2>&1 | tee -a "$LOG"
fi

exit $EXIT_CODE
