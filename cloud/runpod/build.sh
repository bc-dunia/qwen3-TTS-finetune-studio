#!/bin/bash
# Build and push Docker images for RunPod
#
# Usage:
#   ./cloud/runpod/build.sh                    # Build both images
#   ./cloud/runpod/build.sh training           # Build training only
#   ./cloud/runpod/build.sh inference           # Build inference only
#   DOCKER_REGISTRY=myuser ./cloud/runpod/build.sh  # Custom registry
#
# Run from project root:
#   bash cloud/runpod/build.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

REGISTRY="${DOCKER_REGISTRY:-qwen3tts}"
TAG="${DOCKER_TAG:-latest}"
TARGET="${1:-all}"

cd "$PROJECT_ROOT"

build_training() {
    echo "=== Building training image ==="
    docker build \
        -f cloud/runpod/Dockerfile.training \
        -t "$REGISTRY/qwen3-tts-training:$TAG" \
        .
    echo "Built: $REGISTRY/qwen3-tts-training:$TAG"
}

build_inference() {
    echo "=== Building inference image ==="
    echo "NOTE: This downloads ~10GB of model weights. First build will be slow."
    docker build \
        -f cloud/runpod/Dockerfile.inference \
        -t "$REGISTRY/qwen3-tts-inference:$TAG" \
        .
    echo "Built: $REGISTRY/qwen3-tts-inference:$TAG"
}

push_images() {
    echo "=== Pushing images to registry ==="
    if [[ "$TARGET" == "all" || "$TARGET" == "training" ]]; then
        docker push "$REGISTRY/qwen3-tts-training:$TAG"
    fi
    if [[ "$TARGET" == "all" || "$TARGET" == "inference" ]]; then
        docker push "$REGISTRY/qwen3-tts-inference:$TAG"
    fi
}

case "$TARGET" in
    training)
        build_training
        ;;
    inference)
        build_inference
        ;;
    all)
        build_training
        build_inference
        ;;
    push)
        push_images
        ;;
    *)
        echo "Usage: $0 [training|inference|all|push]"
        exit 1
        ;;
esac

echo "Done."
