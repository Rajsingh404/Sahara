#!/usr/bin/env bash
# SAHARA Phase 2-3 Full Pipeline Runner
# Runs: manifest → feature cache → train → evaluate
# Usage: bash scripts/run_pipeline.sh [--skip-manifest] [--skip-cache] [--skip-train] [--augment]
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
VENV="$REPO/.venv"
if [ -d "$VENV" ]; then
    source "$VENV/bin/activate"
fi

SKIP_MANIFEST=0; SKIP_CACHE=0; SKIP_TRAIN=0; AUGMENT=""
for arg in "$@"; do
    case $arg in
        --skip-manifest) SKIP_MANIFEST=1;;
        --skip-cache)    SKIP_CACHE=1;;
        --skip-train)    SKIP_TRAIN=1;;
        --augment)       AUGMENT="--augment";;
    esac
done

cd "$REPO"
echo "========================================"
echo "  SAHARA Pipeline — $(date)"
echo "========================================"

if [ "$SKIP_MANIFEST" -eq 0 ]; then
    echo ""
    echo "--- Step 1: Build manifests ---"
    python -m src.data.manifest_builder
fi

if [ "$SKIP_CACHE" -eq 0 ]; then
    echo ""
    echo "--- Step 2: Extract YAMNet embeddings ---"
    python -m src.preprocessing.build_features_cache
fi

if [ "$SKIP_TRAIN" -eq 0 ]; then
    echo ""
    echo "--- Step 3: Train classifier head ---"
    python -m src.training.train $AUGMENT
fi

echo ""
echo "--- Step 4: Evaluate + confusion matrix ---"
python -m src.evaluation.confusion_matrix

echo ""
echo "========================================"
echo "  Pipeline complete — $(date)"
echo "========================================"
