#!/usr/bin/env bash
# Container entrypoint: ensures a model artifact exists, then starts Uvicorn.
#
# With MODEL_STORE=local and no models present (fresh clone), we train a small
# offline breast-cancer demo model so `docker compose up` is fully self-contained.
# Set DEMO_DATASET=adult to train the UCI model instead (needs network on first run).
set -euo pipefail

if [ "${MODEL_STORE:-local}" = "local" ] && [ -z "$(ls -A /app/models 2>/dev/null)" ]; then
  echo "[entrypoint] No model found - training a ${DEMO_DATASET:-breast_cancer} demo model..."
  python -m pipeline.train --dataset "${DEMO_DATASET:-breast_cancer}"
fi

exec uvicorn app.main:app --host "${API_HOST:-0.0.0.0}" --port "${API_PORT:-8000}"
