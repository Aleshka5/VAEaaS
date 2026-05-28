#!/usr/bin/env bash
set -euo pipefail

CMD=(
  uv run --env-file .env uvicorn api.fastapi_service:app
  --host 0.0.0.0
  --port 8000
  --log-level info
)

echo "Starting FastAPI on 0.0.0.0:8000 (log_level=info)"
exec "${CMD[@]}"
