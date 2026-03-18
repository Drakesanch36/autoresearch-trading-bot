#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-quantvps-autoresearch}"
REMOTE_DIR="${REMOTE_DIR:-/opt/autoresearch-trading-bot}"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

rsync -az --delete \
  --exclude '.git/' \
  --exclude '.venv/' \
  --exclude '__pycache__/' \
  --exclude '.pytest_cache/' \
  --exclude '.env' \
  --exclude '.env.quantvps' \
  --exclude 'data/' \
  --exclude 'logs/' \
  --exclude 'metrics/latest_metrics.json' \
  "${LOCAL_DIR}/" "${REMOTE_HOST}:${REMOTE_DIR}/"

if [[ -f "${LOCAL_DIR}/.env.quantvps" ]]; then
  scp "${LOCAL_DIR}/.env.quantvps" "${REMOTE_HOST}:${REMOTE_DIR}/.env"
fi

ssh "${REMOTE_HOST}" "bash '${REMOTE_DIR}/scripts/remote_install_quantvps.sh' '${REMOTE_DIR}'"
