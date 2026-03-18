#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-quantvps-autoresearch}"
REMOTE_DIR="${REMOTE_DIR:-/opt/autoresearch-trading-bot}"

ssh "${REMOTE_HOST}" "tail -n 200 -f '${REMOTE_DIR}/logs/agent.out.log' '${REMOTE_DIR}/logs/agent.err.log'"
