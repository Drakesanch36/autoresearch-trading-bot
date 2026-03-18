#!/usr/bin/env bash
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run this script as root (sudo -i)."
  exit 1
fi

REPO_URL="${1:-https://github.com/YOUR_GITHUB_USERNAME/autoresearch-trading-bot.git}"
INSTALL_DIR="${2:-/opt/autoresearch-trading-bot}"
RUN_USER="${SUDO_USER:-root}"
RUN_HOME="$(eval echo ~${RUN_USER})"
ENV_FILE="${INSTALL_DIR}/.env"

echo "[1/9] Installing OS dependencies (includes TA-Lib system library first)..."
export DEBIAN_FRONTEND=noninteractive
apt update
apt install -y libta-lib-dev python3-dev build-essential
apt install -y git curl ca-certificates pkg-config python3 python3-venv python3-pip

if ! command -v uv >/dev/null 2>&1; then
  echo "[2/9] Installing uv..."
  su - "${RUN_USER}" -c "curl -LsSf https://astral.sh/uv/install.sh | sh"
fi

UV_BIN="${RUN_HOME}/.local/bin/uv"
if [[ ! -x "${UV_BIN}" ]]; then
  echo "uv not found at ${UV_BIN}."
  exit 1
fi

echo "[3/9] Cloning repository..."
if [[ -d "${INSTALL_DIR}/.git" ]]; then
  git -C "${INSTALL_DIR}" fetch --all --prune
  git -C "${INSTALL_DIR}" pull --ff-only
else
  git clone "${REPO_URL}" "${INSTALL_DIR}"
fi

chown -R "${RUN_USER}":"${RUN_USER}" "${INSTALL_DIR}"

cd "${INSTALL_DIR}"

echo "[4/9] Creating venv and installing Python dependencies..."
su - "${RUN_USER}" -c "cd '${INSTALL_DIR}' && '${UV_BIN}' venv .venv"
su - "${RUN_USER}" -c "cd '${INSTALL_DIR}' && ./.venv/bin/python -m pip install --upgrade pip"
su - "${RUN_USER}" -c "cd '${INSTALL_DIR}' && ./.venv/bin/pip install \
  yfinance vectorbt pandas numpy ccxt TA-Lib pyarrow pytest alpaca-py requests python-dotenv"

if [[ -f "pyproject.toml" ]]; then
  su - "${RUN_USER}" -c "cd '${INSTALL_DIR}' && ./.venv/bin/pip install -e ."
fi

echo "[5/9] Enforcing autoresearch 3-file philosophy (prepare.py, strategy.py, program.md)..."
if [[ -f train.py && ! -f strategy.py ]]; then
  mv train.py strategy.py
fi

mkdir -p tests metrics data logs

echo "[6/9] Creating .env template (never commit real secrets)..."
if [[ ! -f .env ]]; then
  cat > .env <<'EOT'
ANTHROPIC_API_KEY=YOUR_ANTHROPIC_API_KEY_HERE
OPENAI_API_KEY=YOUR_OPENAI_API_KEY_HERE
SLACK_WEBHOOK_URL=
SUMMARY_EMAIL_TO=
SUMMARY_EMAIL_FROM=
SMTP_HOST=
SMTP_PORT=587
SMTP_USER=
SMTP_PASS=
EOT
  chown "${RUN_USER}":"${RUN_USER}" .env
  chmod 600 .env
fi

if [[ ! -f results.tsv ]]; then
  cat > results.tsv <<'EOT'
timestamp	provider	model	iteration	test_pass	input_tokens	output_tokens	est_cost_usd	sharpe	cagr	max_drawdown	objective	git_commit	phase	phase_reason	notes
EOT
  chown "${RUN_USER}":"${RUN_USER}" results.tsv
fi

echo "[7/9] Running full test suite..."
su - "${RUN_USER}" -c "cd '${INSTALL_DIR}' && ./.venv/bin/pytest tests/ -q"

echo "[8/9] Creating systemd service for 24/7 autonomous loop..."
cat > /etc/systemd/system/autoresearch-trading.service <<EOT
[Unit]
Description=Autonomous Autoresearch Trading Agent
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${RUN_USER}
WorkingDirectory=${INSTALL_DIR}
EnvironmentFile=${ENV_FILE}
ExecStart=${INSTALL_DIR}/.venv/bin/python ${INSTALL_DIR}/run_trading_agent.py --loop-forever --provider auto --model claude-3-7-sonnet-latest --fallback-model gpt-4.1-mini
Restart=always
RestartSec=10
TimeoutStopSec=30
StandardOutput=append:${INSTALL_DIR}/logs/agent.out.log
StandardError=append:${INSTALL_DIR}/logs/agent.err.log

[Install]
WantedBy=multi-user.target
EOT

systemctl daemon-reload
systemctl enable autoresearch-trading.service
systemctl restart autoresearch-trading.service

echo "[9/9] Installation complete."
echo "Service status:"
systemctl --no-pager --full status autoresearch-trading.service || true
