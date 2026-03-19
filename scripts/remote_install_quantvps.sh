#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${1:-/opt/autoresearch-trading-bot}"
RUN_USER="${RUN_USER:-root}"
RUN_HOME="$(eval echo "~${RUN_USER}")"
ENV_FILE="${ENV_FILE:-/etc/autoresearch-trading.env}"
GIT_USER_NAME="${GIT_USER_NAME:-Autoresearch Bot}"
GIT_USER_EMAIL="${GIT_USER_EMAIL:-autoresearch-bot@local}"

export DEBIAN_FRONTEND=noninteractive

install_talib_system() {
  start_dir="$(pwd)"
  if ldconfig -p 2>/dev/null | grep -qi "libta-lib"; then
    return
  fi

  if apt-cache show libta-lib-dev >/dev/null 2>&1; then
    apt-get install -y libta-lib-dev
    return
  fi

  workdir="$(mktemp -d)"
  trap 'rm -rf "${workdir}"' RETURN
  curl -fsSL \
    -o "${workdir}/ta-lib.tar.gz" \
    https://github.com/TA-Lib/ta-lib/releases/download/v0.6.4/ta-lib-0.6.4-src.tar.gz
  tar -xzf "${workdir}/ta-lib.tar.gz" -C "${workdir}"
  cd "${workdir}/ta-lib-0.6.4"
  ./configure --prefix=/usr
  make -j"$(nproc)"
  make install
  ldconfig
  cd "${start_dir}"
}

apt-get update
apt-get install -y \
  python3-dev \
  build-essential \
  git \
  curl \
  ca-certificates \
  pkg-config \
  python3 \
  python3-venv \
  python3-pip \
  rsync \
  tar

install_talib_system

if ! command -v uv >/dev/null 2>&1; then
  su - "${RUN_USER}" -c "curl -LsSf https://astral.sh/uv/install.sh | sh"
fi

UV_BIN="${RUN_HOME}/.local/bin/uv"
if [[ ! -x "${UV_BIN}" ]]; then
  echo "uv not found at ${UV_BIN}"
  exit 1
fi

mkdir -p "${APP_DIR}"/{data,logs,metrics,tests}
chown -R "${RUN_USER}":"${RUN_USER}" "${APP_DIR}"

cd "${APP_DIR}"

if [[ ! -d .git ]]; then
  git init -b main
fi

git config user.name "${GIT_USER_NAME}"
git config user.email "${GIT_USER_EMAIL}"

if ! git rev-parse --verify HEAD >/dev/null 2>&1; then
  git add .gitignore README.md prepare.py program.md pyproject.toml results.tsv run_trading_agent.py setup-trading-autoresearch.sh strategy.py tests scripts metrics/.gitkeep || true
  git commit -m "bootstrap remote repo" || true
fi

if [[ ! -x .venv/bin/pip ]]; then
  rm -rf .venv
  su - "${RUN_USER}" -c "cd '${APP_DIR}' && '${UV_BIN}' venv --seed .venv"
fi
su - "${RUN_USER}" -c "cd '${APP_DIR}' && ./.venv/bin/python -m pip install --upgrade pip"
su - "${RUN_USER}" -c "cd '${APP_DIR}' && ./.venv/bin/pip install \
  yfinance vectorbt pandas numpy ccxt TA-Lib pyarrow pytest alpaca-py requests python-dotenv"

if [[ ! -f "${ENV_FILE}" ]]; then
  cat > "${ENV_FILE}" <<'EOF'
ANTHROPIC_API_KEY=YOUR_ANTHROPIC_API_KEY_HERE
OPENAI_API_KEY=YOUR_OPENAI_API_KEY_HERE
SLACK_WEBHOOK_URL=
SUMMARY_EMAIL_TO=
SUMMARY_EMAIL_FROM=
SMTP_HOST=
SMTP_PORT=587
SMTP_USER=
SMTP_PASS=
EOF
  chown root:root "${ENV_FILE}"
  chmod 600 "${ENV_FILE}"
fi

if [[ ! -f results.tsv ]]; then
  cat > results.tsv <<'EOF'
timestamp	provider	model	iteration	test_pass	input_tokens	output_tokens	est_cost_usd	sharpe	cagr	max_drawdown	objective	git_commit	phase	phase_reason	notes
EOF
  chown "${RUN_USER}":"${RUN_USER}" results.tsv
fi

su - "${RUN_USER}" -c "cd '${APP_DIR}' && PYTHONPATH='${APP_DIR}' ./.venv/bin/pytest tests/ -q"

cat > /etc/systemd/system/autoresearch-trading.service <<EOF
[Unit]
Description=Autonomous Autoresearch Trading Agent
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${RUN_USER}
WorkingDirectory=${APP_DIR}
EnvironmentFile=${ENV_FILE}
Environment=PYTHONPATH=${APP_DIR}
ExecStart=${APP_DIR}/.venv/bin/python ${APP_DIR}/run_trading_agent.py --loop-forever --provider auto --alpha-provider openai --alpha-model gpt-5.1-mini --protection-provider anthropic --protection-model claude-sonnet-4-20250514
Restart=always
RestartSec=10
TimeoutStopSec=30
StandardOutput=append:${APP_DIR}/logs/agent.out.log
StandardError=append:${APP_DIR}/logs/agent.err.log

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable autoresearch-trading.service
systemctl restart autoresearch-trading.service
systemctl --no-pager --full status autoresearch-trading.service || true
