from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
USAGE_HEADER = "\t".join(["input_tokens", "output_tokens", "est_cost_usd"])


def read_text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_setup_script_has_expected_safety_and_results_header() -> None:
    text = read_text("setup-trading-autoresearch.sh")
    assert text.startswith("#!/usr/bin/env bash")
    assert "set -euo pipefail" in text
    assert "EnvironmentFile=${ENV_FILE}" in text
    assert USAGE_HEADER in text
    assert "phase\tphase_reason\tnotes" in text
    assert "pytest tests/ -q" in text


def test_deploy_script_excludes_runtime_state_and_calls_remote_installer() -> None:
    text = read_text("scripts/deploy_quantvps.sh")
    assert text.startswith("#!/usr/bin/env bash")
    assert "--exclude '.git/'" in text
    assert "--exclude '.venv/'" in text
    assert "--exclude 'data/'" in text
    assert "remote_install_quantvps.sh" in text


def test_remote_install_script_bootstraps_git_and_systemd() -> None:
    text = read_text("scripts/remote_install_quantvps.sh")
    assert text.startswith("#!/usr/bin/env bash")
    assert "git init -b main" in text
    assert "systemctl enable autoresearch-trading.service" in text
    assert "EnvironmentFile=${ENV_FILE}" in text
    assert USAGE_HEADER in text
    assert "phase\tphase_reason\tnotes" in text


def test_tail_logs_script_targets_quantvps_logs() -> None:
    text = read_text("scripts/tail_quantvps_logs.sh")
    assert text.startswith("#!/usr/bin/env bash")
    assert "quantvps-autoresearch" in text
    assert "agent.out.log" in text
    assert "agent.err.log" in text
