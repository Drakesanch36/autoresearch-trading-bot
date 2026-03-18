from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from live_pilot_guardrails import (
    LivePilotConfig,
    LivePilotRuntimeStatus,
    build_live_pilot_readiness_report,
    format_live_pilot_readiness_report,
    load_live_pilot_config,
    summarize_offline_validation_history,
    summarize_paper_trading_history,
    write_live_pilot_readiness_report,
)


def make_offline_rows(now: datetime, *, iterations: int, committed: int, active_days: int) -> list[dict[str, str]]:
    rows = []
    for index in range(iterations):
        day = now - timedelta(days=(index % max(active_days, 1)))
        rows.append(
            {
                "timestamp": day.isoformat(),
                "iteration": str(index + 1),
                "notes": "committed" if index < committed else "no_improvement",
            }
        )
    return rows


def make_paper_reports(clean_days: int, total_days: int) -> list[dict]:
    reports = []
    for index in range(total_days):
        reports.append({"system_health_flags": {"overall_pass": index < clean_days}})
    return reports


def test_live_pilot_requires_explicit_human_enablement() -> None:
    now = datetime(2026, 3, 17, 12, 0, tzinfo=timezone.utc)
    config = LivePilotConfig(human_live_enable=False)
    report = build_live_pilot_readiness_report(
        config,
        make_offline_rows(now, iterations=40, committed=6, active_days=20),
        make_paper_reports(clean_days=12, total_days=12),
        LivePilotRuntimeStatus(
            broker_healthy=True,
            data_age_seconds=60.0,
            reconciliation_ok=True,
            daily_pnl=0.0,
            requested_position_notional=5_000.0,
            actual_position_notional=5_000.0,
        ),
        now=now,
    )

    assert report["checks"]["explicit_human_enablement"] is False
    assert report["live_mode_permitted"] is False


def test_live_pilot_readiness_passes_with_clean_history_and_runtime() -> None:
    now = datetime(2026, 3, 17, 12, 0, tzinfo=timezone.utc)
    config = LivePilotConfig(human_live_enable=True)
    report = build_live_pilot_readiness_report(
        config,
        make_offline_rows(now, iterations=40, committed=6, active_days=20),
        make_paper_reports(clean_days=12, total_days=12),
        LivePilotRuntimeStatus(
            broker_healthy=True,
            data_age_seconds=30.0,
            reconciliation_ok=True,
            daily_pnl=120.0,
            requested_position_notional=5_000.0,
            actual_position_notional=4_950.0,
        ),
        now=now,
    )

    assert report["live_mode_permitted"] is True
    assert report["kill_switch_active"] is False


def test_live_pilot_fails_with_insufficient_offline_or_paper_history() -> None:
    now = datetime(2026, 3, 17, 12, 0, tzinfo=timezone.utc)
    config = LivePilotConfig(human_live_enable=True)
    report = build_live_pilot_readiness_report(
        config,
        make_offline_rows(now, iterations=10, committed=1, active_days=3),
        make_paper_reports(clean_days=2, total_days=2),
        LivePilotRuntimeStatus(
            broker_healthy=True,
            data_age_seconds=30.0,
            reconciliation_ok=True,
            daily_pnl=0.0,
            requested_position_notional=1_000.0,
            actual_position_notional=1_000.0,
        ),
        now=now,
    )

    assert report["checks"]["minimum_offline_validation_history"] is False
    assert report["checks"]["minimum_clean_paper_trading_history"] is False
    assert report["live_mode_permitted"] is False


def test_live_pilot_kill_switch_trips_on_stale_data_or_reconciliation_failure() -> None:
    now = datetime(2026, 3, 17, 12, 0, tzinfo=timezone.utc)
    config = LivePilotConfig(human_live_enable=True, stale_data_max_seconds=120.0)
    stale_report = build_live_pilot_readiness_report(
        config,
        make_offline_rows(now, iterations=40, committed=6, active_days=20),
        make_paper_reports(clean_days=12, total_days=12),
        LivePilotRuntimeStatus(
            broker_healthy=True,
            data_age_seconds=999.0,
            reconciliation_ok=True,
            daily_pnl=0.0,
            requested_position_notional=1_000.0,
            actual_position_notional=1_000.0,
        ),
        now=now,
    )
    recon_report = build_live_pilot_readiness_report(
        config,
        make_offline_rows(now, iterations=40, committed=6, active_days=20),
        make_paper_reports(clean_days=12, total_days=12),
        LivePilotRuntimeStatus(
            broker_healthy=True,
            data_age_seconds=30.0,
            reconciliation_ok=False,
            daily_pnl=0.0,
            requested_position_notional=1_000.0,
            actual_position_notional=1_000.0,
        ),
        now=now,
    )

    assert stale_report["kill_switch_active"] is True
    assert stale_report["checks"]["stale_data_check_ok"] is False
    assert recon_report["kill_switch_active"] is True
    assert recon_report["checks"]["reconciliation_check_ok"] is False


def test_live_pilot_risk_caps_and_daily_loss_cap_are_enforced() -> None:
    now = datetime(2026, 3, 17, 12, 0, tzinfo=timezone.utc)
    config = LivePilotConfig(human_live_enable=True, max_daily_loss=250.0, max_position_notional=2_000.0)
    report = build_live_pilot_readiness_report(
        config,
        make_offline_rows(now, iterations=40, committed=6, active_days=20),
        make_paper_reports(clean_days=12, total_days=12),
        LivePilotRuntimeStatus(
            broker_healthy=True,
            data_age_seconds=30.0,
            reconciliation_ok=True,
            daily_pnl=-500.0,
            requested_position_notional=4_000.0,
            actual_position_notional=4_000.0,
        ),
        now=now,
    )

    assert report["checks"]["daily_loss_cap_ok"] is False
    assert report["checks"]["max_position_cap_ok"] is False
    assert report["kill_switch_active"] is True


def test_live_pilot_config_loader_and_report_writer(tmp_path: Path) -> None:
    config_path = tmp_path / "live_config.json"
    config_path.write_text(
        json.dumps(
            {
                "human_live_enable": True,
                "min_offline_iterations": 25,
                "min_offline_committed_improvements": 4,
                "min_offline_active_days": 10,
                "min_clean_paper_days": 8,
                "max_daily_loss": 200.0,
                "max_position_notional": 5_000.0,
                "stale_data_max_seconds": 120.0,
                "broker_health_required": True,
            }
        ),
        encoding="utf-8",
    )
    config = load_live_pilot_config(config_path)
    assert config.human_live_enable is True

    report = {
        "checks": {
            "explicit_human_enablement": True,
            "minimum_offline_validation_history": True,
            "minimum_clean_paper_trading_history": True,
            "risk_caps_within_limits": True,
            "daily_loss_cap_ok": True,
            "max_position_cap_ok": True,
            "broker_health_check_ok": True,
            "stale_data_check_ok": True,
            "reconciliation_check_ok": True,
        },
        "offline_history": {"completed_iterations": 30, "committed_improvements": 5, "active_days": 14},
        "paper_history": {"paper_days": 12, "clean_paper_days": 12},
        "runtime_status": {
            "requested_position_notional": 1000.0,
            "actual_position_notional": 900.0,
            "daily_pnl": 50.0,
            "data_age_seconds": 12.0,
        },
        "kill_switch_active": False,
        "live_mode_permitted": True,
    }
    json_path = tmp_path / "live_readiness.json"
    text_path = tmp_path / "live_readiness.txt"
    write_live_pilot_readiness_report(report, json_path=json_path, text_path=text_path)

    assert json.loads(json_path.read_text(encoding="utf-8"))["live_mode_permitted"] is True
    assert "Live Pilot Readiness Report" in text_path.read_text(encoding="utf-8")


def test_history_summaries_are_stable() -> None:
    now = datetime(2026, 3, 17, 12, 0, tzinfo=timezone.utc)
    offline = summarize_offline_validation_history(make_offline_rows(now, iterations=6, committed=2, active_days=3), now=now)
    paper = summarize_paper_trading_history(make_paper_reports(clean_days=2, total_days=4))
    assert offline == {"completed_iterations": 6, "committed_improvements": 2, "active_days": 3}
    assert paper == {"paper_days": 4, "clean_paper_days": 2}
    assert "Kill Switch Active:" in format_live_pilot_readiness_report(
        {
            "checks": {
                "explicit_human_enablement": False,
                "minimum_offline_validation_history": False,
                "minimum_clean_paper_trading_history": False,
                "risk_caps_within_limits": True,
                "daily_loss_cap_ok": True,
                "max_position_cap_ok": True,
                "broker_health_check_ok": True,
                "stale_data_check_ok": True,
                "reconciliation_check_ok": True,
            },
            "offline_history": offline,
            "paper_history": paper,
            "runtime_status": {
                "requested_position_notional": 0.0,
                "actual_position_notional": 0.0,
                "daily_pnl": 0.0,
                "data_age_seconds": 0.0,
            },
            "kill_switch_active": False,
            "live_mode_permitted": False,
        }
    )
