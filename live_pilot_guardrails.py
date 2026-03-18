from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

LIVE_PILOT_READINESS_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class LivePilotConfig:
    human_live_enable: bool = False
    min_offline_iterations: int = 30
    min_offline_committed_improvements: int = 5
    min_offline_active_days: int = 14
    min_clean_paper_days: int = 10
    max_daily_loss: float = 500.0
    max_position_notional: float = 10_000.0
    stale_data_max_seconds: float = 300.0
    broker_health_required: bool = True


@dataclass(frozen=True)
class LivePilotRuntimeStatus:
    broker_healthy: bool
    data_age_seconds: float
    reconciliation_ok: bool
    daily_pnl: float
    requested_position_notional: float
    actual_position_notional: float


def load_live_pilot_config(path: Path) -> LivePilotConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return LivePilotConfig(**payload)


def _parse_timestamp(value: str) -> Optional[datetime]:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def summarize_offline_validation_history(
    rows: List[Dict[str, str]],
    *,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    if now is None:
        now = datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    completed_rows = [row for row in rows if str(row.get("iteration", "")).strip()]
    committed_rows = [row for row in completed_rows if str(row.get("notes", "")).strip() == "committed"]
    active_days = set()
    for row in completed_rows:
        timestamp = _parse_timestamp(str(row.get("timestamp", "")))
        if timestamp is not None:
            active_days.add(timestamp.date())

    return {
        "completed_iterations": len(completed_rows),
        "committed_improvements": len(committed_rows),
        "active_days": len(active_days),
    }


def summarize_paper_trading_history(reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    clean_reports = [report for report in reports if bool(report.get("system_health_flags", {}).get("overall_pass", False))]
    return {
        "paper_days": len(reports),
        "clean_paper_days": len(clean_reports),
    }


def build_live_pilot_readiness_report(
    config: LivePilotConfig,
    offline_rows: List[Dict[str, str]],
    paper_reports: List[Dict[str, Any]],
    runtime_status: LivePilotRuntimeStatus,
    *,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    offline = summarize_offline_validation_history(offline_rows, now=now)
    paper = summarize_paper_trading_history(paper_reports)

    checks = {
        "explicit_human_enablement": bool(config.human_live_enable),
        "minimum_offline_validation_history": bool(
            offline["completed_iterations"] >= config.min_offline_iterations
            and offline["committed_improvements"] >= config.min_offline_committed_improvements
            and offline["active_days"] >= config.min_offline_active_days
        ),
        "minimum_clean_paper_trading_history": bool(paper["clean_paper_days"] >= config.min_clean_paper_days),
        "risk_caps_within_limits": bool(
            abs(runtime_status.requested_position_notional) <= config.max_position_notional
            and abs(runtime_status.actual_position_notional) <= config.max_position_notional
        ),
        "daily_loss_cap_ok": bool(runtime_status.daily_pnl >= -abs(config.max_daily_loss)),
        "max_position_cap_ok": bool(
            abs(runtime_status.requested_position_notional) <= config.max_position_notional
            and abs(runtime_status.actual_position_notional) <= config.max_position_notional
        ),
        "broker_health_check_ok": bool((not config.broker_health_required) or runtime_status.broker_healthy),
        "stale_data_check_ok": bool(runtime_status.data_age_seconds <= config.stale_data_max_seconds),
        "reconciliation_check_ok": bool(runtime_status.reconciliation_ok),
    }
    kill_switch_active = not (
        checks["broker_health_check_ok"]
        and checks["stale_data_check_ok"]
        and checks["reconciliation_check_ok"]
        and checks["daily_loss_cap_ok"]
        and checks["max_position_cap_ok"]
    )
    live_mode_permitted = bool(all(checks.values()) and not kill_switch_active)

    return {
        "schema_version": LIVE_PILOT_READINESS_SCHEMA_VERSION,
        "generated_at": (now or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat(),
        "config": asdict(config),
        "offline_history": offline,
        "paper_history": paper,
        "runtime_status": asdict(runtime_status),
        "checks": checks,
        "kill_switch_active": kill_switch_active,
        "live_mode_permitted": live_mode_permitted,
    }


def format_live_pilot_readiness_report(report: Dict[str, Any]) -> str:
    checks = report["checks"]
    offline = report["offline_history"]
    paper = report["paper_history"]
    runtime = report["runtime_status"]
    return "\n".join(
        [
            "Live Pilot Readiness Report",
            f"Live Mode Permitted: {report['live_mode_permitted']}",
            f"Kill Switch Active: {report['kill_switch_active']}",
            f"Explicit Human Enablement: {checks['explicit_human_enablement']}",
            f"Offline History: iterations={offline['completed_iterations']}, committed={offline['committed_improvements']}, active_days={offline['active_days']}",
            f"Paper History: paper_days={paper['paper_days']}, clean_paper_days={paper['clean_paper_days']}",
            f"Broker Health Check: {checks['broker_health_check_ok']}",
            f"Stale Data Check: {checks['stale_data_check_ok']}",
            f"Reconciliation Check: {checks['reconciliation_check_ok']}",
            f"Daily Loss Cap OK: {checks['daily_loss_cap_ok']}",
            f"Max Position Cap OK: {checks['max_position_cap_ok']}",
            f"Requested Position Notional: {runtime['requested_position_notional']:.2f}",
            f"Actual Position Notional: {runtime['actual_position_notional']:.2f}",
            f"Daily PnL: {runtime['daily_pnl']:.2f}",
            f"Data Age Seconds: {runtime['data_age_seconds']:.2f}",
        ]
    )


def write_live_pilot_readiness_report(
    report: Dict[str, Any],
    *,
    json_path: Path,
    text_path: Path,
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    text_path.write_text(format_live_pilot_readiness_report(report), encoding="utf-8")


def _load_json_list(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [dict(item) for item in payload]
    if isinstance(payload, dict):
        rows = payload.get("rows", payload.get("reports", []))
        return [dict(item) for item in rows]
    return []


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate readiness for a manually enabled live pilot")
    parser.add_argument("--config", default="live_pilot_config.example.json")
    parser.add_argument("--offline-results", required=True)
    parser.add_argument("--paper-reports", required=True)
    parser.add_argument("--runtime-status", required=True)
    parser.add_argument("--output-json", default="metrics/live_pilot_readiness.json")
    parser.add_argument("--output-text", default="metrics/live_pilot_readiness.txt")
    return parser


def main() -> None:
    args = _parser().parse_args()
    config = load_live_pilot_config(Path(args.config))
    offline_rows = _load_json_list(Path(args.offline_results))
    paper_reports = _load_json_list(Path(args.paper_reports))
    runtime_payload = json.loads(Path(args.runtime_status).read_text(encoding="utf-8"))
    report = build_live_pilot_readiness_report(
        config,
        offline_rows,
        paper_reports,
        LivePilotRuntimeStatus(**runtime_payload),
    )
    write_live_pilot_readiness_report(
        report,
        json_path=Path(args.output_json),
        text_path=Path(args.output_text),
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
