from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
from zoneinfo import ZoneInfo


def _coerce_dict(item: Any) -> Dict[str, Any]:
    if isinstance(item, dict):
        return dict(item)
    if is_dataclass(item):
        return asdict(item)
    raise TypeError(f"Unsupported item type: {type(item)!r}")


def _parse_timestamp(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=ZoneInfo("UTC"))
    return parsed


def load_paper_trading_events(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def load_positions_payload(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [dict(item) for item in payload]
    if isinstance(payload, dict):
        rows = payload.get("positions", [])
        return [dict(item) for item in rows]
    return []


def _normalize_desired_positions(desired_positions: Iterable[Any]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for item in desired_positions:
        row = _coerce_dict(item)
        normalized.append(
            {
                "symbol": str(row.get("symbol", "")),
                "target_quantity": float(row.get("target_quantity", 0.0)),
                "signal_id": str(row.get("signal_id", "")),
            }
        )
    return normalized


def _normalize_actual_positions(actual_positions: Iterable[Any]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for item in actual_positions:
        row = _coerce_dict(item)
        normalized.append(
            {
                "symbol": str(row.get("symbol", "")),
                "quantity": float(row.get("quantity", row.get("actual_quantity", 0.0))),
            }
        )
    return normalized


def _event_order_payload(event: Dict[str, Any]) -> Dict[str, Any]:
    return dict(event.get("order", {})) if isinstance(event.get("order"), dict) else {}


def validate_market_hours_behavior(
    events: List[Dict[str, Any]],
    *,
    timezone_name: str = "America/New_York",
    market_open: str = "09:30",
    market_close: str = "16:00",
) -> Tuple[bool, List[str]]:
    tz = ZoneInfo(timezone_name)
    open_hour, open_minute = [int(part) for part in market_open.split(":", 1)]
    close_hour, close_minute = [int(part) for part in market_close.split(":", 1)]
    session_open = time(open_hour, open_minute)
    session_close = time(close_hour, close_minute)
    violations: List[str] = []

    for event in events:
        if event.get("event") not in {"submitted_order", "acknowledged_order", "fill"}:
            continue
        timestamp = _parse_timestamp(str(event.get("timestamp", "")))
        if timestamp is None:
            violations.append(f"missing timestamp for {event.get('event')}")
            continue
        local_time = timestamp.astimezone(tz).time()
        if local_time < session_open or local_time > session_close:
            violations.append(f"{event.get('event')} outside market hours at {timestamp.isoformat()}")
    return not violations, violations


def build_paper_trading_validation_report(
    desired_positions: Iterable[Any],
    actual_positions: Iterable[Any],
    events: List[Dict[str, Any]],
    *,
    account_id: str = "",
    position_tolerance: float = 1e-6,
    timezone_name: str = "America/New_York",
    market_open: str = "09:30",
    market_close: str = "16:00",
) -> Dict[str, Any]:
    desired = _normalize_desired_positions(desired_positions)
    actual = _normalize_actual_positions(actual_positions)
    actual_by_symbol = {row["symbol"]: row["quantity"] for row in actual}
    desired_by_signal = {row["signal_id"] or row["symbol"]: row for row in desired}

    submitted_orders = [event for event in events if event.get("event") == "submitted_order"]
    acknowledged_orders = [event for event in events if event.get("event") == "acknowledged_order"]
    fill_events = [event for event in events if event.get("event") == "fill"]
    restart_events = [event for event in events if event.get("event") == "system_restart"]
    recovery_events = [event for event in events if event.get("event") == "restart_recovery"]

    account_ids = {str(event.get("account_id", "")).strip() for event in events if str(event.get("account_id", "")).strip()}
    broker_desync_violations: List[str] = []
    if len(account_ids) > 1:
        broker_desync_violations.append(f"multiple account_ids observed: {sorted(account_ids)}")
    if account_id and account_ids and account_ids != {account_id}:
        broker_desync_violations.append(f"observed account_ids {sorted(account_ids)} do not match expected {account_id}")

    submitted_keys = {
        _event_order_payload(event).get("client_order_id") or _event_order_payload(event).get("signal_id") or f"{_event_order_payload(event).get('symbol')}:{_event_order_payload(event).get('side')}:{_event_order_payload(event).get('quantity')}"
        for event in submitted_orders
    }
    acknowledged_keys = {
        _event_order_payload(event).get("client_order_id") or _event_order_payload(event).get("signal_id") or f"{_event_order_payload(event).get('symbol')}:{_event_order_payload(event).get('side')}:{_event_order_payload(event).get('quantity')}"
        for event in acknowledged_orders
    }
    unacknowledged_orders = sorted(key for key in submitted_keys if key and key not in acknowledged_keys)

    duplicate_counts: Dict[str, int] = {}
    for event in submitted_orders:
        order = _event_order_payload(event)
        key = order.get("signal_id") or order.get("client_order_id") or f"{order.get('symbol')}:{order.get('side')}:{order.get('quantity')}"
        if not key:
            continue
        duplicate_counts[key] = duplicate_counts.get(key, 0) + 1
    duplicate_orders = sorted(key for key, count in duplicate_counts.items() if count > 1)

    rejected_orders = []
    for event in acknowledged_orders:
        order = _event_order_payload(event)
        status = str(order.get("status", "")).lower()
        if "reject" in status or "cancel" in status:
            rejected_orders.append(
                {
                    "signal_id": order.get("signal_id", ""),
                    "client_order_id": order.get("client_order_id", ""),
                    "symbol": order.get("symbol", ""),
                    "status": order.get("status", ""),
                }
            )

    reconciliations: List[Dict[str, Any]] = []
    for row in desired:
        actual_quantity = actual_by_symbol.get(row["symbol"], 0.0)
        delta_quantity = row["target_quantity"] - actual_quantity
        reconciliations.append(
            {
                "symbol": row["symbol"],
                "signal_id": row["signal_id"],
                "desired_quantity": row["target_quantity"],
                "actual_quantity": actual_quantity,
                "delta_quantity": delta_quantity,
                "within_tolerance": abs(delta_quantity) <= position_tolerance,
            }
        )
    reconciliation_failures = [row for row in reconciliations if not row["within_tolerance"]]

    market_hours_ok, market_hours_violations = validate_market_hours_behavior(
        events,
        timezone_name=timezone_name,
        market_open=market_open,
        market_close=market_close,
    )

    clean_restart_recovery = True
    restart_recovery_violations: List[str] = []
    if restart_events:
        recovery_timestamps = [
            _parse_timestamp(str(event.get("timestamp", "")))
            for event in recovery_events
            if _parse_timestamp(str(event.get("timestamp", ""))) is not None
        ]
        for restart_event in restart_events:
            restart_timestamp = _parse_timestamp(str(restart_event.get("timestamp", "")))
            if restart_timestamp is None or not any(recovery and recovery >= restart_timestamp for recovery in recovery_timestamps):
                clean_restart_recovery = False
                restart_recovery_violations.append(
                    f"restart at {restart_event.get('timestamp', '')} missing matching restart_recovery event"
                )

    signal_mapping_violations: List[str] = []
    signal_to_order_counts: Dict[str, int] = {}
    for event in submitted_orders:
        order = _event_order_payload(event)
        signal_id = str(order.get("signal_id", "")).strip() or str(order.get("symbol", "")).strip()
        if signal_id not in desired_by_signal:
            signal_mapping_violations.append(f"submitted order without intended signal mapping: {signal_id}")
            continue
        desired_row = desired_by_signal[signal_id]
        if str(order.get("symbol", "")) != desired_row["symbol"]:
            signal_mapping_violations.append(f"signal {signal_id} mapped to wrong symbol {order.get('symbol', '')}")
        signal_to_order_counts[signal_id] = signal_to_order_counts.get(signal_id, 0) + 1

    for signal_id, count in signal_to_order_counts.items():
        if count > 1:
            signal_mapping_violations.append(f"signal {signal_id} mapped to {count} submitted orders")

    missed_orders = []
    for row in desired:
        signal_id = row["signal_id"] or row["symbol"]
        has_acknowledged_order = any(
            (
                (_event_order_payload(event).get("signal_id") or _event_order_payload(event).get("symbol")) == signal_id
            )
            for event in acknowledged_orders
        )
        if (not has_acknowledged_order) and abs(row["target_quantity"] - actual_by_symbol.get(row["symbol"], 0.0)) > position_tolerance:
            missed_orders.append(
                {
                    "signal_id": row["signal_id"],
                    "symbol": row["symbol"],
                    "desired_quantity": row["target_quantity"],
                    "actual_quantity": actual_by_symbol.get(row["symbol"], 0.0),
                }
            )

    fill_deviations = []
    for event in fill_events:
        order = _event_order_payload(event)
        signal_id = str(order.get("signal_id", "")).strip() or str(order.get("symbol", "")).strip()
        desired_row = desired_by_signal.get(signal_id)
        if desired_row is None:
            continue
        filled_quantity = float(order.get("filled_quantity", order.get("quantity", 0.0)) or 0.0)
        actual_quantity = actual_by_symbol.get(desired_row["symbol"], 0.0)
        requested_change = abs(desired_row["target_quantity"] - actual_quantity)
        fill_deviations.append(
            {
                "signal_id": desired_row["signal_id"],
                "symbol": desired_row["symbol"],
                "filled_quantity": filled_quantity,
                "reported_fill_price": float(order.get("avg_fill_price", 0.0) or 0.0),
                "fill_quantity_gap": filled_quantity - requested_change,
            }
        )

    health_flags = {
        "no_broker_account_desynchronization": not broker_desync_violations,
        "no_unacknowledged_orders": not unacknowledged_orders,
        "no_duplicate_orders_after_restart": not duplicate_orders,
        "position_reconciliation_within_tolerance": not reconciliation_failures,
        "expected_market_hours_behavior": market_hours_ok,
        "clean_restart_recovery": clean_restart_recovery,
        "consistent_signal_to_order_mapping": not signal_mapping_violations,
    }
    health_flags["overall_pass"] = all(health_flags.values())

    return {
        "intended_positions": desired,
        "actual_broker_positions": actual,
        "reconciliations": reconciliations,
        "missed_orders": missed_orders,
        "rejected_orders": rejected_orders,
        "fill_deviations": fill_deviations,
        "system_health_flags": health_flags,
        "validation_details": {
            "broker_desynchronization": broker_desync_violations,
            "unacknowledged_orders": unacknowledged_orders,
            "duplicate_orders": duplicate_orders,
            "reconciliation_failures": reconciliation_failures,
            "market_hours_violations": market_hours_violations,
            "restart_recovery_violations": restart_recovery_violations,
            "signal_mapping_violations": signal_mapping_violations,
        },
    }


def format_daily_paper_trading_report(report: Dict[str, Any]) -> str:
    flags = report["system_health_flags"]
    return "\n".join(
        [
            "Daily Paper Trading Report",
            f"Intended Positions: {len(report['intended_positions'])}",
            f"Actual Broker Positions: {len(report['actual_broker_positions'])}",
            f"Missed Orders: {len(report['missed_orders'])}",
            f"Rejected Orders: {len(report['rejected_orders'])}",
            f"Fill Deviations: {len(report['fill_deviations'])}",
            f"Health Overall Pass: {flags['overall_pass']}",
            f"No Broker/Account Desynchronization: {flags['no_broker_account_desynchronization']}",
            f"No Unacknowledged Orders: {flags['no_unacknowledged_orders']}",
            f"No Duplicate Orders After Restart: {flags['no_duplicate_orders_after_restart']}",
            f"Position Reconciliation Within Tolerance: {flags['position_reconciliation_within_tolerance']}",
            f"Expected Market-Hours Behavior: {flags['expected_market_hours_behavior']}",
            f"Clean Restart Recovery: {flags['clean_restart_recovery']}",
            f"Consistent Signal-to-Order Mapping: {flags['consistent_signal_to_order_mapping']}",
        ]
    )


def write_daily_paper_trading_report(
    report: Dict[str, Any],
    *,
    json_path: Path,
    text_path: Path,
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    text_path.write_text(format_daily_paper_trading_report(report), encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate paper-trading logs and generate a daily report")
    parser.add_argument("--desired-positions", required=True)
    parser.add_argument("--actual-positions", required=True)
    parser.add_argument("--events", required=True)
    parser.add_argument("--output-json", default="metrics/paper_trading_report.json")
    parser.add_argument("--output-text", default="metrics/paper_trading_report.txt")
    parser.add_argument("--account-id", default="")
    parser.add_argument("--position-tolerance", type=float, default=1e-6)
    return parser


def main() -> None:
    args = _parser().parse_args()
    desired_positions = load_positions_payload(Path(args.desired_positions))
    actual_positions = load_positions_payload(Path(args.actual_positions))
    events = load_paper_trading_events(Path(args.events))
    report = build_paper_trading_validation_report(
        desired_positions,
        actual_positions,
        events,
        account_id=args.account_id,
        position_tolerance=args.position_tolerance,
    )
    write_daily_paper_trading_report(
        report,
        json_path=Path(args.output_json),
        text_path=Path(args.output_text),
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
