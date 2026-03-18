from __future__ import annotations

import json
from pathlib import Path

from ib_paper_adapter import BrokerPosition, DesiredPosition
from paper_trading_validation import (
    build_paper_trading_validation_report,
    format_daily_paper_trading_report,
    load_paper_trading_events,
    load_positions_payload,
    write_daily_paper_trading_report,
)


def make_event(timestamp: str, event: str, *, account_id: str = "DU123", order: dict | None = None, **extra) -> dict:
    payload = {
        "timestamp": timestamp,
        "event": event,
        "account_id": account_id,
    }
    if order is not None:
        payload["order"] = order
    payload.update(extra)
    return payload


def test_build_paper_trading_validation_report_passes_for_clean_day() -> None:
    desired = [DesiredPosition(symbol="SPY", target_quantity=10.0, contract_id=756733, signal_id="sig-spy")]
    actual = [BrokerPosition(symbol="SPY", quantity=10.0, contract_id=756733)]
    events = [
        make_event("2026-03-17T14:35:00+00:00", "desired_position"),
        make_event(
            "2026-03-17T14:36:00+00:00",
            "submitted_order",
            order={"symbol": "SPY", "side": "BUY", "quantity": 3.0, "signal_id": "sig-spy", "client_order_id": "coid-1"},
        ),
        make_event(
            "2026-03-17T14:36:01+00:00",
            "acknowledged_order",
            order={
                "symbol": "SPY",
                "side": "BUY",
                "quantity": 3.0,
                "signal_id": "sig-spy",
                "client_order_id": "coid-1",
                "status": "Submitted",
            },
        ),
        make_event(
            "2026-03-17T14:36:05+00:00",
            "fill",
            order={
                "symbol": "SPY",
                "side": "BUY",
                "quantity": 3.0,
                "filled_quantity": 3.0,
                "avg_fill_price": 505.25,
                "signal_id": "sig-spy",
                "client_order_id": "coid-1",
            },
        ),
        make_event("2026-03-17T20:00:00+00:00", "restart_recovery"),
    ]

    report = build_paper_trading_validation_report(desired, actual, events, account_id="DU123")

    assert report["system_health_flags"]["overall_pass"] is True
    assert report["missed_orders"] == []
    assert report["rejected_orders"] == []
    assert len(report["fill_deviations"]) == 1


def test_build_paper_trading_validation_report_detects_failures() -> None:
    desired = [DesiredPosition(symbol="SPY", target_quantity=10.0, contract_id=756733, signal_id="sig-spy")]
    actual = [BrokerPosition(symbol="SPY", quantity=8.0, contract_id=756733)]
    events = [
        make_event("2026-03-17T13:00:00+00:00", "system_restart"),
        make_event(
            "2026-03-17T13:15:00+00:00",
            "submitted_order",
            order={"symbol": "SPY", "side": "BUY", "quantity": 2.0, "signal_id": "sig-spy", "client_order_id": "coid-1"},
        ),
        make_event(
            "2026-03-17T13:16:00+00:00",
            "submitted_order",
            order={"symbol": "SPY", "side": "BUY", "quantity": 2.0, "signal_id": "sig-spy", "client_order_id": "coid-1"},
        ),
        make_event(
            "2026-03-17T21:30:00+00:00",
            "acknowledged_order",
            order={
                "symbol": "SPY",
                "side": "BUY",
                "quantity": 2.0,
                "signal_id": "sig-spy",
                "client_order_id": "coid-1",
                "status": "Rejected",
            },
            account_id="DU999",
        ),
    ]

    report = build_paper_trading_validation_report(desired, actual, events, account_id="DU123", position_tolerance=0.5)

    flags = report["system_health_flags"]
    assert flags["overall_pass"] is False
    assert flags["no_broker_account_desynchronization"] is False
    assert flags["no_duplicate_orders_after_restart"] is False
    assert flags["position_reconciliation_within_tolerance"] is False
    assert flags["expected_market_hours_behavior"] is False
    assert flags["clean_restart_recovery"] is False
    assert flags["consistent_signal_to_order_mapping"] is False
    assert len(report["rejected_orders"]) == 1


def test_unacknowledged_and_missed_orders_are_reported() -> None:
    desired = [DesiredPosition(symbol="QQQ", target_quantity=5.0, contract_id=320227571, signal_id="sig-qqq")]
    actual = [BrokerPosition(symbol="QQQ", quantity=0.0, contract_id=320227571)]
    events = [
        make_event(
            "2026-03-17T15:00:00+00:00",
            "submitted_order",
            order={"symbol": "QQQ", "side": "BUY", "quantity": 5.0, "signal_id": "sig-qqq", "client_order_id": "coid-qqq"},
        )
    ]

    report = build_paper_trading_validation_report(desired, actual, events, account_id="DU123")

    assert report["validation_details"]["unacknowledged_orders"] == ["coid-qqq"]
    assert len(report["missed_orders"]) == 1


def test_report_writers_emit_machine_and_human_outputs(tmp_path: Path) -> None:
    report = {
        "intended_positions": [{"symbol": "SPY", "target_quantity": 10.0, "signal_id": "sig-spy"}],
        "actual_broker_positions": [{"symbol": "SPY", "quantity": 10.0}],
        "missed_orders": [],
        "rejected_orders": [],
        "fill_deviations": [],
        "system_health_flags": {
            "overall_pass": True,
            "no_broker_account_desynchronization": True,
            "no_unacknowledged_orders": True,
            "no_duplicate_orders_after_restart": True,
            "position_reconciliation_within_tolerance": True,
            "expected_market_hours_behavior": True,
            "clean_restart_recovery": True,
            "consistent_signal_to_order_mapping": True,
        },
        "validation_details": {},
        "reconciliations": [],
    }
    json_path = tmp_path / "paper_report.json"
    text_path = tmp_path / "paper_report.txt"

    write_daily_paper_trading_report(report, json_path=json_path, text_path=text_path)

    assert json.loads(json_path.read_text(encoding="utf-8"))["system_health_flags"]["overall_pass"] is True
    text = text_path.read_text(encoding="utf-8")
    assert "Daily Paper Trading Report" in text
    assert "Missed Orders: 0" in text
    assert "Consistent Signal-to-Order Mapping: True" in text


def test_loaders_read_json_payloads_and_event_logs(tmp_path: Path) -> None:
    events_path = tmp_path / "events.jsonl"
    events_path.write_text(
        json.dumps(make_event("2026-03-17T15:00:00+00:00", "desired_position")) + "\n",
        encoding="utf-8",
    )
    positions_path = tmp_path / "positions.json"
    positions_path.write_text(
        json.dumps({"positions": [{"symbol": "SPY", "quantity": 10.0}]}),
        encoding="utf-8",
    )

    events = load_paper_trading_events(events_path)
    positions = load_positions_payload(positions_path)

    assert events[0]["event"] == "desired_position"
    assert positions[0]["symbol"] == "SPY"
    assert "Health Overall Pass:" in format_daily_paper_trading_report(
        {
            "intended_positions": [],
            "actual_broker_positions": [],
            "missed_orders": [],
            "rejected_orders": [],
            "fill_deviations": [],
            "system_health_flags": {
                "overall_pass": True,
                "no_broker_account_desynchronization": True,
                "no_unacknowledged_orders": True,
                "no_duplicate_orders_after_restart": True,
                "position_reconciliation_within_tolerance": True,
                "expected_market_hours_behavior": True,
                "clean_restart_recovery": True,
                "consistent_signal_to_order_mapping": True,
            },
        }
    )
