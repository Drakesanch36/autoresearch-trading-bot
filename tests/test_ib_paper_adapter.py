from __future__ import annotations

import io
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ib_paper_adapter import BrokerPosition, DesiredPosition, IBPaperAdapter, OrderRequest


class DummyResponse:
    def __init__(self, payload, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"http error {self.status_code}")

    def json(self):
        return self._payload


class DummySession:
    def __init__(self) -> None:
        self.get_calls = []
        self.post_calls = []
        self.get_payloads = {}
        self.post_payloads = {}

    def get(self, url: str, *, params=None, timeout: float = 10.0, verify: bool = True):
        self.get_calls.append({"url": url, "params": params, "timeout": timeout, "verify": verify})
        return DummyResponse(self.get_payloads[url])

    def post(self, url: str, *, json=None, timeout: float = 10.0, verify: bool = True):
        self.post_calls.append({"url": url, "json": json, "timeout": timeout, "verify": verify})
        return DummyResponse(self.post_payloads[url])


def parse_log_lines(buffer: io.StringIO) -> list[dict]:
    return [json.loads(line) for line in buffer.getvalue().splitlines() if line.strip()]


def write_readiness_report(
    path: Path,
    *,
    generated_at: datetime | None = None,
    schema_version: int = 1,
    live_mode_permitted: bool = True,
    kill_switch_active: bool = False,
) -> None:
    timestamp = generated_at or datetime.now(timezone.utc)
    path.write_text(
        json.dumps(
            {
                "generated_at": timestamp.isoformat(),
                "schema_version": schema_version,
                "live_mode_permitted": live_mode_permitted,
                "kill_switch_active": kill_switch_active,
            }
        ),
        encoding="utf-8",
    )


def test_fetch_positions_reads_paper_positions() -> None:
    session = DummySession()
    session.get_payloads["https://paper.local/v1/api/portfolio/DU123/positions/0"] = [
        {"ticker": "SPY", "position": 12, "conid": 756733, "mktPrice": 500.25, "mktValue": 6003.0},
    ]
    adapter = IBPaperAdapter(
        mode="paper",
        account_id="DU123",
        base_url="https://paper.local/v1/api",
        session=session,
        log_file=io.StringIO(),
    )

    positions = adapter.fetch_positions()

    assert len(positions) == 1
    assert positions[0].symbol == "SPY"
    assert positions[0].quantity == 12.0
    assert positions[0].contract_id == 756733


def test_execute_target_position_dry_run_logs_without_network() -> None:
    log_buffer = io.StringIO()
    adapter = IBPaperAdapter(
        mode="dry-run",
        account_id="DU123",
        session=DummySession(),
        log_file=log_buffer,
    )
    adapter.fetch_positions = lambda: [BrokerPosition(symbol="SPY", quantity=5.0, contract_id=756733)]

    reconciliation = adapter.execute_target_position(
        DesiredPosition(symbol="SPY", target_quantity=8.0, contract_id=756733),
        poll_fill=False,
    )

    assert reconciliation.delta_quantity == 3.0
    events = parse_log_lines(log_buffer)
    assert [entry["event"] for entry in events] == [
        "desired_position",
        "submitted_order",
        "acknowledged_order",
        "end_of_day_reconciliation",
    ]
    assert events[1]["order"]["side"] == "BUY"
    assert events[2]["order"]["status"] == "dry_run_acknowledged"


def test_submit_order_and_track_fill_in_paper_mode() -> None:
    session = DummySession()
    session.post_payloads["https://paper.local/v1/api/iserver/account/DU123/orders"] = [
        {"order_id": "abc123", "order_status": "Submitted", "filled": 0.0, "avg_fill_price": 0.0}
    ]
    session.get_payloads["https://paper.local/v1/api/iserver/account/order/status/abc123"] = {
        "order_id": "abc123",
        "order_status": "Filled",
        "ticker": "SPY",
        "side": "BUY",
        "quantity": 3.0,
        "filled": 3.0,
        "avg_fill_price": 501.25,
    }
    log_buffer = io.StringIO()
    adapter = IBPaperAdapter(
        mode="paper",
        account_id="DU123",
        base_url="https://paper.local/v1/api",
        session=session,
        log_file=log_buffer,
    )

    ack = adapter.submit_order(
        OrderRequest(
            symbol="SPY",
            quantity=3.0,
            side="BUY",
            contract_id=756733,
        )
    )
    fill = adapter.get_order_status("abc123")

    assert ack.order_id == "abc123"
    assert ack.status == "Submitted"
    assert fill.status == "Filled"
    assert fill.filled_quantity == 3.0
    assert session.post_calls[0]["json"]["orders"][0]["conid"] == 756733
    events = parse_log_lines(log_buffer)
    assert [entry["event"] for entry in events] == ["submitted_order", "acknowledged_order", "fill"]


def test_live_mode_blocks_order_submission_without_readiness_report(tmp_path: Path) -> None:
    session = DummySession()
    log_buffer = io.StringIO()
    adapter = IBPaperAdapter(
        mode="live",
        account_id="DU123",
        base_url="https://paper.local/v1/api",
        session=session,
        log_file=log_buffer,
        live_readiness_report_path=tmp_path / "missing.json",
    )

    try:
        adapter.submit_order(
            OrderRequest(
                symbol="SPY",
                quantity=1.0,
                side="BUY",
                contract_id=756733,
                signal_id="sig-live",
            )
        )
    except RuntimeError as exc:
        assert "missing live readiness report" in str(exc)
    else:
        raise AssertionError("expected live readiness block")

    assert session.post_calls == []
    assert parse_log_lines(log_buffer)[-1]["event"] == "live_guardrail_blocked"


def test_live_mode_blocks_order_submission_when_readiness_fails(tmp_path: Path) -> None:
    session = DummySession()
    readiness_path = tmp_path / "readiness.json"
    write_readiness_report(
        readiness_path,
        live_mode_permitted=False,
        kill_switch_active=True,
    )
    log_buffer = io.StringIO()
    adapter = IBPaperAdapter(
        mode="live",
        account_id="DU123",
        base_url="https://paper.local/v1/api",
        session=session,
        log_file=log_buffer,
        live_readiness_report_path=readiness_path,
    )

    try:
        adapter.submit_order(
            OrderRequest(
                symbol="SPY",
                quantity=1.0,
                side="BUY",
                contract_id=756733,
                signal_id="sig-live",
            )
        )
    except RuntimeError as exc:
        assert "does not permit live mode" in str(exc)
    else:
        raise AssertionError("expected live readiness block")

    assert session.post_calls == []
    events = parse_log_lines(log_buffer)
    assert events[-1]["event"] == "live_guardrail_blocked"
    assert events[-1]["kill_switch_active"] is True


def test_live_mode_allows_submission_only_when_readiness_passes(tmp_path: Path) -> None:
    session = DummySession()
    session.post_payloads["https://paper.local/v1/api/iserver/account/DU123/orders"] = [
        {"order_id": "live123", "order_status": "Submitted", "filled": 0.0, "avg_fill_price": 0.0}
    ]
    readiness_path = tmp_path / "readiness.json"
    write_readiness_report(readiness_path, live_mode_permitted=True, kill_switch_active=False)
    log_buffer = io.StringIO()
    adapter = IBPaperAdapter(
        mode="live",
        account_id="DU123",
        base_url="https://paper.local/v1/api",
        session=session,
        log_file=log_buffer,
        live_readiness_report_path=readiness_path,
    )

    ack = adapter.submit_order(
        OrderRequest(
            symbol="SPY",
            quantity=1.0,
            side="BUY",
            contract_id=756733,
            signal_id="sig-live",
        )
    )

    assert ack.order_id == "live123"
    assert len(session.post_calls) == 1


def test_live_mode_blocks_order_submission_for_malformed_json(tmp_path: Path) -> None:
    session = DummySession()
    readiness_path = tmp_path / "readiness.json"
    readiness_path.write_text("{not-json", encoding="utf-8")
    adapter = IBPaperAdapter(
        mode="live",
        account_id="DU123",
        base_url="https://paper.local/v1/api",
        session=session,
        log_file=io.StringIO(),
        live_readiness_report_path=readiness_path,
    )

    try:
        adapter.submit_order(OrderRequest(symbol="SPY", quantity=1.0, side="BUY", contract_id=756733))
    except RuntimeError as exc:
        assert "malformed live readiness report" in str(exc)
    else:
        raise AssertionError("expected malformed readiness block")

    assert session.post_calls == []


def test_live_mode_blocks_order_submission_when_required_fields_are_missing(tmp_path: Path) -> None:
    session = DummySession()
    readiness_path = tmp_path / "readiness.json"
    readiness_path.write_text(json.dumps({"live_mode_permitted": True}), encoding="utf-8")
    adapter = IBPaperAdapter(
        mode="live",
        account_id="DU123",
        base_url="https://paper.local/v1/api",
        session=session,
        log_file=io.StringIO(),
        live_readiness_report_path=readiness_path,
    )

    try:
        adapter.submit_order(OrderRequest(symbol="SPY", quantity=1.0, side="BUY", contract_id=756733))
    except RuntimeError as exc:
        assert "missing required readiness report fields" in str(exc)
    else:
        raise AssertionError("expected missing-field readiness block")

    assert session.post_calls == []


def test_live_mode_blocks_order_submission_for_wrong_schema_version(tmp_path: Path) -> None:
    session = DummySession()
    readiness_path = tmp_path / "readiness.json"
    write_readiness_report(
        readiness_path,
        schema_version=999,
        live_mode_permitted=True,
        kill_switch_active=False,
    )
    adapter = IBPaperAdapter(
        mode="live",
        account_id="DU123",
        base_url="https://paper.local/v1/api",
        session=session,
        log_file=io.StringIO(),
        live_readiness_report_path=readiness_path,
    )

    try:
        adapter.submit_order(OrderRequest(symbol="SPY", quantity=1.0, side="BUY", contract_id=756733))
    except RuntimeError as exc:
        assert "unexpected readiness report schema_version" in str(exc)
    else:
        raise AssertionError("expected wrong-schema readiness block")

    assert session.post_calls == []


def test_live_mode_blocks_order_submission_when_readiness_report_is_stale(tmp_path: Path) -> None:
    session = DummySession()
    readiness_path = tmp_path / "readiness.json"
    write_readiness_report(
        readiness_path,
        generated_at=datetime.now(timezone.utc) - timedelta(seconds=600),
        live_mode_permitted=True,
        kill_switch_active=False,
    )
    adapter = IBPaperAdapter(
        mode="live",
        account_id="DU123",
        base_url="https://paper.local/v1/api",
        session=session,
        log_file=io.StringIO(),
        live_readiness_report_path=readiness_path,
        live_readiness_max_age_seconds=60.0,
    )

    try:
        adapter.submit_order(OrderRequest(symbol="SPY", quantity=1.0, side="BUY", contract_id=756733))
    except RuntimeError as exc:
        assert "stale live readiness report" in str(exc)
    else:
        raise AssertionError("expected stale readiness block")

    assert session.post_calls == []


def test_live_mode_blocks_order_submission_when_report_is_future_dated_beyond_tolerance(tmp_path: Path) -> None:
    session = DummySession()
    readiness_path = tmp_path / "readiness.json"
    write_readiness_report(
        readiness_path,
        generated_at=datetime.now(timezone.utc) + timedelta(seconds=120),
        live_mode_permitted=True,
        kill_switch_active=False,
    )
    adapter = IBPaperAdapter(
        mode="live",
        account_id="DU123",
        base_url="https://paper.local/v1/api",
        session=session,
        log_file=io.StringIO(),
        live_readiness_report_path=readiness_path,
        live_readiness_future_tolerance_seconds=5.0,
    )

    try:
        adapter.submit_order(OrderRequest(symbol="SPY", quantity=1.0, side="BUY", contract_id=756733))
    except RuntimeError as exc:
        assert "future-dated live readiness report" in str(exc)
    else:
        raise AssertionError("expected future-dated readiness block")

    assert session.post_calls == []


def test_live_mode_allows_fresh_valid_report(tmp_path: Path) -> None:
    session = DummySession()
    session.post_payloads["https://paper.local/v1/api/iserver/account/DU123/orders"] = [
        {"order_id": "fresh123", "order_status": "Submitted", "filled": 0.0, "avg_fill_price": 0.0}
    ]
    readiness_path = tmp_path / "readiness.json"
    write_readiness_report(
        readiness_path,
        generated_at=datetime.now(timezone.utc) - timedelta(seconds=10),
        live_mode_permitted=True,
        kill_switch_active=False,
    )
    adapter = IBPaperAdapter(
        mode="live",
        account_id="DU123",
        base_url="https://paper.local/v1/api",
        session=session,
        log_file=io.StringIO(),
        live_readiness_report_path=readiness_path,
        live_readiness_max_age_seconds=60.0,
    )

    ack = adapter.submit_order(OrderRequest(symbol="SPY", quantity=1.0, side="BUY", contract_id=756733))

    assert ack.order_id == "fresh123"
    assert len(session.post_calls) == 1


def test_reconcile_positions_reports_intended_vs_actual() -> None:
    session = DummySession()
    session.get_payloads["https://paper.local/v1/api/portfolio/DU123/positions/0"] = [
        {"ticker": "SPY", "position": 7, "conid": 756733},
        {"ticker": "QQQ", "position": -2, "conid": 320227571},
    ]
    log_buffer = io.StringIO()
    adapter = IBPaperAdapter(
        mode="paper",
        account_id="DU123",
        base_url="https://paper.local/v1/api",
        session=session,
        log_file=log_buffer,
    )

    reconciliations = adapter.reconcile_positions(
        [
            DesiredPosition(symbol="SPY", target_quantity=10.0, contract_id=756733),
            DesiredPosition(symbol="QQQ", target_quantity=-2.0, contract_id=320227571),
        ]
    )

    assert len(reconciliations) == 2
    assert reconciliations[0].delta_quantity == 3.0
    assert reconciliations[1].in_sync is True
    events = parse_log_lines(log_buffer)
    assert events[-1]["event"] == "end_of_day_reconciliation"
    assert events[-1]["reconciliations"][0]["symbol"] == "SPY"


def test_execute_target_position_in_sync_only_logs_reconciliation() -> None:
    log_buffer = io.StringIO()
    adapter = IBPaperAdapter(
        mode="dry-run",
        account_id="DU123",
        session=DummySession(),
        log_file=log_buffer,
    )
    adapter.fetch_positions = lambda: [BrokerPosition(symbol="SPY", quantity=8.0, contract_id=756733)]

    reconciliation = adapter.execute_target_position(
        DesiredPosition(symbol="SPY", target_quantity=8.0, contract_id=756733),
        poll_fill=False,
    )

    assert reconciliation.in_sync is True
    events = parse_log_lines(log_buffer)
    assert [entry["event"] for entry in events] == ["desired_position", "end_of_day_reconciliation"]
