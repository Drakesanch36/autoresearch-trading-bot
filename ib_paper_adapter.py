from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from dataclasses import asdict, dataclass
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, TextIO

import requests
from live_pilot_guardrails import LIVE_PILOT_READINESS_SCHEMA_VERSION


DEFAULT_BASE_URL = "https://127.0.0.1:5000/v1/api"


class HTTPSessionLike(Protocol):
    def get(self, url: str, *, params: Optional[Dict[str, Any]] = None, timeout: float = 10.0, verify: bool = True): ...

    def post(
        self,
        url: str,
        *,
        json: Optional[Dict[str, Any]] = None,
        timeout: float = 10.0,
        verify: bool = True,
    ): ...


@dataclass(frozen=True)
class DesiredPosition:
    symbol: str
    target_quantity: float
    contract_id: Optional[int] = None
    signal_id: str = ""
    order_type: str = "MKT"
    tif: str = "DAY"
    exchange: str = "SMART"
    currency: str = "USD"


@dataclass(frozen=True)
class BrokerPosition:
    symbol: str
    quantity: float
    contract_id: Optional[int] = None
    market_price: float = 0.0
    market_value: float = 0.0


@dataclass(frozen=True)
class OrderRequest:
    symbol: str
    quantity: float
    side: str
    contract_id: Optional[int]
    signal_id: str = ""
    client_order_id: str = ""
    order_type: str = "MKT"
    tif: str = "DAY"


@dataclass(frozen=True)
class OrderAck:
    order_id: str
    status: str
    symbol: str
    side: str
    quantity: float
    signal_id: str = ""
    client_order_id: str = ""
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0


@dataclass(frozen=True)
class PositionReconciliation:
    symbol: str
    desired_quantity: float
    actual_quantity: float
    delta_quantity: float
    in_sync: bool


class IBPaperAdapter:
    def __init__(
        self,
        *,
        mode: str = "dry-run",
        account_id: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        session: Optional[HTTPSessionLike] = None,
        timeout_sec: float = 10.0,
        verify_ssl: bool = False,
        log_file: Optional[TextIO] = None,
        live_readiness_report_path: Optional[Path] = None,
        live_readiness_max_age_seconds: float = 300.0,
        live_readiness_future_tolerance_seconds: float = 5.0,
    ) -> None:
        if mode not in {"dry-run", "paper", "live"}:
            raise ValueError("mode must be 'dry-run', 'paper', or 'live'")
        self.mode = mode
        self.account_id = account_id
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()
        self.timeout_sec = timeout_sec
        self.verify_ssl = verify_ssl
        self.log_file = log_file
        self.live_readiness_report_path = live_readiness_report_path
        self.live_readiness_max_age_seconds = live_readiness_max_age_seconds
        self.live_readiness_future_tolerance_seconds = live_readiness_future_tolerance_seconds

    def _log(self, event: str, **payload: Any) -> None:
        entry = {
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            "event": event,
            "account_id": self.account_id or "",
            **payload,
        }
        line = json.dumps(entry, sort_keys=True)
        if self.log_file is None:
            print(line, flush=True)
            return
        self.log_file.write(line + "\n")
        self.log_file.flush()

    def log_system_event(self, event: str, **payload: Any) -> None:
        self._log(event, mode=self.mode, **payload)

    def assert_live_execution_allowed(self) -> None:
        if self.mode != "live":
            return
        if self.live_readiness_report_path is None or not self.live_readiness_report_path.exists():
            self._log("live_guardrail_blocked", mode=self.mode, reason="missing_readiness_report")
            raise RuntimeError("live order submission blocked: missing live readiness report")

        try:
            report = json.loads(self.live_readiness_report_path.read_text(encoding="utf-8"))
        except JSONDecodeError as exc:
            self._log("live_guardrail_blocked", mode=self.mode, reason="malformed_readiness_report", error=str(exc))
            raise RuntimeError("live order submission blocked: malformed live readiness report")

        if not isinstance(report, dict):
            self._log("live_guardrail_blocked", mode=self.mode, reason="invalid_readiness_report_type")
            raise RuntimeError("live order submission blocked: invalid live readiness report schema")

        required_fields = ("generated_at", "schema_version", "live_mode_permitted", "kill_switch_active")
        missing_fields = [field for field in required_fields if field not in report]
        if missing_fields:
            self._log(
                "live_guardrail_blocked",
                mode=self.mode,
                reason="missing_readiness_fields",
                missing_fields=missing_fields,
            )
            raise RuntimeError("live order submission blocked: missing required readiness report fields")

        generated_at_raw = report.get("generated_at")
        schema_version = report.get("schema_version")
        live_mode_permitted = report.get("live_mode_permitted")
        kill_switch_active = report.get("kill_switch_active")
        try:
            generated_at = dt.datetime.fromisoformat(str(generated_at_raw))
        except ValueError:
            self._log("live_guardrail_blocked", mode=self.mode, reason="invalid_generated_at")
            raise RuntimeError("live order submission blocked: invalid readiness report timestamp")
        if generated_at.tzinfo is None:
            generated_at = generated_at.replace(tzinfo=dt.timezone.utc)
        generated_at = generated_at.astimezone(dt.timezone.utc)

        if schema_version != LIVE_PILOT_READINESS_SCHEMA_VERSION:
            self._log(
                "live_guardrail_blocked",
                mode=self.mode,
                reason="invalid_schema_version",
                schema_version=schema_version,
                expected_schema_version=LIVE_PILOT_READINESS_SCHEMA_VERSION,
            )
            raise RuntimeError("live order submission blocked: unexpected readiness report schema_version")
        if not isinstance(live_mode_permitted, bool) or not isinstance(kill_switch_active, bool):
            self._log("live_guardrail_blocked", mode=self.mode, reason="invalid_readiness_flag_types")
            raise RuntimeError("live order submission blocked: invalid readiness report flag types")

        now = dt.datetime.now(dt.timezone.utc)
        future_skew_seconds = (generated_at - now).total_seconds()
        if future_skew_seconds > self.live_readiness_future_tolerance_seconds:
            self._log(
                "live_guardrail_blocked",
                mode=self.mode,
                reason="future_dated_readiness_report",
                future_skew_seconds=future_skew_seconds,
                tolerance_seconds=self.live_readiness_future_tolerance_seconds,
            )
            raise RuntimeError("live order submission blocked: future-dated live readiness report")

        report_age_seconds = max((now - generated_at).total_seconds(), 0.0)
        if report_age_seconds > self.live_readiness_max_age_seconds:
            self._log(
                "live_guardrail_blocked",
                mode=self.mode,
                reason="stale_readiness_report",
                report_age_seconds=report_age_seconds,
                max_age_seconds=self.live_readiness_max_age_seconds,
            )
            raise RuntimeError("live order submission blocked: stale live readiness report")

        if not live_mode_permitted or kill_switch_active:
            self._log(
                "live_guardrail_blocked",
                mode=self.mode,
                reason="readiness_failed",
                live_mode_permitted=live_mode_permitted,
                kill_switch_active=kill_switch_active,
                schema_version=schema_version,
                report_age_seconds=report_age_seconds,
            )
            raise RuntimeError(
                "live order submission blocked: readiness report does not permit live mode"
            )

    def _request(self, method: str, path: str, *, payload: Optional[Dict[str, Any]] = None) -> Any:
        url = f"{self.base_url}{path}"
        if method == "GET":
            response = self.session.get(url, params=payload, timeout=self.timeout_sec, verify=self.verify_ssl)
        elif method == "POST":
            response = self.session.post(url, json=payload, timeout=self.timeout_sec, verify=self.verify_ssl)
        else:
            raise ValueError(f"unsupported method: {method}")

        response.raise_for_status()
        return response.json()

    def resolve_account_id(self) -> str:
        if self.account_id:
            return self.account_id
        payload = self._request("GET", "/iserver/accounts")
        if isinstance(payload, dict):
            accounts = payload.get("accounts", [])
        else:
            accounts = payload
        if not accounts:
            raise RuntimeError("no Interactive Brokers accounts available")
        self.account_id = str(accounts[0])
        return self.account_id

    def fetch_positions(self) -> List[BrokerPosition]:
        account_id = self.resolve_account_id()
        payload = self._request("GET", f"/portfolio/{account_id}/positions/0")
        positions: List[BrokerPosition] = []
        for row in payload:
            quantity = float(row.get("position", row.get("quantity", 0.0)))
            positions.append(
                BrokerPosition(
                    symbol=str(row.get("ticker", row.get("symbol", ""))),
                    quantity=quantity,
                    contract_id=_safe_optional_int(row.get("conid")),
                    market_price=float(row.get("mktPrice", row.get("market_price", 0.0)) or 0.0),
                    market_value=float(row.get("mktValue", row.get("market_value", 0.0)) or 0.0),
                )
            )
        return positions

    def build_reconciliation(
        self,
        desired_position: DesiredPosition,
        positions: Optional[List[BrokerPosition]] = None,
    ) -> PositionReconciliation:
        current_positions = positions if positions is not None else self.fetch_positions()
        actual_quantity = 0.0
        for position in current_positions:
            if position.symbol == desired_position.symbol:
                actual_quantity = position.quantity
                break
        delta_quantity = desired_position.target_quantity - actual_quantity
        return PositionReconciliation(
            symbol=desired_position.symbol,
            desired_quantity=desired_position.target_quantity,
            actual_quantity=actual_quantity,
            delta_quantity=delta_quantity,
            in_sync=abs(delta_quantity) <= 1e-9,
        )

    def submit_order(self, order: OrderRequest) -> OrderAck:
        self._log("submitted_order", mode=self.mode, order=asdict(order))
        if self.mode == "dry-run":
            ack = OrderAck(
                order_id="dry-run-order",
                status="dry_run_acknowledged",
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                signal_id=order.signal_id,
                client_order_id=order.client_order_id or "dry-run-client-order",
                filled_quantity=0.0,
                avg_fill_price=0.0,
            )
            self._log("acknowledged_order", mode=self.mode, order=asdict(ack))
            return ack

        self.assert_live_execution_allowed()

        if order.contract_id is None:
            raise ValueError(f"{self.mode} mode requires contract_id for order submission")

        account_id = self.resolve_account_id()
        client_order_id = order.client_order_id or f"autoresearch-{order.signal_id or int(time.time() * 1000)}"
        payload = {
            "orders": [
                {
                    "acctId": account_id,
                    "conid": order.contract_id,
                    "secType": f"{order.contract_id}:STK",
                    "cOID": client_order_id,
                    "orderType": order.order_type,
                    "listingExchange": "SMART",
                    "outsideRTH": False,
                    "side": order.side,
                    "ticker": order.symbol,
                    "tif": order.tif,
                    "quantity": order.quantity,
                }
            ]
        }
        response = self._request("POST", f"/iserver/account/{account_id}/orders", payload=payload)
        ack_payload = response[0] if isinstance(response, list) and response else response
        ack = OrderAck(
            order_id=str(ack_payload.get("order_id", ack_payload.get("id", ""))),
            status=str(ack_payload.get("order_status", ack_payload.get("status", "Submitted"))),
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            signal_id=order.signal_id,
            client_order_id=str(ack_payload.get("cOID", client_order_id)),
            filled_quantity=float(ack_payload.get("filled", 0.0) or 0.0),
            avg_fill_price=float(ack_payload.get("avg_fill_price", 0.0) or 0.0),
        )
        self._log("acknowledged_order", mode=self.mode, order=asdict(ack))
        return ack

    def get_order_status(self, order_id: str) -> OrderAck:
        if self.mode == "dry-run":
            return OrderAck(
                order_id=order_id,
                status="dry_run_unfilled",
                symbol="",
                side="",
                quantity=0.0,
                signal_id="",
                client_order_id="",
                filled_quantity=0.0,
                avg_fill_price=0.0,
            )

        payload = self._request("GET", f"/iserver/account/order/status/{order_id}")
        order = OrderAck(
            order_id=str(payload.get("order_id", payload.get("id", order_id))),
            status=str(payload.get("order_status", payload.get("status", "Unknown"))),
            symbol=str(payload.get("ticker", payload.get("symbol", ""))),
            side=str(payload.get("side", "")),
            quantity=float(payload.get("quantity", 0.0) or 0.0),
            signal_id=str(payload.get("signal_id", "")),
            client_order_id=str(payload.get("cOID", payload.get("client_order_id", ""))),
            filled_quantity=float(payload.get("filled", 0.0) or 0.0),
            avg_fill_price=float(payload.get("avg_fill_price", 0.0) or 0.0),
        )
        if order.filled_quantity > 0:
            self._log("fill", mode=self.mode, order=asdict(order))
        return order

    def execute_target_position(self, desired_position: DesiredPosition, *, poll_fill: bool = True) -> PositionReconciliation:
        self._log("desired_position", mode=self.mode, desired_position=asdict(desired_position))
        reconciliation = self.build_reconciliation(desired_position)
        if reconciliation.in_sync:
            self._log("end_of_day_reconciliation", mode=self.mode, reconciliation=asdict(reconciliation))
            return reconciliation

        side = "BUY" if reconciliation.delta_quantity > 0 else "SELL"
        order = OrderRequest(
            symbol=desired_position.symbol,
            quantity=abs(reconciliation.delta_quantity),
            side=side,
            contract_id=desired_position.contract_id,
            signal_id=desired_position.signal_id,
            order_type=desired_position.order_type,
            tif=desired_position.tif,
        )
        ack = self.submit_order(order)
        if poll_fill and self.mode == "paper" and ack.order_id:
            self.get_order_status(ack.order_id)

        end_reconciliation = self.build_reconciliation(desired_position)
        self._log("end_of_day_reconciliation", mode=self.mode, reconciliation=asdict(end_reconciliation))
        return end_reconciliation

    def reconcile_positions(self, desired_positions: List[DesiredPosition]) -> List[PositionReconciliation]:
        positions = self.fetch_positions()
        reconciliations = [self.build_reconciliation(desired, positions=positions) for desired in desired_positions]
        self._log(
            "end_of_day_reconciliation",
            mode=self.mode,
            reconciliations=[asdict(item) for item in reconciliations],
        )
        return reconciliations


def _safe_optional_int(value: Any) -> Optional[int]:
    if value in {None, ""}:
        return None
    return int(value)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal Interactive Brokers paper-trading adapter")
    parser.add_argument("--mode", choices=["dry-run", "paper", "live"], default="dry-run")
    parser.add_argument("--account-id", default="")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--target-quantity", type=float, required=True)
    parser.add_argument("--contract-id", type=int)
    parser.add_argument("--log-file", default="")
    parser.add_argument("--live-readiness-report", default="")
    parser.add_argument("--live-readiness-max-age-seconds", type=float, default=300.0)
    parser.add_argument("--live-readiness-future-tolerance-seconds", type=float, default=5.0)
    parser.add_argument("--no-poll-fill", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    log_handle: Optional[TextIO] = None
    try:
        if args.log_file:
            log_path = Path(args.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = log_path.open("a", encoding="utf-8")

        adapter = IBPaperAdapter(
            mode=args.mode,
            account_id=args.account_id or None,
            base_url=args.base_url,
            log_file=log_handle,
            live_readiness_report_path=Path(args.live_readiness_report) if args.live_readiness_report else None,
            live_readiness_max_age_seconds=args.live_readiness_max_age_seconds,
            live_readiness_future_tolerance_seconds=args.live_readiness_future_tolerance_seconds,
        )
        adapter.execute_target_position(
            DesiredPosition(
                symbol=args.symbol,
                target_quantity=args.target_quantity,
                contract_id=args.contract_id,
            ),
            poll_fill=not args.no_poll_fill,
        )
    finally:
        if log_handle is not None:
            log_handle.close()


if __name__ == "__main__":
    main()
