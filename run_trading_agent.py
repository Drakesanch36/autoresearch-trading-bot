from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import smtplib
import subprocess
import time
from dataclasses import dataclass
from email.mime.text import MIMEText
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

ROOT = Path(__file__).resolve().parent
RESULTS_PATH = ROOT / "results.tsv"
METRICS_PATH = ROOT / "metrics" / "latest_metrics.json"
STRATEGY_PATH = ROOT / "strategy.py"
PROGRAM_PATH = ROOT / "program.md"
EVOLVABLE_REGION_START = "# === EVOLVABLE REGION START ==="
EVOLVABLE_REGION_END = "# === EVOLVABLE REGION END ==="


@dataclass
class CmdResult:
    ok: bool
    code: int
    stdout: str
    stderr: str
    duration_sec: float


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        key, value = s.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _coerce_output_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def run_cmd(cmd: List[str], timeout_sec: int = 30) -> CmdResult:
    start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
        return CmdResult(
            ok=(proc.returncode == 0),
            code=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            duration_sec=time.time() - start,
        )
    except subprocess.TimeoutExpired as exc:
        return CmdResult(
            ok=False,
            code=124,
            stdout=_coerce_output_text(exc.stdout),
            stderr=_coerce_output_text(exc.stderr) + f"\nTIMEOUT: exceeded {timeout_sec}s",
            duration_sec=time.time() - start,
        )


def parse_metrics(path: Path) -> Dict[str, float]:
    if not path.exists():
        return {"objective": -1e9, "sharpe": -1e9, "cagr": -1e9, "max_drawdown": -1.0}
    data = json.loads(path.read_text(encoding="utf-8"))
    test_metrics = data.get("test_metrics", {})
    return {
        "objective": float(data.get("objective", -1e9)),
        "sharpe": float(test_metrics.get("sharpe", -1e9)),
        "cagr": float(test_metrics.get("cagr", -1e9)),
        "max_drawdown": float(test_metrics.get("max_drawdown", -1.0)),
    }


def best_objective(results_path: Path) -> float:
    if not results_path.exists():
        return -1e9
    best = -1e9
    with results_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            try:
                best = max(best, float(row.get("objective", "-1e9")))
            except ValueError:
                continue
    return best


def append_result(
    provider: str,
    model: str,
    iteration: int,
    tests_ok: bool,
    metrics: Dict[str, float],
    commit_hash: str,
    notes: str,
) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not RESULTS_PATH.exists():
        RESULTS_PATH.write_text(
            "timestamp\tprovider\tmodel\titeration\ttest_pass\tsharpe\tcagr\tmax_drawdown\tobjective\tgit_commit\tnotes\n",
            encoding="utf-8",
        )

    now = dt.datetime.now(dt.timezone.utc).isoformat()
    row = [
        now,
        provider,
        model,
        str(iteration),
        str(int(tests_ok)),
        f"{metrics.get('sharpe', 0.0):.8f}",
        f"{metrics.get('cagr', 0.0):.8f}",
        f"{metrics.get('max_drawdown', 0.0):.8f}",
        f"{metrics.get('objective', -1e9):.8f}",
        commit_hash,
        notes.replace("\t", " ").replace("\n", " ")[:300],
    ]
    with RESULTS_PATH.open("a", encoding="utf-8") as f:
        f.write("\t".join(row) + "\n")


def git_commit(message: str) -> str:
    add = run_cmd(["git", "add", "strategy.py", "results.tsv"])
    if not add.ok:
        return ""
    commit = run_cmd(["git", "commit", "-m", message])
    if not commit.ok:
        return ""
    rev = run_cmd(["git", "rev-parse", "--short", "HEAD"])
    return rev.stdout.strip() if rev.ok else ""


def _find_region_bounds(text: str, start_marker: str, end_marker: str) -> Tuple[int, int]:
    start = text.find(start_marker)
    if start < 0:
        raise ValueError(f"missing start marker: {start_marker}")
    end = text.find(end_marker)
    if end < 0:
        raise ValueError(f"missing end marker: {end_marker}")
    if end <= start:
        raise ValueError("invalid marker order")
    return start, end + len(end_marker)


def _mask_region(text: str, start_marker: str, end_marker: str) -> str:
    start, end = _find_region_bounds(text, start_marker, end_marker)
    return text[:start] + "<EVOLVABLE_REGION>" + text[end:]


def validate_strategy_update(old_code: str, new_code: str) -> Tuple[bool, str]:
    try:
        old_masked = _mask_region(old_code, EVOLVABLE_REGION_START, EVOLVABLE_REGION_END)
        new_masked = _mask_region(new_code, EVOLVABLE_REGION_START, EVOLVABLE_REGION_END)
    except ValueError as exc:
        return False, f"strategy_guardrails_missing:{exc}"

    if old_masked != new_masked:
        return False, "immutable_research_surface_changed"

    return True, ""


def extract_code_block(text: str) -> Optional[str]:
    block = re.search(r"```python\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if block:
        return block.group(1).strip() + "\n"
    if "def " in text and "import" in text:
        return text.strip() + "\n"
    return None


def pick_provider(preferred: str) -> Tuple[str, str]:
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()

    if preferred == "anthropic" and anthropic_key:
        return "anthropic", anthropic_key
    if preferred == "openai" and openai_key:
        return "openai", openai_key

    if anthropic_key:
        return "anthropic", anthropic_key
    if openai_key:
        return "openai", openai_key

    raise RuntimeError("No API key found. Set ANTHROPIC_API_KEY and/or OPENAI_API_KEY in .env")


def call_anthropic(api_key: str, model: str, prompt: str, max_retries: int = 5) -> str:
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": model,
        "max_tokens": 3500,
        "temperature": 0.4,
        "messages": [{"role": "user", "content": prompt}],
    }

    wait = 2
    for _ in range(max_retries):
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        if resp.status_code == 429:
            time.sleep(wait)
            wait = min(wait * 2, 60)
            continue
        resp.raise_for_status()
        data = resp.json()
        parts = data.get("content", [])
        text = "\n".join(p.get("text", "") for p in parts if p.get("type") == "text")
        if text:
            return text
        raise RuntimeError("Anthropic response missing text content")
    raise RuntimeError("Anthropic request failed after retries")


def call_openai(api_key: str, model: str, prompt: str, max_retries: int = 5) -> str:
    url = "https://api.openai.com/v1/responses"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "input": prompt,
        "temperature": 0.4,
        "max_output_tokens": 3500,
    }

    wait = 2
    for _ in range(max_retries):
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        if resp.status_code == 429:
            time.sleep(wait)
            wait = min(wait * 2, 60)
            continue
        resp.raise_for_status()
        data = resp.json()

        if isinstance(data.get("output_text"), str) and data["output_text"]:
            return data["output_text"]

        chunks = []
        for item in data.get("output", []):
            for content in item.get("content", []):
                if content.get("type") in {"output_text", "text"}:
                    chunks.append(content.get("text", ""))
        text = "\n".join([c for c in chunks if c])
        if text:
            return text
        raise RuntimeError("OpenAI response missing text content")
    raise RuntimeError("OpenAI request failed after retries")


def send_slack(webhook: str, message: str) -> None:
    if not webhook:
        return
    try:
        requests.post(webhook, json={"text": message[:3500]}, timeout=15)
    except Exception:
        pass


def send_email_summary(message: str) -> None:
    smtp_host = os.getenv("SMTP_HOST", "").strip()
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER", "").strip()
    smtp_pass = os.getenv("SMTP_PASS", "").strip()
    email_to = os.getenv("SUMMARY_EMAIL_TO", "").strip()
    email_from = os.getenv("SUMMARY_EMAIL_FROM", smtp_user).strip()

    if not smtp_host or not email_to or not email_from:
        return

    msg = MIMEText(message, "plain", "utf-8")
    msg["Subject"] = "Autoresearch Trading Daily Summary"
    msg["From"] = email_from
    msg["To"] = email_to

    try:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=20) as server:
            server.starttls()
            if smtp_user:
                server.login(smtp_user, smtp_pass)
            server.sendmail(email_from, [email_to], msg.as_string())
    except Exception:
        pass


def build_prompt(program_md: str, current_strategy: str, metrics: Dict[str, float], iteration: int) -> str:
    return f"""
You are editing strategy.py for autonomous trading research.

Rules:
- Return ONLY the full updated strategy.py in a single ```python fenced block.
- Only edit code between `{EVOLVABLE_REGION_START}` and `{EVOLVABLE_REGION_END}`.
- Preserve all code outside the evolvable region byte-for-byte.
- Keep the evolvable region focused on raw signal logic only.
- Preserve all function names and CLI behavior.
- Keep no-lookahead behavior and walk-forward validation.
- Keep runtime for each command under 30 seconds on CPU.
- Favor robust out-of-sample performance over fragile in-sample overfit.

Current objective: {metrics.get('objective', -1e9):.6f}
Current sharpe: {metrics.get('sharpe', -1e9):.6f}
Current cagr: {metrics.get('cagr', -1e9):.6f}
Current max_drawdown: {metrics.get('max_drawdown', -1.0):.6f}
Iteration: {iteration}

PROGRAM.MD:
{program_md}

Current strategy.py:
```python
{current_strategy}
```
""".strip()


def run_iteration(
    iteration: int,
    provider: str,
    key: str,
    model: str,
    args: argparse.Namespace,
) -> Tuple[bool, Dict[str, float], str]:
    prepare_cmd = [
        "python",
        "prepare.py",
        "--symbol",
        args.symbol,
        "--source",
        args.source,
        "--timeframe",
        args.timeframe,
        "--start",
        args.start,
        "--out-dir",
        "data",
    ]
    if args.source == "ccxt":
        prepare_cmd.extend(["--exchange", args.exchange])

    prep = run_cmd(prepare_cmd, timeout_sec=30)
    if not prep.ok:
        return False, {"objective": -1e9, "sharpe": -1e9, "cagr": -1e9, "max_drawdown": -1.0}, "prepare_failed"

    # Baseline run before proposing a mutation.
    base = run_cmd(["python", "strategy.py", "--data-dir", "data", "--output", "metrics/latest_metrics.json"], timeout_sec=30)
    if not base.ok:
        return False, {"objective": -1e9, "sharpe": -1e9, "cagr": -1e9, "max_drawdown": -1.0}, "baseline_strategy_failed"

    current_metrics = parse_metrics(METRICS_PATH)
    best_seen = best_objective(RESULTS_PATH)

    program_text = PROGRAM_PATH.read_text(encoding="utf-8")
    old_code = STRATEGY_PATH.read_text(encoding="utf-8")
    prompt = build_prompt(program_text, old_code, current_metrics, iteration)

    try:
        if provider == "anthropic":
            llm_text = call_anthropic(key, model, prompt)
        else:
            llm_text = call_openai(key, model, prompt)
    except Exception as exc:
        return False, current_metrics, f"llm_error:{exc}"

    new_code = extract_code_block(llm_text)
    if not new_code:
        return False, current_metrics, "llm_response_missing_code"

    valid_update, violation = validate_strategy_update(old_code, new_code)
    if not valid_update:
        return False, current_metrics, violation

    STRATEGY_PATH.write_text(new_code, encoding="utf-8")

    test_run = run_cmd(["pytest", "tests/", "-q"], timeout_sec=30)
    if not test_run.ok:
        STRATEGY_PATH.write_text(old_code, encoding="utf-8")
        return False, current_metrics, "pytest_failed_after_edit"

    strat_run = run_cmd(["python", "strategy.py", "--data-dir", "data", "--output", "metrics/latest_metrics.json"], timeout_sec=30)
    if not strat_run.ok:
        STRATEGY_PATH.write_text(old_code, encoding="utf-8")
        return False, current_metrics, "candidate_strategy_failed"

    candidate_metrics = parse_metrics(METRICS_PATH)

    if candidate_metrics["objective"] <= best_seen:
        STRATEGY_PATH.write_text(old_code, encoding="utf-8")
        return False, candidate_metrics, "no_improvement"

    # Required: run tests again immediately before commit.
    pre_commit_test = run_cmd(["pytest", "tests/", "-q"], timeout_sec=30)
    if not pre_commit_test.ok:
        STRATEGY_PATH.write_text(old_code, encoding="utf-8")
        return False, candidate_metrics, "pre_commit_pytest_failed"

    commit_hash = git_commit(
        f"agent: improve objective to {candidate_metrics['objective']:.6f} at iter {iteration}"
    )
    if not commit_hash:
        return False, candidate_metrics, "git_commit_failed"

    return True, candidate_metrics, commit_hash


def summarize_last_day(rows: List[Dict[str, str]]) -> str:
    if not rows:
        return "No iterations recorded today."
    best = max(rows, key=lambda r: float(r.get("objective", "-1e9")))
    return (
        f"Daily trading-agent summary: iterations={len(rows)}, "
        f"best_objective={best.get('objective')}, best_sharpe={best.get('sharpe')}, "
        f"best_cagr={best.get('cagr')}, worst_drawdown={best.get('max_drawdown')}"
    )


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Autonomous 24/7 strategy evolution loop")
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--loop-forever", action="store_true")
    p.add_argument("--sleep-seconds", type=float, default=2.0)
    p.add_argument("--provider", choices=["auto", "anthropic", "openai"], default="auto")
    p.add_argument("--model", default="claude-3-7-sonnet-latest")
    p.add_argument("--fallback-model", default="gpt-4.1-mini")
    p.add_argument("--symbol", default="SPY")
    p.add_argument("--source", choices=["yfinance", "ccxt"], default="yfinance")
    p.add_argument("--exchange", default="binance")
    p.add_argument("--timeframe", default="1d")
    p.add_argument("--start", default="2016-01-01")
    return p


def main() -> None:
    load_dotenv(ROOT / ".env")
    args = _parser().parse_args()

    preferred = "anthropic" if args.provider == "auto" else args.provider
    provider, key = pick_provider(preferred)

    # Fallback model/provider logic.
    primary_model = args.model
    if provider == "openai" and primary_model.startswith("claude"):
        primary_model = args.fallback_model
    if provider == "anthropic" and primary_model.startswith("gpt"):
        primary_model = "claude-3-7-sonnet-latest"

    webhook = os.getenv("SLACK_WEBHOOK_URL", "").strip()

    iteration = 1
    daily_rows: List[Dict[str, str]] = []
    current_day = dt.datetime.now(dt.timezone.utc).date()

    while True:
        ok, metrics, meta = run_iteration(iteration, provider, key, primary_model, args)

        notes = "committed" if ok else meta
        commit_hash = meta if ok else ""
        append_result(
            provider=provider,
            model=primary_model,
            iteration=iteration,
            tests_ok=ok,
            metrics=metrics,
            commit_hash=commit_hash,
            notes=notes,
        )

        daily_rows.append(
            {
                "objective": f"{metrics.get('objective', -1e9):.6f}",
                "sharpe": f"{metrics.get('sharpe', -1e9):.6f}",
                "cagr": f"{metrics.get('cagr', -1e9):.6f}",
                "max_drawdown": f"{metrics.get('max_drawdown', 0.0):.6f}",
            }
        )

        now_day = dt.datetime.now(dt.timezone.utc).date()
        if now_day != current_day:
            summary = summarize_last_day(daily_rows)
            send_slack(webhook, summary)
            send_email_summary(summary)
            daily_rows = []
            current_day = now_day

        if not args.loop_forever and iteration >= args.iterations:
            break

        iteration += 1
        time.sleep(max(args.sleep_seconds, 0.0))

    summary = summarize_last_day(daily_rows)
    send_slack(webhook, summary)
    send_email_summary(summary)


if __name__ == "__main__":
    main()
