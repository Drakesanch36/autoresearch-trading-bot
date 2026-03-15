# autoresearch-trading-bot

Autonomous quant strategy evolution system inspired by Karpathy's autoresearch loop using:
- `prepare.py`
- `strategy.py`
- `program.md`

Core loop:
1. Prepare data
2. Generate/edit strategy with LLM
3. Run tests and backtests
4. Commit only improvements

Quick start:
```bash
sudo bash setup-trading-autoresearch.sh https://github.com/YOUR_GITHUB_USERNAME/autoresearch-trading-bot.git /opt/autoresearch-trading-bot
```

Run agent manually:
```bash
./.venv/bin/python run_trading_agent.py --loop-forever
```

Run tests locally:
```bash
pytest tests/
```
