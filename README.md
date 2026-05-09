# AI Capex Momentum Scanner

A simple Python scanner for an AI-capex momentum swing trading workflow.

It reads a manually maintained watchlist, downloads daily OHLCV data with `yfinance`, calculates technical indicators, classifies setups, and writes a daily Markdown report plus CSV.

## What it does

- Scans a broad AI capex universe, not just a copied portfolio.
- Calculates EMA10, EMA21, EMA50, RSI14, ATR14, 20-day average volume, relative volume, distances to EMAs, and 20/50-day highs/lows.
- Classifies tickers as:
  - Pullback Setup
  - Breakout Watch
  - Extended / Do Not Chase
  - Breakdown Risk
  - Neutral
- Generates:
  - `output/scan_results.csv`
  - `output/daily_report.md`
  - `output/failures.txt` when some tickers fail

## What it does not do

- It does not place trades.
- It does not connect to a brokerage account.
- It does not provide financial advice.
- It does not predict future returns.

Use it as a screening and journaling aid only.

## Install

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

Then open:

```text
output/daily_report.md
output/scan_results.csv
```

## Edit the watchlist

Edit `watchlist.yaml`. You can add or remove tickers without changing Python code.

If a ticker belongs to multiple categories, keep it in each relevant section. The scanner deduplicates tickers and preserves all category labels.

## Edit the rules

Edit `config.yaml`. For example:

- Raise `relative_volume_breakout_min` to make breakout rules stricter.
- Lower `pullback_distance_to_ema_percent` to require pullbacks closer to EMA support.
- Lower `rsi_extended` to mark overextended stocks earlier.

## Practical interpretation

- `Pullback Setup`: possible healthy trend pullback or RSI reset.
- `Breakout Watch`: near a possible breakout area; manually check the chart.
- `Extended / Do Not Chase`: trend may still be strong, but new-entry risk/reward is poor.
- `Breakdown Risk`: momentum/structure may be weakening.
- `Neutral`: no clear technical setup.

Always confirm with the chart, market environment, and your own risk plan.
