# AI-Powered Quantitative Trading System

This project downloads historical market data, engineers financial features, trains an XGBoost classifier to predict next-day price direction, generates trading signals, backtests the strategy against Buy & Hold, and stress-tests the result with Monte Carlo simulation.

## What It Does

- Collects historical stock data with `yfinance`
- Creates technical and statistical features
- Trains an `XGBoost` model for next-day direction
- Produces long/flat/short trading signals
- Simulates strategy performance versus Buy & Hold
- Runs Monte Carlo robustness analysis on both paths

## Run It

Install dependencies:

```bash
py -m pip install -e .
```

Run against a live symbol:

```bash
quant-trading --symbol AAPL --start 2018-01-01
```

Run the offline demo mode, which generates synthetic data and validates the full pipeline without network access:

```bash
quant-trading --offline-demo
```

## Output

The CLI prints:

- model evaluation metrics
- backtest performance for the strategy and Buy & Hold
- Monte Carlo summary statistics

It also saves a CSV with the test-period signals and equity curves when `--output-dir` is provided.
