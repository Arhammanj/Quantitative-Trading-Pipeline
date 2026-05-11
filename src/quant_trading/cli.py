from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .backtest import run_backtest
from .data import MarketDataRequest, generate_synthetic_market_data, load_market_data
from .features import create_features
from .model import train_xgboost_model
from .monte_carlo import run_monte_carlo


def _build_signals(probabilities: pd.Series, upper_threshold: float, lower_threshold: float) -> pd.Series:
    signals = pd.Series(0, index=probabilities.index, dtype=float)
    signals.loc[probabilities >= upper_threshold] = 1.0
    signals.loc[probabilities <= lower_threshold] = -1.0
    signals.name = "signal"
    return signals


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI-powered quantitative trading system")
    parser.add_argument("--symbol", default="AAPL", help="Market symbol to download from Yahoo Finance")
    parser.add_argument("--start", default="2018-01-01", help="Start date for historical data")
    parser.add_argument("--end", default=None, help="Optional end date for historical data")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of data reserved for testing")
    parser.add_argument("--upper-threshold", type=float, default=0.55, help="Probability threshold for long signals")
    parser.add_argument("--lower-threshold", type=float, default=0.45, help="Probability threshold for short signals")
    parser.add_argument("--transaction-cost-bps", type=float, default=5.0, help="Per-trade transaction cost in basis points")
    parser.add_argument("--initial-capital", type=float, default=10_000.0, help="Initial capital for the backtest")
    parser.add_argument("--cache-dir", default="data_cache", help="Optional cache directory for downloaded market data")
    parser.add_argument("--output-dir", default="outputs", help="Directory for generated artifacts")
    parser.add_argument("--offline-demo", action="store_true", help="Run the entire pipeline on synthetic data")
    parser.add_argument("--num-simulations", type=int, default=2_000, help="Monte Carlo simulation count")
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.offline_demo:
        market_data = generate_synthetic_market_data(start=args.start)
        source_name = "synthetic-demo"
    else:
        request = MarketDataRequest(
            symbol=args.symbol,
            start=args.start,
            end=args.end,
            cache_dir=Path(args.cache_dir),
        )
        market_data = load_market_data(request)
        source_name = args.symbol

    feature_frame = create_features(market_data)
    model_result = train_xgboost_model(feature_frame, test_size=args.test_size)
    signals = _build_signals(model_result.probabilities, args.upper_threshold, args.lower_threshold)
    backtest_result = run_backtest(
        model_result.test_frame,
        signals=signals,
        initial_capital=args.initial_capital,
        transaction_cost_bps=args.transaction_cost_bps,
    )
    monte_carlo_result = run_monte_carlo(
        backtest_result.frame["strategy_return"],
        backtest_result.frame["benchmark_return"],
        initial_capital=args.initial_capital,
        num_simulations=args.num_simulations,
    )

    artifact_path = output_dir / f"{source_name.lower().replace(' ', '_')}_backtest.csv"
    backtest_result.frame.to_csv(artifact_path, index=True)

    print(f"Data source: {source_name}")
    print("\nModel metrics")
    for metric_name, metric_value in model_result.metrics.items():
        if metric_name.endswith("samples"):
            print(f"  {metric_name}: {int(metric_value)}")
        else:
            print(f"  {metric_name}: {metric_value:.4f}")

    print("\nBacktest")
    print(f"  strategy_final_value: {backtest_result.strategy_final_value:.2f}")
    print(f"  benchmark_final_value: {backtest_result.benchmark_final_value:.2f}")
    print(f"  strategy_total_return: {backtest_result.strategy_total_return:.4f}")
    print(f"  benchmark_total_return: {backtest_result.benchmark_total_return:.4f}")
    print(f"  strategy_sharpe: {backtest_result.strategy_sharpe:.4f}")
    print(f"  benchmark_sharpe: {backtest_result.benchmark_sharpe:.4f}")
    print(f"  strategy_max_drawdown: {backtest_result.strategy_max_drawdown:.4f}")
    print(f"  benchmark_max_drawdown: {backtest_result.benchmark_max_drawdown:.4f}")

    print("\nMonte Carlo")
    print(f"  probability_strategy_beats_benchmark: {monte_carlo_result.probability_strategy_beats_benchmark:.4f}")
    print(f"  strategy_median_final_value: {monte_carlo_result.strategy_summary['median']:.2f}")
    print(f"  benchmark_median_final_value: {monte_carlo_result.benchmark_summary['median']:.2f}")
    print(f"\nArtifacts saved to: {artifact_path}")


if __name__ == "__main__":
    main()
