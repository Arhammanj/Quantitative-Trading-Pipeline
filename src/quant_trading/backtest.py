from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestResult:
    frame: pd.DataFrame
    initial_capital: float
    strategy_final_value: float
    benchmark_final_value: float
    strategy_total_return: float
    benchmark_total_return: float
    strategy_sharpe: float
    benchmark_sharpe: float
    strategy_max_drawdown: float
    benchmark_max_drawdown: float


def _annualized_sharpe(returns: pd.Series, periods_per_year: int = 252) -> float:
    volatility = returns.std(ddof=0)
    if volatility == 0 or pd.isna(volatility):
        return 0.0
    return float(np.sqrt(periods_per_year) * returns.mean() / volatility)


def _max_drawdown(equity_curve: pd.Series) -> float:
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1
    return float(drawdown.min())


def run_backtest(
    frame: pd.DataFrame,
    signals: pd.Series,
    initial_capital: float = 10_000.0,
    transaction_cost_bps: float = 5.0,
) -> BacktestResult:
    if "next_return" not in frame.columns:
        raise ValueError("The frame must include a next_return column.")

    result_frame = frame[["next_return"]].copy()
    result_frame["signal"] = signals.reindex(result_frame.index).fillna(0).astype(float)
    result_frame["position"] = result_frame["signal"]
    result_frame["turnover"] = result_frame["position"].diff().abs().fillna(result_frame["position"].abs())

    cost_rate = transaction_cost_bps / 10_000.0
    result_frame["strategy_return"] = result_frame["position"] * result_frame["next_return"] - result_frame["turnover"] * cost_rate
    result_frame["benchmark_return"] = result_frame["next_return"]

    result_frame["strategy_equity"] = initial_capital * (1 + result_frame["strategy_return"]).cumprod()
    result_frame["benchmark_equity"] = initial_capital * (1 + result_frame["benchmark_return"]).cumprod()

    strategy_final_value = float(result_frame["strategy_equity"].iloc[-1])
    benchmark_final_value = float(result_frame["benchmark_equity"].iloc[-1])

    return BacktestResult(
        frame=result_frame,
        initial_capital=initial_capital,
        strategy_final_value=strategy_final_value,
        benchmark_final_value=benchmark_final_value,
        strategy_total_return=strategy_final_value / initial_capital - 1,
        benchmark_total_return=benchmark_final_value / initial_capital - 1,
        strategy_sharpe=_annualized_sharpe(result_frame["strategy_return"]),
        benchmark_sharpe=_annualized_sharpe(result_frame["benchmark_return"]),
        strategy_max_drawdown=_max_drawdown(result_frame["strategy_equity"]),
        benchmark_max_drawdown=_max_drawdown(result_frame["benchmark_equity"]),
    )
