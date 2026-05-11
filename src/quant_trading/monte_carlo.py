from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MonteCarloResult:
    strategy_final_values: np.ndarray
    benchmark_final_values: np.ndarray
    probability_strategy_beats_benchmark: float
    strategy_summary: dict[str, float]
    benchmark_summary: dict[str, float]


def _simulate_final_values(
    returns: np.ndarray,
    num_simulations: int,
    horizon: int,
    initial_capital: float,
    rng: np.random.Generator,
) -> np.ndarray:
    final_values = np.empty(num_simulations, dtype=float)
    for simulation_index in range(num_simulations):
        sampled_returns = rng.choice(returns, size=horizon, replace=True)
        final_values[simulation_index] = initial_capital * float(np.prod(1 + sampled_returns))
    return final_values


def _summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p05": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
        "std": float(np.std(values, ddof=0)),
    }


def run_monte_carlo(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    initial_capital: float = 10_000.0,
    num_simulations: int = 2_000,
    random_state: int = 42,
) -> MonteCarloResult:
    strategy = strategy_returns.dropna().to_numpy(dtype=float)
    benchmark = benchmark_returns.dropna().to_numpy(dtype=float)
    if len(strategy) == 0 or len(benchmark) == 0:
        raise ValueError("Monte Carlo requires non-empty return series.")

    horizon = min(len(strategy), len(benchmark))
    rng = np.random.default_rng(random_state)

    strategy_final_values = _simulate_final_values(strategy, num_simulations, horizon, initial_capital, rng)
    benchmark_final_values = _simulate_final_values(benchmark, num_simulations, horizon, initial_capital, rng)
    probability_strategy_beats_benchmark = float(np.mean(strategy_final_values > benchmark_final_values))

    return MonteCarloResult(
        strategy_final_values=strategy_final_values,
        benchmark_final_values=benchmark_final_values,
        probability_strategy_beats_benchmark=probability_strategy_beats_benchmark,
        strategy_summary=_summary(strategy_final_values),
        benchmark_summary=_summary(benchmark_final_values),
    )
