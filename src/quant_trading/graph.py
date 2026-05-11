import matplotlib.pyplot as plt

def plot_equity_curves(backtest_result):

    frame = backtest_result.frame

    plt.figure(figsize=(12,6))

    plt.plot(
        frame.index,
        frame["strategy_equity"],
        label="Strategy"
    )

    plt.plot(
        frame.index,
        frame["benchmark_equity"],
        label="Buy & Hold"
    )

    plt.title("Strategy vs Buy & Hold")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid(True)

    plt.show()