import yfinance as yf
import numpy as np
import pandas as pd


def download_wmt_prices(start: str, end: str) -> pd.DataFrame:
    ticker = yf.Ticker("WMT")
    data = ticker.history(start=start, end=end, auto_adjust=True)
    if data.empty:
        raise ValueError(f"No data returned for WMT between {start} and {end}")
    return data


def compute_log_returns(prices: pd.Series) -> pd.Series:
    return np.log(prices / prices.shift(1)).dropna()


def annualized_mean_return(log_returns: pd.Series, trading_days: int = 252) -> float:
    mean_daily_return = log_returns.mean()
    return mean_daily_return * trading_days


def main() -> None:
    start_date = "2020-01-01"
    end_date = "2023-12-31"
    data = download_wmt_prices(start_date, end_date)
    log_returns = compute_log_returns(data["Close"])
    annual_mean = annualized_mean_return(log_returns)

    print(f"WMT daily prices from {start_date} to {end_date}")
    print(f"Number of trading days: {len(log_returns)}")
    print(f"Mean daily log return: {log_returns.mean():.6f}")
    print(f"Mean annual log return: {annual_mean:.6f}")

    output = pd.DataFrame({
        "Close": data["Close"],
        "LogReturn": log_returns,
    })
    output.to_csv("wmt_daily_log_returns.csv", index=True)
    print("Saved daily close prices and log returns to wmt_daily_log_returns.csv")


if __name__ == "__main__":
    main()
