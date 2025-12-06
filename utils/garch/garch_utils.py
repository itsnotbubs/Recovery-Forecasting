from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Literal

import numpy as np
import pandas as pd
from arch import arch_model
import yfinance as yf


GarchCoreModel = Literal["garch", "gjr"]
GarchDist = Literal["normal", "t"]


@dataclass
class RollingGarchResult:
    params: pd.DataFrame
    last_sigma: pd.Series
    std_resid: pd.DataFrame | None = None


def compute_log_returns(futures: pd.Series) -> pd.Series:
    r = np.log(futures / futures.shift(1))
    return r.dropna()


def _make_arch_model(
    window_returns: pd.Series,
    model: GarchCoreModel = "garch",
    dist: GarchDist = "normal",
):

    r = 100 * window_returns
    p, q = 1, 1

    # o > 0 -> GJR-GARCH
    o = 1 if model == "gjr" else 0

    dist_name = "normal" if dist == "normal" else "StudentsT"

    am = arch_model(
        r,
        mean="Constant",
        vol="GARCH",
        p=p,
        o=o,
        q=q,
        dist=dist_name,
    )
    return am


def fit_garch_window(
    window_returns: pd.Series,
    model: GarchCoreModel = "garch",
    dist: GarchDist = "normal",
):
    am = _make_arch_model(window_returns, model=model, dist=dist)
    res = am.fit(disp="off", show_warning=False)
    return res


def rolling_garch_fits(
    returns: pd.Series,
    obs_dates: Iterable[pd.Timestamp],
    window_len: int,
    model: GarchCoreModel = "garch",
    dist: GarchDist = "normal",
) -> RollingGarchResult:
    """
    Rolling GARCH/GJR fits
    """
    params_dict: Dict[pd.Timestamp, pd.Series] = {}
    sigma_last: Dict[pd.Timestamp, float] = {}

    # For GJR: collect standardized residuals per window
    std_resid_records: list[pd.DataFrame] = []

    returns = returns.sort_index()

    for date in obs_dates:
        date = pd.Timestamp(date)

        # strictly ex-ante: only use data up to (and including) 'date'
        window = returns.loc[:date].tail(window_len)
        if len(window) < window_len:
            continue

        res = fit_garch_window(window, model=model, dist=dist)

        params_dict[date] = res.params
        sigma_last[date] = float(res.conditional_volatility.iloc[-1])

        if model == "gjr":
            sr = res.std_resid.dropna()
            if not sr.empty:
                tmp = pd.DataFrame(
                    {
                        "obs_date": date,
                        "in_sample_date": sr.index,
                        "std_resid": sr.values,
                    }
                )
                std_resid_records.append(tmp)

    params_df = pd.DataFrame(params_dict).T.sort_index()

    sigma_series = pd.Series(sigma_last).sort_index()
    sigma_series.name = "sigma_last"

    if std_resid_records:
        std_resid_df = pd.concat(std_resid_records, ignore_index=True)
        std_resid_df.set_index(["obs_date", "in_sample_date"], inplace=True)
    else:
        std_resid_df = None

    return RollingGarchResult(
        params=params_df,
        last_sigma=sigma_series,
        std_resid=std_resid_df,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Rolling GARCH/GJR fits on an index using yfinance.\n\n"
            "Models:\n"
            "  garch_n : GARCH(1,1) with normal innovations\n"
            "  garch_t : GARCH(1,1) with Student-t innovations\n"
            "  gjr     : GJR(1,1) with normal innovations (for FHS)\n"
        )
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2018-01-01",
        help="Forecast start date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end",
        type=str,
        default="2024-12-31",
        help="Forecast end date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--window_years",
        type=int,
        default=5,
        help="Rolling window length in years (trading-year approx: 252 days/year).",
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default="^GSPC",
        help="Yahoo Finance ticker for underlying (default: ^GSPC).",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["garch_n", "garch_t", "gjr"],
        default="garch_n",
        help=(
            "Which model to fit:\n"
            "  garch_n : GARCH(1,1) with normal innovations\n"
            "  garch_t : GARCH(1,1) with Student-t innovations\n"
            "  gjr     : GJR(1,1) with normal innovations\n"
        ),
    )

    args = parser.parse_args()

    if args.model == "garch_n":
        core_model: GarchCoreModel = "garch"
        dist: GarchDist = "normal"
    elif args.model == "garch_t":
        core_model = "garch"
        dist = "t"
    elif args.model == "gjr":
        core_model = "gjr"
        dist = "normal"

    forecast_start = pd.to_datetime(args.start)
    forecast_end = pd.to_datetime(args.end)

    window_len_days = 252 * args.window_years

    hist_start = forecast_start - pd.DateOffset(years=args.window_years)

    print(
        f"Downloading {args.ticker} from {hist_start.date()} to {forecast_end.date()}..."
    )
    data = yf.download(
        args.ticker,
        start=hist_start,
        end=forecast_end + pd.Timedelta(days=1),
        auto_adjust=True,
        progress=False,
    )

    if data.empty:
        raise RuntimeError("No data downloaded. Check ticker and date range.")

    prices = data["Close"].dropna()
    returns = compute_log_returns(prices)

    obs_dates = returns.loc[forecast_start:forecast_end].index

    print(
        f"Fitting {args.model.upper()} with {window_len_days} trading-day window "
        f"for {len(obs_dates)} observation dates..."
    )

    rolling_result = rolling_garch_fits(
        returns=returns,
        obs_dates=obs_dates,
        window_len=window_len_days,
        model=core_model,
        dist=dist,
    )

    model_tag = args.model
    ticker_tag = args.ticker.strip("^").lower()

    out_dir = Path("artifacts") / f"{model_tag}_{ticker_tag}_{args.window_years}y"
    out_dir.mkdir(parents=True, exist_ok=True)

    params_path = out_dir / "rolling_params.csv"
    sigma_last_path = out_dir / "rolling_last_sigma.csv"

    rolling_result.params.to_csv(params_path)
    rolling_result.last_sigma.to_csv(sigma_last_path)

    print(f"Saved rolling params to:        {params_path}")
    print(f"Saved last-window sigma to:     {sigma_last_path}")

    if rolling_result.std_resid is not None:
        std_resid_path = out_dir / "rolling_std_resid.csv"
        rolling_result.std_resid.to_csv(std_resid_path)
        print(f"Saved GJR std residuals to:    {std_resid_path}")

    print("Done.")
