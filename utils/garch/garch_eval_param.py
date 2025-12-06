from __future__ import annotations

import argparse
from math import sqrt, pi, exp, lgamma
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
import yfinance as yf

def load_artifacts(params_path: Path, sigma_path: Path) -> Tuple[pd.DataFrame, pd.Series]:
    params_df = pd.read_csv(params_path, index_col=0, parse_dates=True)
    sigma_last = pd.read_csv(sigma_path, index_col=0, parse_dates=True).iloc[:, 0]
    sigma_last.name = "sigma_last"

    idx = params_df.index.intersection(sigma_last.index)
    params_df = params_df.loc[idx].sort_index()
    sigma_last = sigma_last.loc[idx].sort_index()

    if params_df.empty:
        raise RuntimeError(
            f"No overlapping dates between params and sigma artifacts at {params_path} / {sigma_path}."
        )

    return params_df, sigma_last


def load_std_resid(std_resid_path: Path) -> pd.DataFrame:
    """
    Load GJR standardized residuals, indexed by (obs_date, in_sample_date).
    """
    df = pd.read_csv(
        std_resid_path,
        index_col=[0, 1],
        parse_dates=[0, 1],
    )
    if "std_resid" not in df.columns:
        raise RuntimeError(f"'std_resid' column not found in {std_resid_path}")
    return df


def download_prices(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    data = yf.download(
        ticker,
        start=start,
        end=end + pd.Timedelta(days=1),
        auto_adjust=True,
        progress=False,
    )
    if data.empty:
        raise RuntimeError(f"No price data downloaded for {ticker}.")
    prices = data["Close"].dropna()
    prices.name = "Close"
    return prices


def forecast_garch_variance_path(
    omega: float,
    alpha: float,
    beta: float,
    sigma0: float,
    horizon: int,
) -> np.ndarray:
    phi = alpha + beta
    var_path = np.zeros(horizon, dtype=float)
    var_curr = sigma0**2

    for h in range(horizon):
        var_curr = omega + phi * var_curr
        var_path[h] = var_curr

    return var_path


def normal_pdf(x: float, mean: float, var: float, eps: float = 1e-16) -> float:
    var = max(var, eps)
    std = sqrt(var)
    z = (x - mean) / std
    return (1.0 / (std * sqrt(2.0 * pi))) * exp(-0.5 * z * z)


def student_t_pdf(
    x: float, mean: float, var: float, nu: float, eps: float = 1e-16
) -> float:
    nu = float(nu)
    if nu <= 2:
        var = max(var, eps)
        return normal_pdf(x, mean, var)

    var = max(var, eps)
    s = sqrt(var * (nu - 2.0) / nu)

    z = (x - mean) / s
    ln_coef = (
        lgamma((nu + 1.0) / 2.0) - lgamma(nu / 2.0) - 0.5 * np.log(nu * pi) - np.log(s)
    )
    ln_kernel = -0.5 * (nu + 1.0) * np.log(1.0 + (z * z) / nu)
    ln_f = ln_coef + ln_kernel
    return float(max(exp(ln_f), eps))


def evaluate_loglik_parametric(
    model_name: str,
    params_df: pd.DataFrame,
    sigma_last: pd.Series,
    prices: pd.Series,
    horizon_days: int,
    dist: str,
    snapshot_dates: pd.DatetimeIndex,
) -> dict:
    prices = prices.sort_index()
    params_df = params_df.sort_index()
    sigma_last = sigma_last.sort_index()

    snapshot_dates = pd.DatetimeIndex(snapshot_dates)
    valid_dates = params_df.index.intersection(snapshot_dates)
    params_df = params_df.loc[valid_dates]
    sigma_last = sigma_last.loc[valid_dates]

    total_loglik = 0.0
    num_obs_used = 0
    price_index = prices.index

    for t in params_df.index:
        if t not in price_index:
            continue
        pos = price_index.get_loc(t)
        future_pos = pos + horizon_days
        if future_pos >= len(price_index):
            continue

        t_h = price_index[future_pos]
        S0 = float(prices.loc[t])
        S_h = float(prices.loc[t_h])

        if S0 <= 0.0 or S_h <= 0.0:
            continue

        realized_ret = 100.0 * np.log(S_h / S0)

        row = params_df.loc[t]
        try:
            mu = float(row["mu"])
            omega = float(row["omega"])
            alpha = float(row["alpha[1]"])
            beta = float(row["beta[1]"])
        except KeyError as e:
            raise KeyError(
                f"[{model_name}] Missing required parameter in params_df columns: {e}"
            )

        sigma0 = float(sigma_last.loc[t])

        var_path = forecast_garch_variance_path(
            omega=omega,
            alpha=alpha,
            beta=beta,
            sigma0=sigma0,
            horizon=horizon_days,
        )
        var_sum = float(var_path.sum())
        mean_sum = float(horizon_days * mu)

        if dist == "normal":
            f_val = normal_pdf(realized_ret, mean_sum, var_sum)
        elif dist == "t":
            if "nu" not in row.index:
                raise KeyError(
                    f"[{model_name}] GARCH-t selected but 'nu' not found in params_df columns."
                )
            nu = float(row["nu"])
            f_val = student_t_pdf(realized_ret, mean_sum, var_sum, nu=nu)
        else:
            raise ValueError(f"[{model_name}] Unknown dist: {dist}")

        total_loglik += float(np.log(f_val))
        num_obs_used += 1

    avg_loglik = total_loglik / num_obs_used if num_obs_used > 0 else np.nan

    return {
        "model": model_name,
        "horizon_days": horizon_days,
        "num_obs": num_obs_used,
        "total_loglik": total_loglik,
        "avg_loglik": avg_loglik,
    }


def simulate_gjr_paths_cumret(
    mu: float,
    omega: float,
    alpha: float,
    gamma: float,
    beta: float,
    sigma0: float,
    resid_pool: np.ndarray,
    horizon: int,
    num_sims: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if resid_pool.size == 0:
        raise ValueError("Empty resid_pool in simulate_gjr_paths_cumret.")

    sigma = np.full(num_sims, sigma0, dtype=float)
    cum_ret = np.zeros(num_sims, dtype=float)

    for _ in range(horizon):
    
        z = rng.choice(resid_pool, size=num_sims, replace=True)
        eps = sigma * z
        r_step = mu + eps
        cum_ret += r_step

        indicator = (eps < 0.0).astype(float)
        sigma2_next = omega + alpha * eps**2 + gamma * eps**2 * indicator + beta * sigma**2
        sigma = np.sqrt(np.clip(sigma2_next, 1e-12, None))

    return cum_ret


def kde_gaussian_density(x: float, samples: np.ndarray, eps: float = 1e-300) -> float:
    n = samples.size
    if n <= 1:
        return eps

    std = samples.std(ddof=1)
    if std <= 0:
        return eps

    h = 1.06 * std * (n ** (-1.0 / 5.0))
    if h <= 0:
        return eps

    u = (x - samples) / h
    kern = np.exp(-0.5 * u**2) / np.sqrt(2.0 * pi)
    f_hat = kern.mean() / h
    return float(max(f_hat, eps))


def evaluate_loglik_fhs_gjr(
    model_name: str,
    params_df: pd.DataFrame,
    sigma_last: pd.Series,
    std_resid_df: pd.DataFrame,
    prices: pd.Series,
    horizon_days: int,
    snapshot_dates: pd.DatetimeIndex,
    num_sims: int,
    rng: np.random.Generator,
) -> dict:
    """
    Evaluate GJR forecasts via Filtered Historical Simulation (FHS):
    """
    prices = prices.sort_index()
    params_df = params_df.sort_index()
    sigma_last = sigma_last.sort_index()

    snapshot_dates = pd.DatetimeIndex(snapshot_dates)
    valid_dates = params_df.index.intersection(snapshot_dates)
    params_df = params_df.loc[valid_dates]
    sigma_last = sigma_last.loc[valid_dates]

    total_loglik = 0.0
    num_obs_used = 0
    price_index = prices.index

    for t in params_df.index:
        if t not in price_index:
            continue
        pos = price_index.get_loc(t)
        future_pos = pos + horizon_days
        if future_pos >= len(price_index):
            continue

        t_h = price_index[future_pos]
        S0 = float(prices.loc[t])
        S_h = float(prices.loc[t_h])

        if S0 <= 0.0 or S_h <= 0.0:
            continue

        realized_ret = 100.0 * np.log(S_h / S0)

        row = params_df.loc[t]
        try:
            mu = float(row["mu"])
            omega = float(row["omega"])
            alpha = float(row["alpha[1]"])
            gamma = float(row["gamma[1]"])
            beta = float(row["beta[1]"])
        except KeyError as e:
            raise KeyError(
                f"[{model_name}] Missing required GJR parameter in params_df columns: {e}"
            )

        sigma0 = float(sigma_last.loc[t])

        try:
            resid_block = std_resid_df.xs(t, level="obs_date")["std_resid"].values
        except KeyError:
            continue

        if resid_block.size < 30:
            continue

        cum_rets = simulate_gjr_paths_cumret(
            mu=mu,
            omega=omega,
            alpha=alpha,
            gamma=gamma,
            beta=beta,
            sigma0=sigma0,
            resid_pool=resid_block,
            horizon=horizon_days,
            num_sims=num_sims,
            rng=rng,
        )

        f_hat = kde_gaussian_density(realized_ret, cum_rets)
        total_loglik += float(np.log(f_hat))
        num_obs_used += 1

    avg_loglik = total_loglik / num_obs_used if num_obs_used > 0 else np.nan

    return {
        "model": model_name,
        "horizon_days": horizon_days,
        "num_obs": num_obs_used,
        "total_loglik": total_loglik,
        "avg_loglik": avg_loglik,
    }



def find_model_artifacts(artifact_root: Path) -> Dict[str, Dict[str, object]]:
    model_paths: Dict[str, Dict[str, object]] = {}
    for entry in artifact_root.iterdir():
        if not entry.is_dir():
            continue

        params_path = entry / "rolling_params.csv"
        sigma_path = entry / "rolling_last_sigma.csv"
        if not (params_path.exists() and sigma_path.exists()):
            continue

        dirname = entry.name 
        dlow = dirname.lower()

        if "garch_n" in dlow:
            model_type = "garch_n"
            dist = "normal"
        elif "garch_t" in dlow:
            model_type = "garch_t"
            dist = "t"
        elif "gjr" in dlow:
            model_type = "gjr"
            dist = "normal"  
        else:
            print(f"Skipping directory with unknown model type: {entry}")
            continue

        std_resid_path = entry / "rolling_std_resid.csv"
        has_std_resid = std_resid_path.exists()

        model_paths[dirname] = {
            "params": params_path,
            "sigma": sigma_path,
            "dist": dist,
            "dir": entry,
            "model_type": model_type,
            "std_resid_path": std_resid_path if has_std_resid else None,
        }

    if not model_paths:
        raise RuntimeError(f"No valid model artifacts found under {artifact_root}")

    return model_paths


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate GARCH-type density forecasts for ALL models' artifacts in a root directory.\n"
            "GARCH-N and GARCH-t use parametric multi-step distributions.\n"
            "GJR uses Filtered Historical Simulation (FHS) with KDE density.\n"
            "Evaluation is only on snapshot dates (weekly Fridays)."
        )
    )
    parser.add_argument(
        "--artifact-root",
        type=str,
        required=True,
        help="Root directory containing per-model artifact subdirectories.",
    )
    parser.add_argument(
        "--horizons",
        type=str,
        default="5,10,21,42",
        help=(
            "Comma-separated forecast horizons in trading days. "
            "Default '5,10,21,42' ≈ 1w, 2w, 1m, 2m."
        ),
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default="^GSPC",
        help="Yahoo Finance ticker for underlying (default: ^GSPC).",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Optional start date (YYYY-MM-DD) to restrict evaluation.",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="Optional end date (YYYY-MM-DD) to restrict evaluation.",
    )
    parser.add_argument(
        "--num-sims",
        type=int,
        default=5000,
        help="Number of FHS simulation paths per snapshot for GJR models.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for FHS simulation.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to write a CSV summary over all models and horizons.",
    )

    args = parser.parse_args()
    artifact_root = Path(args.artifact_root)

    # Parse horizons
    horizon_list: List[int] = [
        int(h.strip()) for h in args.horizons.split(",") if h.strip()
    ]
    if not horizon_list:
        raise ValueError("No valid horizons provided.")

    rng = np.random.default_rng(args.seed)

    # Discover models
    model_artifacts = find_model_artifacts(artifact_root)

    # Load artifacts, apply date filters, and track global price range
    model_data: Dict[str, Dict[str, object]] = {}
    global_min_date = None
    global_max_date = None

    for model_tag, paths in model_artifacts.items():
        params_df, sigma_last = load_artifacts(paths["params"], paths["sigma"])

        if args.start_date is not None:
            start = pd.to_datetime(args.start_date)
            params_df = params_df.loc[params_df.index >= start]
            sigma_last = sigma_last.loc[sigma_last.index >= start]
        if args.end_date is not None:
            end = pd.to_datetime(args.end_date)
            params_df = params_df.loc[params_df.index <= end]
            sigma_last = sigma_last.loc[sigma_last.index <= end]

        if params_df.empty:
            print(f"[{model_tag}] No artifact dates left after applying start/end filters. Skipping.")
            continue

        snapshot_idx = params_df.index[params_df.index.weekday == 4]  # Fridays
        if len(snapshot_idx) == 0:
            print(f"[{model_tag}] No Friday snapshot dates in the filtered artifact range. Skipping.")
            continue

        min_date = snapshot_idx.min()
        max_date = snapshot_idx.max()

        if (global_min_date is None) or (min_date < global_min_date):
            global_min_date = min_date
        if (global_max_date is None) or (max_date > global_max_date):
            global_max_date = max_date

        # Load std_resid if GJR and available
        std_resid_df = None
        if paths["model_type"] == "gjr" and paths["std_resid_path"] is not None:
            std_resid_df = load_std_resid(paths["std_resid_path"])

        model_data[model_tag] = {
            "params_df": params_df,
            "sigma_last": sigma_last,
            "snapshot_idx": snapshot_idx,
            "dist": paths["dist"],
            "model_type": paths["model_type"],
            "std_resid_df": std_resid_df,
        }

    if not model_data:
        raise RuntimeError("No models left after filtering / snapshot selection.")

    assert global_min_date is not None and global_max_date is not None
    max_horizon = max(horizon_list)
    price_start = global_min_date
    price_end = global_max_date + pd.Timedelta(days=max_horizon * 3)

    print(
        f"Downloading {args.ticker} prices from {price_start.date()} "
        f"to {price_end.date()}..."
    )
    prices = download_prices(args.ticker, start=price_start, end=price_end)

    # Evaluate each model x horizon
    summaries: List[dict] = []

    print("\n=== GARCH-Type Log-Likelihood Summary (Weekly Snapshots) ===")
    print(f"Horizons (trading days): {horizon_list}\n")

    for horizon_days in horizon_list:
        print(f"--- Horizon = {horizon_days} trading days ---")
        for model_tag, data in model_data.items():
            params_df = data["params_df"]
            sigma_last = data["sigma_last"]
            snapshot_idx = data["snapshot_idx"]
            dist = data["dist"]
            model_type = data["model_type"]
            std_resid_df = data["std_resid_df"]

            if model_type in ("garch_n", "garch_t"):
                summary = evaluate_loglik_parametric(
                    model_name=model_tag,
                    params_df=params_df,
                    sigma_last=sigma_last,
                    prices=prices,
                    horizon_days=horizon_days,
                    dist=dist,
                    snapshot_dates=snapshot_idx,
                )
            elif model_type == "gjr":
                if std_resid_df is None:
                    print(f"[{model_tag}] No std_resid artifact for GJR; skipping FHS.")
                    continue
                summary = evaluate_loglik_fhs_gjr(
                    model_name=model_tag,
                    params_df=params_df,
                    sigma_last=sigma_last,
                    std_resid_df=std_resid_df,
                    prices=prices,
                    horizon_days=horizon_days,
                    snapshot_dates=snapshot_idx,
                    num_sims=args.num_sims,
                    rng=rng,
                )
            else:
                continue

            summaries.append(summary)

            print(
                f"Model: {summary['model']:25s} | "
                f"num_obs = {summary['num_obs']:4d} | "
                f"total_ll = {summary['total_loglik']:.4f} | "
                f"avg_ll = {summary['avg_loglik']:.6f}"
            )
        print()

    if args.output is not None:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(summaries).to_csv(out_path, index=False)
        print(f"\nSaved summary CSV to: {out_path}")


if __name__ == "__main__":
    main()
