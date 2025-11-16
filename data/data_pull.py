import ivolatility as ivol
from pathlib import Path
import sys
import pandas as pd
import requests
import yfinance as yf
import os
import time
from tqdm import tqdm
from typing import Any
import random
from requests.exceptions import HTTPError, RequestException

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.config import api_key, user, password


def grab_optionsymbol(
    snapshot_date: str,
    expiry_start_date: str,
    forward_date: str,
    option: str,
    type: str,
    strikeFrom: int,
    strikeTo: int,
    getOptionsChain: Any,
):
    optionsChain = getOptionsChain(
        symbol=option,
        expFrom=expiry_start_date,
        expTo=forward_date,
        strikeFrom=strikeFrom,
        strikeTo=strikeTo,
        callPut=type,
        date=snapshot_date,
    )

    return optionsChain


def grab_optionprices(
    snapshot_date: str, optionsChain: pd.DataFrame, getMarketData: Any
):
    frames = []
    for optionSymbol in optionsChain["OptionSymbol"]:
        max_retries = 4
        for attempt in range(max_retries):
            try:
                marketData = getMarketData(
                    symbol=optionSymbol,
                    from_=snapshot_date,
                    to=snapshot_date,
                    delayBetweenRequests=3,
                )

                frames.append(marketData)
                break

            except HTTPError as e:
                msg = str(e)
                if "429" in msg or "Too Many Requests" in msg:
                    wait = 2 ** attempt + random.random()
                    print(
                        f"[WARN] 429 for {optionSymbol} on {snapshot_date} "
                        f"(retry {attempt + 1}/{max_retries}) → waiting {wait:.2f}s"
                    )
                    time.sleep(wait)
                    continue  
                else:
                    print(f"[WARN] HTTP error for {optionSymbol} on {snapshot_date}: {e}")
                    break  

            except RequestException as e:
                print(f"[WARN] Request error for {optionSymbol}: {e}")
                break

            except Exception as e:
                print(f"[WARN] Unexpected error for {optionSymbol}: {e}")
                break

    if not frames:
        print(f"[WARN] No market data returned on {snapshot_date}.")
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def collection(
    dates: list[pd.Timestamp], underlying: str, getOptionsChain: Any, getMarketData: Any
):
    all_frames = []
    start = time.time()

    for snapshot_date in tqdm(dates):
        forward_date = snapshot_date + pd.Timedelta(days=93)
        expiry_start_date = snapshot_date + pd.Timedelta(days=3)

        # Underlying price
        try:
            price = int(underlying_price(snapshot_date, underlying))
        except IndexError:
            print(
                f"[WARN] No underlying price for {underlying} on {snapshot_date.date()}, skipping."
            )
            continue
        except Exception as e:
            print(
                f"[WARN] Error getting price for {underlying} on {snapshot_date.date()}: {e}"
            )
            continue

        # Option symbols
        try:
            symbols = grab_optionsymbol(
                snapshot_date=snapshot_date.strftime("%Y-%m-%d"),
                expiry_start_date=expiry_start_date.strftime("%Y-%m-%d"),
                forward_date=forward_date.strftime("%Y-%m-%d"),
                option="SPX",
                type="C",
                strikeFrom=price - 100,
                strikeTo=price + 150,
                getOptionsChain=getOptionsChain,
            )
        except Exception as e:
            print(f"[WARN] Error getting option chain on {snapshot_date.date()}: {e}")
            continue

        # Market data 
        try:
            option_NBBO = grab_optionprices(
                snapshot_date=snapshot_date.strftime("%Y-%m-%d"),
                optionsChain=symbols,
                getMarketData=getMarketData,
            )
        except RequestException as e:
            print(f"[WARN] Market data request failed on {snapshot_date.date()}: {e}")
            continue
        except Exception as e:
            print(
                f"[WARN] Unexpected error in grab_optionprices on {snapshot_date.date()}: {e}"
            )
            continue

        all_frames.append(option_NBBO)

    end = time.time()
    print("Data Collected in:", (end - start) / 60, "minutes")

    if not all_frames:
        print("[WARN] No data collected, returning empty DataFrame.")
        return pd.DataFrame() 

    return pd.concat(all_frames, ignore_index=True)


def write(data: pd.DataFrame, year: str):
    path = f"data/options_data_{year}.parquet"
    data.to_parquet(path, index=False, compression="snappy")
    print(f"File has been saved for {year}")


def underlying_price(date: str, underlying: str):
    return yf.download(
        underlying,
        start=date,
        end=date + pd.Timedelta(days=1),
        auto_adjust=True,
        progress=False,
    )["Close"].values.flatten()[0]


def allfridays(year):
    return pd.date_range(start=str(year), end=str(year + 1), freq="W-FRI").tolist()


def main():
    print(f"Loaded API_KEY (len={len(api_key) if api_key else 0})")

    ivol.setLoginParams(username={user}, password={password})

    ivol.setLoginParams(apiKey=api_key)

    getOptionsChain = ivol.setMethod("/equities/eod/option-series-on-date")

    getMarketData = ivol.setMethod("/equities/eod/single-stock-option")

    years = [2019, 2020, 2021, 2022, 2023, 2024]

    for year in years:
        print("****Starting Api call****")
        data_range = allfridays(year)

        retrieve = collection(
            dates=data_range,
            underlying="^SPX",
            getOptionsChain=getOptionsChain,
            getMarketData=getMarketData,
        )
        write(retrieve, str(year))


if __name__ == "__main__":
    main()
