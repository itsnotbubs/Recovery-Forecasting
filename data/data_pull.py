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

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.config import api_key, user, password



def grab_optionsymbol(
    snapshot_date: str,
    expiry_start_date:str, 
    forward_date: str,
    option: str,
    type: str,
    strikeFrom: int,
    strikeTo: int,
    getOptionsChain: Any
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


def grab_optionprices(snapshot_date: str, optionsChain: pd.DataFrame, getMarketData: Any):
    frames = []
    for optionSymbol in optionsChain["OptionSymbol"]:
        try:
            marketData = getMarketData(
                symbol=optionSymbol,
                from_=snapshot_date,
                to=snapshot_date,
                delayBetweenRequests=1,
            )
            frames.append(marketData)
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data for {optionSymbol}: {e}")
  
    return pd.concat(frames, ignore_index=True)


def collection(dates: list[pd.Timestamp], underlying: str, getOptionsChain: Any, getMarketData: Any):
    all_frames = []
    start = time.time()
    for snapshot_date in tqdm(dates):
        forward_date = snapshot_date + pd.Timedelta(days=93)
        expiry_start_date = snapshot_date + pd.Timedelta(days=3)
        try:
            price = int(underlying_price(snapshot_date, underlying))
        except IndexError:
            print(f"[WARN] No underlying price for {underlying} on {snapshot_date.date()}, skipping.")
            continue     
        except Exception as e:
            print(f"[WARN] Error getting price for {underlying} on {snapshot_date.date()}: {e}")
            continue

        symbols = grab_optionsymbol(
            snapshot_date=snapshot_date.strftime("%Y-%m-%d"),
            expiry_start_date = expiry_start_date.strftime("%Y-%m-%d"),
            forward_date=forward_date.strftime("%Y-%m-%d"),
            option="SPX",
            type="C",
            strikeFrom=price - 100,
            strikeTo=price + 160,
            getOptionsChain=getOptionsChain,
        )

        option_NBBO = grab_optionprices(
            snapshot_date=snapshot_date.strftime("%Y-%m-%d"),
            optionsChain=symbols,
            getMarketData=getMarketData,
        )

        all_frames.append(option_NBBO)
    
    end = time.time()
    print("Data Collected in:", (end - start)/60, "minutes")

    return pd.concat(all_frames, ignore_index=True) 


def write(data: pd.DataFrame, year: str):
    path = f"data/options_data_{year}.parquet"
    data.to_parquet(path, index=False, compression="snappy")
    print(f"File has been saved for {year}")


def put_call_parity():
    # if using European option implement this
    pass


def underlying_price(date: str, underlying: str):
    return yf.download(underlying, start=date, end=date + pd.Timedelta(days=1),auto_adjust = True,progress=False)[
        "Close"
    ].values.flatten()[0]


def allfridays(year):
    return pd.date_range(start=str(year), end=str(year + 1), freq="W-FRI").tolist()



# data_range = allfridays(2019)[:2]

# retrieve = collection(
#     forward_dates=data_range,
#     underlying="^SPX",
# )

# write(retrieve, "2019")



def main():
    print(f"Loaded API_KEY (len={len(api_key) if api_key else 0})")

    ivol.setLoginParams(username={user}, password={password})

    ivol.setLoginParams(apiKey=api_key)
    
    getOptionsChain = ivol.setMethod(
        "/equities/eod/option-series-on-date"
    ) 

    getMarketData = ivol.setMethod("/equities/eod/single-stock-option")

    years = [2019,2020,2021,2022,2023,2024]

    for year in years:
        print("****Starting Api call****")
        data_range = allfridays(year)

        retrieve = collection(
        dates=data_range,
        underlying="^SPX",
        getOptionsChain=getOptionsChain,
        getMarketData=getMarketData
    )
        write(retrieve, str(year))


if __name__ == "__main__":
    main()