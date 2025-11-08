import ivolatility as ivol
from pathlib import Path
import sys
import pandas as pd
import requests
import yfinance as yf
import os

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.config import api_key, user, password


print(f"Loaded API_KEY (len={len(api_key) if api_key else 0})")

ivol.setLoginParams(username={user}, password={password})

ivol.setLoginParams(apiKey=api_key)


def grab_optionsymbol(
    snapshot_date: str,
    forward_date: str,
    option: str,
    type: str,
    strikeFrom: int,
    strikeTo: int,
):
    getOptionsChain = ivol.setMethod(
        "/equities/eod/option-series-on-date"
    )  # can be moved to main or collector
    optionsChain = getOptionsChain(
        symbol=f"{option}",
        expFrom=f"{forward_date}",
        expTo=f"{forward_date}",
        strikeFrom=strikeFrom,
        strikeTo=strikeTo,
        callPut=f"{type}",
        date=f"{snapshot_date}",
    )

    return optionsChain


def grab_optionprices(snapshot_date: str, optionsChain: pd.DataFrame):
    allData = pd.DataFrame()
    getMarketData = ivol.setMethod("/equities/eod/single-stock-option")

    for optionSymbol in optionsChain["OptionSymbol"]:
        try:
            marketData = getMarketData(
                symbol=optionSymbol,
                from_=f"{snapshot_date}",
                to=f"{snapshot_date}",
                delayBetweenRequests=1,
            )
            allData = pd.concat([allData, marketData], axis=0)
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data for {optionSymbol}: {e}")
    return allData


def collection(forward_dates: [], underlying: str):
    # want to essentially loop over dates for a given underlying depending on a
    # the specified forward date ie in the paper its 28 days or forward month
    # this colleciton should just be a merge between calls and puts
    data = pd.DataFrame()
    for forward_date in forward_dates:
        print(forward_date)
        snapshot_date = forward_date - pd.Timedelta(days=7)
        print(snapshot_date)
        price = int(underlying_price(snapshot_date, underlying))
        symbols = grab_optionsymbol(
            snapshot_date=snapshot_date.strftime("%Y-%m-%d"),
            forward_date=forward_date.strftime("%Y-%m-%d"),
            option="SPX",
            type="C",
            strikeFrom=price - 30,
            strikeTo=price + 100,
        )
        option_NBBO = grab_optionprices(
            snapshot_date=snapshot_date.strftime("%Y-%m-%d"), optionsChain=symbols
        )

        data = pd.concat([data, option_NBBO], axis=0)
    return data


def write(data: pd.DataFrame, year: str):
    # Takes collection and writes into a csv to ensure if anything breaks along the lines
    # we have the data and can continue as needed
    if os.path.exists("data/raw_data_" + year):
        return data.to_csv("data/raw_data_" + year, mode="a", header=False, index=False)
    else:
        return data.to_csv("data/raw_data_" + year, mode="w", header=False)


def put_call_parity():
    # if using European option implement this
    pass


def underlying_price(date: str, underlying: str):
    return yf.download(underlying, start=date, end=date + pd.Timedelta(days=1))[
        "Close"
    ].values.flatten()[0]


def allfridays(year):
    return pd.date_range(start=str(year), end=str(year + 1), freq="W-FRI").tolist()


# option_data = grab_optionsymbol(
#     option="SPX",
#     forward_date="2025-11-07",
#     strikeFrom=6700,
#     strikeTo=6750,
#     type="C",
#     snapshot_date="2025-10-31",
# )

# test = grab_optionprices(snapshot_date="2025-10-31", optionsChain=option_data)

# print(test)
# print(print(test['date']),test['expiration'])
data_range = allfridays(2019)

retrieve = collection(
    forward_dates=data_range,
    underlying="^SPX",
)

write(retrieve, "2019")
