import ivolatility as ivol
from pathlib import Path
import sys
import pandas as pd
import requests

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
    getOptionsChain = ivol.setMethod("/equities/eod/option-series-on-date") # can be moved to main or collector
    optionsChain = getOptionsChain(
        symbol=f"{option}",
        expTo=f"{forward_date}",
        strikeFrom=strikeFrom,
        strikeTo=strikeTo,
        callPut=f"{type}",
        date=f"{snapshot_date}",
    )

    return optionsChain


def grab_optionprices(
    snapshot_date: str,
    optionsChain: pd.DataFrame
):
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
    

def collection():
    #want to essentially loop over dates for a given underlying depending on a 
    # the specified forward date ie in the paper its 28 days or forward month 
    # this colleciton should just be a merge between calls and puts
    pass

def write():
    #Takes collection and writes into a csv to ensure if anything breaks along the lines
    # we have the data and can continue as needed 
    # 
    #  


def put_call_parity():
    # if using European option implement this
    pass



option_data = grab_optionsymbol(option="SPX",
    forward_date="2025-11-06",
    strikeFrom=6700,
    strikeTo=6750,
    type="C",
    snapshot_date="2025-11-05")


print(grab_optionprices(snapshot_date="2025-11-05", optionsChain=option_data))