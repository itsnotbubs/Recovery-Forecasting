import ivolatility as ivol
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.config import api_key, user, password


print(f"Loaded API_KEY (len={len(api_key) if api_key else 0})")

ivol.setLoginParams(username={user}, password={password})

ivol.setLoginParams(apiKey=api_key)


getOptionsChain = ivol.setMethod('/equities/option-series')
optionsChain = getOptionsChain(symbol='SPX', expFrom='2021-12-23', expTo='2025-12-23', strikeFrom=3000, strikeTo=3010, callPut='C')
print(optionsChain)
#optionsChain.to_csv('optionsChain.csv', header=True) 

getMarketData = ivol.setMethod('/equities/rt/options-rawiv')
marketData = getMarketData(symbols=optionsChain['SPXW  251212C03000000'])
print(marketData)
