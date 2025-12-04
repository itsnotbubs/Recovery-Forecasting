from zipfile import ZipFile
from tqdm import tqdm

import pandas as pd
import numpy as np

from ...utils.options.option_utils import ProjEuropeanPricer_fix
from ...utils.options.fit import fit_model, initial_params
from .constants import models, fwd, div_disc


dfs = []
with ZipFile('../../data/raw_options_data.zip') as zf:
    for file in zf.namelist():
        with zf.open(file) as f:
            dfs.append(pd.read_parquet(f, engine='pyarrow'))
df = pd.concat(dfs)

df['expiration'] = pd.to_datetime(df['expiration'])
df['date'] = pd.to_datetime(df['date'])
df['ttm'] = (df['expiration'] - df['date']).dt.days


for model_name in models.keys():
    model = models.get(model_name)
    guess = np.array(list(initial_params[model_name].values()), dtype=np.float64)

    pricers = [
        ProjEuropeanPricer_fix(model=model, N=2 ** 9, L=14, order=1),
        ProjEuropeanPricer_fix(model=model, N=2 ** 9, L=13, order=1),
        ProjEuropeanPricer_fix(model=model, N=2 ** 9, L=12, order=1),
        ProjEuropeanPricer_fix(model=model, N=2 ** 9, L=11, order=1)
    ]
    
    params_fit = {} # {date: {col: val}}

    for date in tqdm(df['date'].unique()):
        market_df = df[df['date'] == date]
        
        try:
            params_fit[date], guess = fit_model(market_df, fwd, div_disc, model, model_name, pricers, guess)
        except Exception as e:
            # failed to converge
            pass

    params_fit = pd.DataFrame.from_dict(params_fit, orient='index')
    params_fit.to_csv(model_name + '_fits_51_PE_multiL_linear_softl1.csv')