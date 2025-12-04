from zipfile import ZipFile
from tqdm import tqdm

import pandas as pd

from fypy.pricing.fourier.CarrMadanEuropeanPricer import CarrMadanEuropeanPricer

from ...utils.options.fit import fit_model_residual_vol
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

top_n_volume = 51
for model_name in models.keys():
    params_df = pd.read_csv(
        model_name + '_fits_51_PE_multiL_linear_softl1.csv',
        index_col=0
    )
    params_df.index = pd.to_datetime(params_df.index)

    model = models[model_name]
    pricer = CarrMadanEuropeanPricer(model=model, N=2 ** 11)
    
    rv_data = []
    for date in tqdm(params_df.index):
        market_df = df[df['date'] == date]
        guess = params_df.loc[date].to_numpy()[:-2]
        
        rv_data.extend(fit_model_residual_vol(
            market_df, fwd, div_disc, model, model_name, pricer, guess,
            top_n_volume=top_n_volume
        ))
    pd.DataFrame(rv_data).to_csv(model_name + '_RV_51_PE_multiL_linear_softl1.csv')