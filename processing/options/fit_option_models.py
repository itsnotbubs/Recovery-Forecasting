from zipfile import ZipFile

import pandas as pd

from fypy.pricing.fourier.ProjEuropeanPricer import ProjEuropeanPricer
from fypy.model.levy import VarianceGamma, MertonJD, BlackScholes
from fypy.model.sv.Bates import Bates
from fypy.termstructures.EquityForward import EquityForward
from fypy.termstructures.DiscountCurve import DiscountCurve_ConstRate

from ...utils.options.option_utils import DiscountCurve_Measured, Heston_fix
from ...utils.options.fit import fit_model


t_bills_rate = pd.read_csv('../../data/DGS1MO.csv')

t_bills_rate['observation_date'] = pd.to_datetime(t_bills_rate['observation_date'])
t_bills_rate = t_bills_rate[ # min and max of option data dates
    (t_bills_rate['observation_date'] >= pd.to_datetime('2019-01-04 00:00:00'))
    & (t_bills_rate['observation_date'] <= pd.to_datetime('2024-12-27 00:00:00'))
]
t_bills_rate = t_bills_rate.set_index('observation_date')
daily_decimal_t_bill_yield = (1 + t_bills_rate['DGS1MO'].ffill() / 100) ** (1/252) - 1


dfs = []
with ZipFile('../../data/raw_options_data.zip') as zf:
    for file in zf.namelist():
        with zf.open(file) as f:
            dfs.append(pd.read_parquet(f, engine='pyarrow'))
df = pd.concat(dfs)

df['expiration'] = pd.to_datetime(df['expiration'])
df['date'] = pd.to_datetime(df['date'])
df['ttm'] = (df['expiration'] - df['date']).dt.days


disc_curve = DiscountCurve_Measured(rates=daily_decimal_t_bill_yield)
div_disc = DiscountCurve_ConstRate(rate=0)
fwd = EquityForward(S0=0, discount=disc_curve, divDiscount=div_disc)

models = {
    'BSM': BlackScholes(forwardCurve=fwd, discountCurve=disc_curve),
    'MJD': MertonJD(forwardCurve=fwd, discountCurve=disc_curve),
    'Hes': Heston_fix(forwardCurve=fwd, discountCurve=disc_curve),
    'BJD': Bates(forwardCurve=fwd, discountCurve=disc_curve),
    'VG': VarianceGamma(forwardCurve=fwd, discountCurve=disc_curve),
}
param_orders = {
    'BSM': ['sigma'],
    'MJD': ['sigma', 'lam', 'muj', 'sigj'],
    'Hes': ['v_0', 'theta', 'kappa', 'sigma_v', 'rho'],
    'BJD': ['v_0', 'theta', 'kappa', 'sigma_v', 'rho', 'lam', 'muj', 'sigj'],
    'VG': ['sigma', 'theta', 'nu'],
}


for model_name in models.keys():
    model = models.get(model_name)

    pricer = ProjEuropeanPricer(model=model, N=2 ** 11, L=16 if model_name == 'Hes' else 12)
    
    params_fit = {} # {date: {col: val}}

    for date in df['date'].unique():
        market_df = df[df['date'] == date]
        
        params_fit[date] = fit_model(market_df, fwd, div_disc, model, model_name, pricer)

    params_fit = pd.DataFrame.from_dict(params_fit, orient='index')
    params_fit.to_csv(model_name + '_fits.csv')