import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ...utils.risk_neutral_dist.GilPeleazEuropeanCDF import GilPeleazEuropeanPricer_cdf
from ...utils.risk_neutral_dist.functions import get_pdf_RV
from .constants import models, IV_param_locs

model_name = 'Hes'

params_df = pd.read_csv(
    model_name + '_fits_51_PE_multiL_linear_softl1.csv',
    index_col=0
)
params_df.index = pd.to_datetime(params_df.index)

df_RV = pd.read_csv(model_name + '_RV_51_PE_multiL_linear_softl1.csv', index_col=0)
df_RV['date'] = pd.to_datetime(df_RV['date'])


date = params_df.index[50]

df_RV_date = df_RV[df_RV['date'] == date]
params = params_df.loc[date].to_numpy()[:-2]
S0 = params_df.loc[date].to_numpy()[-1]
model = models[model_name]

model.discountCurve.current_time = date
model.forwardCurve.discountCurve.current_time = date
model.forwardCurve._S0 = S0

cdf_pricer = GilPeleazEuropeanPricer_cdf(model=model)
strikes_ = np.linspace(0.7, 1.5, 50) * params_df.loc[date]['env: S0']
ttms = [28]
EPS = 1e-10

risk_neutral_pdf = get_pdf_RV(
    cdf_pricer, ttms, strikes_, np.ones(len(strikes_), dtype=bool), params,
    IV_param_locs[model_name], df_RV_date, EPS=EPS
)

plt.plot(strikes_ / S0, risk_neutral_pdf[0,:] * strikes_ / S0, label=str(ttms[0]) + ' dte')

plt.legend()
plt.xlabel(r'rel strike, $K/S_0$')
plt.ylabel('risk_neutral_pdf')
plt.show()