import pandas as pd

from fypy.model.levy import VarianceGamma, MertonJD, BlackScholes
from fypy.model.sv.Bates import Bates
from fypy.termstructures.EquityForward import EquityForward
from fypy.termstructures.DiscountCurve import DiscountCurve_ConstRate

from ...utils.options.option_utils import DiscountCurve_Measured, Heston_fix


t_bills_rate = pd.read_csv('../../data/DGS1MO.csv')

t_bills_rate['observation_date'] = pd.to_datetime(t_bills_rate['observation_date'])
t_bills_rate = t_bills_rate[ # min and max of option data dates
    (t_bills_rate['observation_date'] >= pd.to_datetime('2019-01-04 00:00:00'))
    & (t_bills_rate['observation_date'] <= pd.to_datetime('2024-12-27 00:00:00'))
]
t_bills_rate = t_bills_rate.set_index('observation_date')
daily_decimal_t_bill_yield = (1 + t_bills_rate['DGS1MO'].ffill() / 100) ** (1/252) - 1


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
IV_param_locs = {
    'BSM': 0,
    'MJD': 0,
    'Hes': 0,
    'BJD': 0,
    'VG': 0,
}