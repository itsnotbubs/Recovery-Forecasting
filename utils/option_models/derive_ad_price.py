import numpy as np
import pandas as pd
from itertools import chain
import matplotlib.pyplot as plt
from fypy.model.levy import BlackScholes, MertonJD, VarianceGamma
from fypy.model.sv.Bates import Bates
from fypy.model.sv.Heston import Heston
from fypy.termstructures.EquityForward import EquityForward
from fypy.termstructures.DiscountCurve import DiscountCurve_ConstRate
from fypy.market.MarketSurface import MarketSlice, MarketSurface
from fypy.fit.Targets import Targets
from fypy.fit.Calibrator import Calibrator, LeastSquares
from utils.risk_neutral_dist.ProjEuropeanPricerCDF import ProjEuropeanPricer_cdf
from utils.risk_neutral_dist.HilbertEuropeanCDF import HilbertEuropeanPricer_cdf
from utils.risk_neutral_dist.functions import get_cdf, get_pdf, get_discrete_pdf


MODELS = {
    # model name with initial guesses for parameters
    'BlackScholes': {'sigma': 0.2},
    # 'Bates': {
    #     'v_0': 0.04,
    #     'theta': 0.04,
    #     'kappa': 1.0,
    #     'sigma_v': 0.2,
    #     'rho': -0.5,
    #     'lam': 0.1,
    #     'muj': -0.1,
    #     'sigj': 0.2,
    # },
    'Heston': {
        'v_0': 0.04,
        'theta': 0.04,
        'kappa': 0.1,
        'sigma_v': 0.5,
        'rho': -0.5,
    },
    'MertonJD': {
        'sigma': 0.2,
        'lam': 0.1,
        'muj': -0.1,
        'sigj': 0.2,
    },
    'VarianceGamma': {
        'sigma': 0.2,
        'theta': 0.04,
        'nu': 0.2,
    }
}


class OptionModel:
    """
    Wrapper class for fypy option pricing models.
    Encapsulates model initialization with term structures and parameter management.
    """

    def __init__(self, model_name='BlackScholes', initial_params=None, S0=None, rate=None):
        """
        Initialize OptionModel

        Parameters:
        - model_name: Name of model ('BlackScholes', 'Heston', 'MertonJD', etc.)
        - initial_params: Dict of parameter names -> initial values
        - S0: Spot price (used to create forward curve)
        - rate: Risk-free rate annualized (used to create discount curve)
        """
        self.model_name = model_name
        self.initial_params = initial_params or {}
        self.S0 = S0
        self.rate = rate
        self.model_instance = None

        # Create term structures if S0 and rate provided
        if S0 is not None and rate is not None:
            self._create_model()

    def _create_model(self):
        """Create the underlying fpfy model instance with term structures"""
        # Create term structures
        self.disc_curve = DiscountCurve_ConstRate(rate=self.rate)
        self.div_disc = DiscountCurve_ConstRate(rate=0)  # No dividends
        self.fwd_curve = EquityForward(S0=self.S0, discount=self.disc_curve, divDiscount=self.div_disc)

        # Instantiate model based on model_name
        if self.model_name == 'BlackScholes':
            self.model_instance = BlackScholes(
                # static volatility
                sigma=self.initial_params.get('sigma', 0.2),
                forwardCurve=self.fwd_curve,
                discountCurve=self.disc_curve
            )
        elif self.model_name == 'Heston':
            # volatiilty is also stochastic
            self.model_instance = Heston(
                v_0=self.initial_params.get('v_0', 0.04),
                theta=self.initial_params.get('theta', 0.04),
                kappa=self.initial_params.get('kappa', 0.1),
                sigma_v=self.initial_params.get('sigma_v', 0.5),
                rho=self.initial_params.get('rho', -0.5),
                forwardCurve=self.fwd_curve,
                discountCurve=self.disc_curve
            )
        elif self.model_name == 'MertonJD':
            # volatility is static, but there are jumps
            self.model_instance = MertonJD(
                sigma=self.initial_params.get('sigma', 0.2),
                lam=self.initial_params.get('lam', 0.1),
                muj=self.initial_params.get('muj', -0.1),
                sigj=self.initial_params.get('sigj', 0.2),
                forwardCurve=self.fwd_curve,
                discountCurve=self.disc_curve
            )
        elif self.model_name == 'Bates':
            # volatiilty is stochastic, and there are jumps
            self.model_instance = Bates(
                v_0=self.initial_params.get('v_0', 0.04),
                theta=self.initial_params.get('theta', 0.04),
                kappa=self.initial_params.get('kappa', 1.0),
                sigma_v=self.initial_params.get('sigma_v', 0.2),
                rho=self.initial_params.get('rho', -0.5),
                lam=self.initial_params.get('lam', 0.1),
                muj=self.initial_params.get('muj', -0.1),
                sigj=self.initial_params.get('sigj', 0.2),
                forwardCurve=self.fwd_curve,
                discountCurve=self.disc_curve
            )
        elif self.model_name == 'VarianceGamma':
            # pure jump model
            self.model_instance = VarianceGamma(
                sigma=self.initial_params.get('sigma', 0.2),
                theta=self.initial_params.get('theta', 0.04),
                nu=self.initial_params.get('nu', 0.2),
                forwardCurve=self.fwd_curve,
                discountCurve=self.disc_curve
            )
        else:
            raise ValueError(f"Model '{self.model_name}' not implemented")

    def __repr__(self):
        return f"OptionModel(name='{self.model_name}', params={self.initial_params}, S0={self.S0}, rate={self.rate})"


def calibrate_model_for_date(date_data, option_model, top_n_volume=50):
    """
    Generalized calibration function for any pricing model with arbitrary parameters

    Parameters:
    - date_data: DataFrame with columns [strike, expiration, price, Call/Put, T]
    - model: Initialized fypy model instance (e.g., BlackScholes, Heston, MertonJD, etc.)
        - model contains S_0, rate, and initial parameters

    Returns:
    - calibrated_params: Dictionary with calibrated parameter names as keys
    - rmse: Fit error (RMSE)
    - model_updated: Updated model instance with calibrated parameters
    """
    try:
        # Create pricer
        if option_model.model_name in ('Bates'):
            pricer_params = {'N': 2**11, 'Nh': 0.9}
        elif option_model.model_name in ('VarianceGamma'):
            pricer_params = {'N': 2**11, 'Nh': 2**4}
        else:
            pricer_params = {'N': 2**11, 'Nh': 2**2}

        model_instance = option_model.model_instance

        # Nh > 2**4 causes it to not match GilPeleazEuropeanPricer_cdf for simpler models
        pricer = HilbertEuropeanPricer_cdf(
            model=model_instance, N=pricer_params['N'], Nh=pricer_params['Nh']
            )

        # if option_model.model_name in ('Hes'):
        #     pricer_params = {'N': 2**11, 'L': 17, 'limit': 1000, 'order': 3, 'Nh': 2**3}
        # elif option_model.model_name in ('Bates'):
        #     pricer_params = {'N': 2**11, 'L': 17, 'limit': 1000, 'order': 3, 'Nh': 0.5}
        # elif option_model.model_name in ('VarianceGamma'):
        #     pricer_params = {'N': 2**11, 'L': 12, 'limit': 1000, 'order': 1, 'Nh': 2**4}
        # else:
        #     pricer_params = {'N': 2**11, 'L': 12, 'limit': 1000, 'order': 3, 'Nh': 0.9}


        # pricer = ProjEuropeanPricer_cdf(
        #     model=model_instance, N=pricer_params['N'], L=pricer_params['L'], order=pricer_params['order']
        #     )
        # if option_model.model_name in ('BlackScholes', 'Heston', 'MertonJD'):
        #     # if variance gamma - add order 3 for cubic and 1 for linear - default 1
        #     pricer = ProjEuropeanPricer_cdf(
        #         model=model_instance, N=pricer_params['N'], L=pricer_params['L'], order=pricer_params['order']
        #         )
        # elif option_model.model_name in ('Bates', 'VarianceGamma'):
        #     # Nh > 2**4 causes it to not match GilPeleazEuropeanPricer_cdf for simpler models
        #     pricer = HilbertEuropeanPricer_cdf(
        #         model=model_instance, N=pricer_params['N'], Nh=pricer_params['Nh']
        #         )
        # else:
        #     raise ValueError(f"{option_model.model_name} was not implemented")
        # pricer = GilPeleazEuropeanPricer_cdf(model=model_instance, limit=pricer_params['limit'])

        # Prepare market data - group by time to maturity
        ttm_groups = date_data.groupby('T')
        surface = MarketSurface()
        target_prices = []

        for ttm, group in ttm_groups:
            # not sure why a sorted df would ruin results, but keeping same order maintains it
            filtered_group = group[group['volume'].isin(
                group['volume'].nlargest(top_n_volume).tolist()
                )]
            strikes = filtered_group['strike'].values
            is_calls = filtered_group['is_call'].values.astype(bool)
            mid_prices = filtered_group['price'].values

            # Create market slice
            market_slice = MarketSlice(
                T=ttm,
                F=option_model.fwd_curve(ttm),
                disc=option_model.disc_curve(ttm),
                strikes=strikes,
                is_calls=is_calls,
                mid_prices=mid_prices
            )
            surface.add_slice(ttm, market_slice)
            target_prices.append(mid_prices)

        target_prices = np.concatenate(target_prices)

        # Define pricer function for targets
        def targets_pricer():
            # Function used to evaluate the model prices for each target
            # specifically for VarianceGamma model which can have log(-1) in it's convexity correction
            if 'convexity_correction' in dir(pricer._model):
                if np.isnan(pricer._model.convexity_correction()):
                    print("Warning: VarianceGamma model convexity correction resulted in NaN")
                    return np.zeros_like(target_prices)
            all_prices = []
            for ttm, market_slice in surface.slices.items():
                prices = pricer.price_strikes(T=ttm, K=market_slice.strikes, is_calls=market_slice.is_calls)

                all_prices.append(prices)
            return np.concatenate(all_prices)

        # Create calibrator
        calibrator = Calibrator(model=model_instance, minimizer=LeastSquares(max_nfev=1000 * top_n_volume, loss='soft_l1'))
        # override library which makes the guess the default_params()
        # the default_params for some of the models look dodgy, not in an array format
        calibrator._guess = model_instance.get_params()
        targets = Targets(target_prices, targets_pricer)
        calibrator.add_objective("Targets", targets)

        # Run calibration
        result = calibrator.calibrate()

        # Extract calibrated parameters
        calibrated_params_array = model_instance.get_params()
        param_names = option_model.initial_params.keys()
        calibrated_params = dict(zip(param_names, calibrated_params_array))

        # Calculate RMSE
        model_prices = targets_pricer()
        rmse = np.sqrt(np.mean((model_prices - target_prices) ** 2))

        return calibrated_params, rmse, pricer, surface

    except Exception as e:
        print(f"Calibration failed: {e}")
        return None, None, None, None


def get_rnds(pricer, surface):
    pdfs = {}
    cdfs = {}
    pdfs_discrete = {}
    for ttm, market_slice in surface.slices.items():
        # could do a range of strikes instead but would need to match with is_calls
        # strikes = np.linspace(min(market_slice.strikes), max(market_slice.strikes), 10)
        strikes_standard = np.linspace(market_slice.strikes.min(), market_slice.strikes.max(), 50)
        kwargs = {
            'ttms': [ttm], 
            'strikes': strikes_standard, 
            'is_calls': np.ones(len(strikes_standard), dtype=bool),
        }
        # there is only one ttm per call, so we index at 0
        cdf = get_cdf(pricer, **kwargs)[0]
        pdf = get_pdf(pricer, **kwargs)[0]
        EPS = np.diff(strikes_standard)[0]
        pdf_discrete = get_pdf(pricer, EPS=EPS, **kwargs) * EPS
        # pdf_discrete, strike_buckets = get_discrete_pdf(pricer, **kwargs)
        cdfs[ttm] = {'strikes': strikes_standard, 'rnd': cdf}
        pdfs[ttm] = {'strikes': strikes_standard, 'rnd': pdf}
        pdfs_discrete[ttm] = {'strikes': strikes_standard, 'rnd': pdf_discrete[0]}
        # pdfs_discrete[ttm] = {'strikes': strike_buckets[0], 'rnd': pdf_discrete[0]}
    return pdfs, cdfs, pdfs_discrete


def check_rnd_properties(pdfs, cdfs, ttms, model_name, S0=1):
    _, (ax1, ax2) = plt.subplots(1, 2)
    for ttm in ttms:
        # each plot is on it's own strikes
        ax1.plot(pdfs[ttm]['strikes'] / S0, pdfs[ttm]['rnd'])
        ax2.plot(cdfs[ttm]['strikes'] / S0, cdfs[ttm]['rnd'])
    ax1.set_xlabel('Strikes')
    ax2.set_xlabel('Strikes')
    ax1.set_title(f'Risk-Neutral PDFs - {model_name}')
    ax2.set_title(f'Risk-Neutral CDFs - {model_name}')
    ax1.grid()
    ax2.grid()
    plt.show()  


def check_single_pdf(ttms, pdfs, S0=1):
    fig, axs = plt.subplots(nrows=len(ttms)//3, ncols=3, figsize=(10, 8))
    for i, ax in enumerate(axs.flatten()):
        ttm = ttms[i]
        ax.plot(pdfs[ttm]['strikes'] / S0, pdfs[ttm]['rnd'])
        ax.set_title(f"ttm - {round(ttm * 365, 0)} days")
        ax.grid()
    plt.tight_layout()
    fig.show()


def compute_arrow_debreu_prices(pricer, pdfs, rf_in_pricer=False):
    values = [
        [(ttm, k, v) for k, v in zip(pdfs[ttm]['strikes'], pdfs[ttm]['rnd'])]
        for ttm in pdfs.keys()
        ]
    pdf_df_pivot = (
        pd.DataFrame(chain(*values), columns=['ttm', 'strike', 'pdf'])
          .drop_duplicates()
        # avoid duplicates
          .pivot_table(index='strike', columns='ttm', values='pdf')
          )
    # interpolate using Malz (2014): 
    # cubic-spline interpolation and flat fill at end points outside data
    pdf_df_pivot_interp = (
        pdf_df_pivot.interpolate(method='cubicspline', axis='index')
                    .bfill(axis='index')
                    .ffill(axis='index')
                    )

    adp_df = pdf_df_pivot_interp.copy()
    for ttm in adp_df.columns:
        # same r for all ttms since flat discount curve
        # but will accept a proper discount curve
        if rf_in_pricer:
            adp_df[ttm] = np.exp(-1 * pricer.get_r(ttm) * ttm) * pdf_df_pivot_interp[ttm]
        else:
            adp_df[ttm] = np.exp(-1 * pdfs[ttm]['rf'] * ttm) * pdf_df_pivot_interp[ttm]

    return adp_df

