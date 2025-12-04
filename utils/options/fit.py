import numpy as np

from fypy.market.MarketSurface import MarketSlice, MarketSurface
from fypy.fit.Targets import Targets
from fypy.fit.Calibrator import Calibrator, LeastSquares
from fypy.model.levy import LevyModel

from .option_utils import PartialCalibrator


param_orders = {
    'BSM': ['sigma'],
    'MJD': ['sigma', 'lam', 'muj', 'sigj'],
    'Hes': ['v_0', 'theta', 'kappa', 'sigma_v', 'rho'],
    'BJD': ['v_0', 'theta', 'kappa', 'sigma_v', 'rho', 'lam', 'muj', 'sigj'],
    'VG': ['sigma', 'theta', 'nu'],
}
initial_params = {
    # model name with initial guesses for parameters
    'BSM': {'sigma': 0.002},
    'BJD': {
        'v_0': 0.0008,
        'theta': 0.0001,
        'kappa': 2.7,
        'sigma_v': 0.8,
        'rho': -0.6,
        'lam': 0.3,
        'muj': -0.001,
        'sigj': 0.003,
    },
    'Hes': {
        'v_0': 0.003,
        'theta': 0.00016,
        'kappa': 0.01,
        'sigma_v': 0.08,
        'rho': 0.2,
    },
    'MJD': {
        'sigma': 0.004,
        'lam': 0.0001,
        'muj': 0.07,
        'sigj': 0.07,
    },
    'VG': {
        'sigma': 0.004,
        'theta': 0.01,
        'nu': 0.02,
    }
}
IV_param_locs = {
    'BSM': 0,
    'MJD': 0,
    'Hes': 0,
    'BJD': 0,
    'VG': 0,
}


def fit_model(market_df, fwd, div_disc, model, model_name, pricers, guess, top_n_volume=51, lmbda=1e2):
    S0 = market_df['Adjusted close'].iloc[0]
    date = market_df['date'].iloc[0]
    
    fwd.discountCurve.current_time = date
    fwd._S0 = S0
    
    model.discountCurve.current_time = date
    model.forwardCurve.discountCurve.current_time = date
    model.forwardCurve._S0 = S0
    
    surface = MarketSurface()
    
    target_prices = []
    
    for i, pricer in enumerate(pricers):
        pricer._model.discountCurve.current_time = date
        pricer._model.forwardCurve.discountCurve.current_time = date
        pricer._model.forwardCurve._S0 = S0
        pricer._logS0 = np.log(S0) # required for CarrMadanEuropeanPricer weirdness, if used.

        for ttm in market_df['ttm'].unique():
            filtered = market_df[market_df['ttm'] == ttm].sort_values('volume').iloc[:top_n_volume]
            strikes = filtered['strike'].to_numpy()
            prices = filtered['price'].to_numpy()
            
            if i == 0:
                market_slice = MarketSlice(
                    T=ttm, F=fwd(ttm), disc=div_disc(ttm), strikes=strikes,
                    is_calls=np.ones(len(strikes), dtype=bool), mid_prices=prices
                )
                surface.add_slice(ttm, market_slice)
            
            target_prices.append(prices)

    # Full set of market target prices
    target_prices = np.concatenate(target_prices)


    def targets_pricer() -> np.ndarray:
        # Function used to evaluate the model prices for each target
        if issubclass(model.__class__, LevyModel.LevyModel) and np.isnan(pricers[0]._model.convexity_correction()):
            return np.zeros_like(target_prices)
        
        all_prices = []
        for pricer in pricers:
            for ttm, market_slice in surface.slices.items():
                prices = pricer.price_strikes(T=ttm, K=market_slice.strikes, is_calls=market_slice.is_calls)
                all_prices.append(prices)
        prices = np.concatenate(all_prices)
        if np.any(~np.isfinite(prices)):
            prices[~np.isfinite(prices)] = 0
        return prices


    # Create the calibrator for the model
    calibrator = Calibrator(model=model, minimizer=LeastSquares(max_nfev=1000 * top_n_volume, loss='soft_l1'))
    calibrator.set_initial_guess(guess)

    # Targets for the calibrator
    targets = Targets(target_prices, targets_pricer)
    calibrator.add_objective("Targets", targets)
    
    def penalty_pricer():
        return np.ones_like(pricers[0]._model.get_params()) * lmbda * pricers[0]._model.get_params()
    
    targets_penalty = Targets(np.zeros_like(pricers[0]._model.get_params()), penalty_pricer)
    calibrator.add_objective("Targets_penalty", targets_penalty)

    # Calibrate the model
    result = calibrator.calibrate()
    if not result.success:
        print(model_name, date, result.message)
        # raise ValueError()

    calibrated_params = model.get_params()
    data = {k: v for k, v in zip(param_orders[model_name], calibrated_params)}
    data['env: rf'] = fwd.discountCurve._r.loc[fwd.discountCurve.current_time]
    data['env: S0'] = S0
    return data, calibrated_params


def fit_model_residual_vol(
    market_df, fwd, div_disc, model, model_name, pricer, guess,
    top_n_volume=51, lmbda=1e2):

    rv_data = []

    S0 = market_df['Adjusted close'].iloc[0]
    date = market_df['date'].iloc[0]

    fwd.discountCurve.current_time = date
    fwd._S0 = S0

    pricer._model.discountCurve.current_time = date
    pricer._model.forwardCurve.discountCurve.current_time = date
    pricer._model.forwardCurve._S0 = S0
    pricer._logS0 = np.log(S0)

    model.discountCurve.current_time = date
    model.forwardCurve.discountCurve.current_time = date
    model.forwardCurve._S0 = S0

    surface = MarketSurface()
    # ivc = ImpliedVolCalculator_Black76(disc_curve=disc_curve, fwd_curve=fwd)

    for ttm in market_df['ttm'].unique():
        filtered = market_df[market_df['ttm'] == ttm].sort_values('volume', ascending=False).iloc[:top_n_volume]
        filtered = filtered[filtered['volume'] > 0]
        strikes = filtered['strike'].to_numpy()
        prices = filtered['price'].to_numpy()
        
        # vols = ivc.imply_vols(strikes=strikes, prices=prices, is_calls=np.ones(len(strikes), dtype=bool), ttm=ttm)
        
        market_slice = MarketSlice(
            T=ttm, F=fwd(ttm), disc=div_disc(ttm), strikes=strikes,
            is_calls=np.ones(len(strikes), dtype=bool), mid_prices=prices
        )
        surface.add_slice(ttm, market_slice)
        
        for strike, price in zip(strikes, prices):
            target_prices = np.array([price], dtype=np.float64)

            def targets_pricer() -> np.ndarray:
                # Function used to evaluate the model prices for each target
                if issubclass(model.__class__, LevyModel.LevyModel) and np.isnan(pricer._model.convexity_correction()):
                    return np.zeros(1, dtype=np.float64)
                
                return pricer.price_strikes(T=ttm, K=[strike], is_calls=[True])

            _guess = guess.copy()
            # _guess[IV_param_locs[model_name]] = iv
            
            # Create the calibrator for the model
            calibrator = PartialCalibrator(
                model=model,
                param_loc=IV_param_locs[model_name],
                minimizer=LeastSquares(max_nfev=1000),
            )
            calibrator.set_initial_guess(_guess)

            # Targets for the calibrator
            targets = Targets(target_prices, targets_pricer)
            calibrator.add_objective("Targets", targets)

            # Calibrate the model
            result = calibrator.calibrate()
            if not result.success:
                print(model_name, date, result.message)

            calibrated_params = model.get_params()
            rv_data.append({
                'RV': calibrated_params[IV_param_locs[model_name]],
                'date': date,
                'ttm': ttm,
                'strike': strike,
            })
    return rv_data