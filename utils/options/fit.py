import numpy as np


from fypy.market.MarketSurface import MarketSlice, MarketSurface
from fypy.fit.Targets import Targets
from fypy.fit.Calibrator import Calibrator, LeastSquares
from fypy.model.levy import LevyModel


param_orders = {
    'BSM': ['sigma'],
    'MJD': ['sigma', 'lam', 'muj', 'sigj'],
    'Hes': ['v_0', 'theta', 'kappa', 'sigma_v', 'rho'],
    'BJD': ['v_0', 'theta', 'kappa', 'sigma_v', 'rho', 'lam', 'muj', 'sigj'],
    'VG': ['sigma', 'theta', 'nu'],
}


def fit_model(market_df, fwd, div_disc, model, model_name, pricer, top_n_volume=50):
    S0 = market_df['Adjusted close'].iloc[0]
    date = market_df['date'].iloc[0]
    
    fwd.discountCurve.current_time = date
    fwd._S0 = S0
    
    pricer._model.discountCurve.current_time = date
    pricer._model.forwardCurve.discountCurve.current_time = date
    pricer._model.forwardCurve._S0 = S0
    
    model.discountCurve.current_time = date
    model.forwardCurve.discountCurve.current_time = date
    model.forwardCurve._S0 = S0
    
    surface = MarketSurface()
    
    target_prices = []

    for ttm in market_df['ttm'].unique():
        filtered = market_df[market_df['ttm'] == ttm].sort_values('volume').iloc[:top_n_volume]
        strikes = filtered['strike'].to_numpy()
        prices = filtered['price'].to_numpy()
        
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
        if issubclass(model.__class__, LevyModel.LevyModel) and np.isnan(pricer._model.convexity_correction()):
            return np.zeros_like(target_prices)
        
        all_prices = []
        for ttm, market_slice in surface.slices.items():
            prices = pricer.price_strikes(T=ttm, K=market_slice.strikes, is_calls=market_slice.is_calls)
            all_prices.append(prices)
        return np.concatenate(all_prices)


    # Create the calibrator for the model
    calibrator = Calibrator(model=model, minimizer=LeastSquares())

    # Targets for the calibrator
    targets = Targets(target_prices, targets_pricer)
    calibrator.add_objective("Targets", targets)

    # Calibrate the model
    result = calibrator.calibrate()

    calibrated_params = model.get_params()
    data = {k: v for k, v in zip(param_orders[model_name], calibrated_params)}
    data['env: rf'] = fwd.discountCurve._r.loc[fwd.discountCurve.current_time]
    data['env: S0'] = S0
    return data