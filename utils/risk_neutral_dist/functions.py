import numpy as np
from scipy.interpolate import interp1d, CubicSpline


def get_cdf(cdf_pricer, ttms, strikes, is_calls):
    """
    ttms: array of floats of times to maturity
    strikes: array of floats of strikes
    
    returns: 2d array of floats, each row ttm provding the cdf value at each column strike
    """
    return np.array([[cdf_pricer.risk_neutral_cdf(T=ttm, K=strike, is_call=is_calls) for strike in strikes] for ttm in ttms])


def get_pdf(cdf_pricer, ttms, strikes, is_calls, EPS=1e-10):
    """
    ttms: array of floats of times to maturity
    strikes: array of floats of strikes
    
    returns: 2d array of floats, each row ttm provding the pdf value at each column strike
    """
    return np.array([[
        (
            cdf_pricer.risk_neutral_cdf(T=ttm, K=strike + EPS/2, is_call=is_calls)
            - cdf_pricer.risk_neutral_cdf(T=ttm, K=strike - EPS/2, is_call=is_calls)
        ) / EPS
        for strike in strikes
    ] for ttm in ttms])


def get_cdf_RV(cdf_pricer, ttms, strikes, is_calls, params, RV_param_loc, df_RV_date):
    """
    uses residual volatility for computation of cdf
    
    ttms: array of floats of times to maturity
    strikes: array of floats of strikes
    
    returns: 2d array of floats, each row ttm provding the cdf value at each column strike
    """
    _params = params.copy()
    risk_neutral_cdf = []
    for ttm in ttms:
        df_RV_date_ttm = df_RV_date[df_RV_date['ttm'] == ttm].sort_values('strike').drop_duplicates('strike')
        interp_bounds = interp1d(df_RV_date_ttm['strike'], df_RV_date_ttm['RV'], kind='linear', fill_value='extrapolate')
        interp_core = CubicSpline(df_RV_date_ttm['strike'], df_RV_date_ttm['RV'], extrapolate=False)
        def interp(x):
            y = interp_core(x)
            mask = np.isnan(y)
            if np.any(mask):
                y[mask] = interp_bounds(x[mask])
            return np.maximum(y, np.amin(df_RV_date_ttm['RV']))
        
        risk_neutral_cdf_ = []
        for strike, is_call in zip(strikes, is_calls):
            _params[RV_param_loc] = interp(strike)
            cdf_pricer._model.set_params(_params)
            val = cdf_pricer.risk_neutral_cdf(T=ttm, K=strike, is_call=is_call)
            risk_neutral_cdf_.append(0 if np.isnan(val) else val)
        risk_neutral_cdf.append(risk_neutral_cdf_)
    return np.array(risk_neutral_cdf, dtype=np.float64)


def get_pdf_RV(cdf_pricer, ttms, strikes, is_calls, params, RV_param_loc, df_RV_date, EPS=1e-10):
    """
    uses residual volatility and numerical differentiation for computation of pdf
    
    ttms: array of floats of times to maturity
    strikes: array of floats of strikes
    
    returns: 2d array of floats, each row ttm provding the pdf value at each column strike
    """
    _params = params.copy()
    risk_neutral_pdf = []
    for ttm in ttms:
        df_RV_date_ttm = df_RV_date[df_RV_date['ttm'] == ttm].sort_values('strike').drop_duplicates('strike')
        interp_bounds = interp1d(df_RV_date_ttm['strike'], df_RV_date_ttm['RV'], kind='linear', fill_value='extrapolate')
        interp_core = CubicSpline(df_RV_date_ttm['strike'], df_RV_date_ttm['RV'], extrapolate=False)
        def interp(x):
            y = interp_core(x)
            mask = np.isnan(y)
            if np.any(mask):
                y[mask] = interp_bounds(x[mask])
            return np.maximum(y, np.amin(df_RV_date_ttm['RV']))
        
        risk_neutral_pdf_ = []
        for strike, is_call in zip(strikes, is_calls):
            
            _params[RV_param_loc] = interp(strike - EPS/2)
            cdf_pricer._model.set_params(_params)
            lower = cdf_pricer.risk_neutral_cdf(T=ttm, K=strike - EPS/2, is_call=is_call)
            
            _params[RV_param_loc] = interp(strike - EPS/2)
            cdf_pricer._model.set_params(_params)
            upper = cdf_pricer.risk_neutral_cdf(T=ttm, K=strike + EPS/2, is_call=is_call)
            
            val = (upper - lower) / EPS
            risk_neutral_pdf_.append(0 if np.isnan(val) else val)
        risk_neutral_pdf.append(risk_neutral_pdf_)
    return np.array(risk_neutral_pdf, dtype=np.float64)