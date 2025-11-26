import numpy as np


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