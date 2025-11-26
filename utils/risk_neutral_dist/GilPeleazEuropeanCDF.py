import numpy as np
from scipy.integrate import quad

from fypy.pricing.fourier.GilPeleazEuropeanPricer import GilPeleazEuropeanPricer


class GilPeleazEuropeanPricer_cdf(GilPeleazEuropeanPricer):
    def risk_neutral_cdf(self, T: float, K: float, is_call: bool):
        """
        Get the risk-neutral CDF at a single strike of European option
        :param T: float, time to maturity
        :param K: float, strike of option
        :param is_call: bool, indicator of if strike is call (true) or put (false)
        :return: float, risk-neutral CDF at strike of European option
        """
        S0 = self._model.spot()
        k = np.log(K / S0)
        chf = lambda x: self._model.chf(T=T, xi=x)

        integrand2 = lambda u: np.real(np.exp(-u * k * 1j) / (u * 1j) * chf(u))
        int2 = 1 / 2 + 1 / np.pi * quad(integrand2, 1e-15, np.inf, limit=self._limit)[0]

        return 1 - int2