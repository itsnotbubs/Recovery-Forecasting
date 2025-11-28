import numpy as np

from fypy.termstructures.DiscountCurve import DiscountCurve
from fypy.model.sv.Heston import Heston


class DiscountCurve_Measured(DiscountCurve):
    def __init__(self, rates):
        """
        Constant rate discount curve, exp(-r*T)
        :param rates: array of float, rate of discounting (e.g interest rate, div yield, etc)
        """
        super().__init__()
        self._r = rates
        self.current_time = rates.index[0]

    def discount_T(self, T):
        """
        Discount at time T in the future
        :param T: float or np.ndarray, time(s) in the future
        :return: float or np.ndarray, discounts(s) at time(s) in the future
        """
        return np.exp(-self._r.loc[self.current_time] * T)

    def implied_rate(self,  T):
        """
        Acquire the value for constant rate curves to circumvent numerical discrepancies
        arising from mathematical computations
        :param T: float or np.ndarray, time from which we imply the continuous rate, over [0,T]
        :return: float or np.ndarray (matches shape of input), the implied rate
        """
        return self._r.loc[self.current_time]*np.ones_like(T)


class Heston_fix(Heston):
    def default_params(self):
        """
        v_0: float = 0.04,
                 theta: float = 0.04,
                 kappa: float = 2.,
                 sigma_v: float = 0.3,
                 rho: float = -0.6
        :return:
        """
        # v_0, theta, kappa, sigma_v, rho
        return np.asarray(self._heston_default_params())