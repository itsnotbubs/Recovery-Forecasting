import numpy as np

from fypy.termstructures.DiscountCurve import DiscountCurve
from fypy.model.sv.Heston import Heston
from fypy.pricing.fourier.ProjEuropeanPricer import ProjEuropeanPricer
from fypy.fit.Calibrator import Calibrator, LeastSquares


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


class ProjEuropeanPricer_fix(ProjEuropeanPricer):
    def get_nbar(self, a: float, lws: float, lam: float) -> int:
        try:
            nbar = max(min(
                int(np.floor(a * (lws - lam) + 1)), self._N - 1
            ), 3)
        except Exception as e:
            raise e
        return nbar


class PartialCalibrator(Calibrator):
    """
    partial calibrator for optimizing to volatility surface after initial calibration,
    implementation only considers the case of a single parameter for partial calibration.
    """
    def __init__(self,
                 model,
                 param_loc,
                 minimizer = LeastSquares(),
                 loss = None):
        self._model = model
        self._minimizer = minimizer

        self._objectives = {}
        self._loss = loss

        # Initialize the guess and bounds, using model defaults. These can be overridden
        self.param_loc = param_loc
        self._bounds = model.param_bounds()

        self._constraints = {}

    def calibrate(self):
        """ Run the calibration, fits the model in place, returns the optimization summary """
        if len(self._objectives) == 0:
            raise RuntimeError("You never set any objectives ")

        result = self._minimizer.minimize(self._objective_value if self._loss else self._objective_vector,
                                          bounds=[self._bounds[self.param_loc]],
                                          guess=np.array([self._guess[self.param_loc]]),
                                          constraints=self._constraints.values())

        # Set the final parameters in the model
        params = self._guess.copy()
        params[self.param_loc] = result.params[0]
        self._model.set_params(params)
        return result

    def _objective_vector(self, partial_params):
        params = self._guess.copy()
        params[self.param_loc] = partial_params[0]
        
        # Set the parameters into model
        self._model.set_params(params)

        # Evaluate the residuals for all objectives
        return np.concatenate([objective.value() for _, objective in self._objectives.items()])

    def _objective_value(self, partial_params):
        params = self._guess.copy()
        params[self.param_loc] = partial_params[0]
        
        # Set the parameters into model
        self._model.set_params(params)

        # Evaluate the residuals for all
        val = self._loss.agg_apply([self._loss.residual_apply(objective.value())
                                    for _, objective in self._objectives.items() if objective.strength > 0])
        return val