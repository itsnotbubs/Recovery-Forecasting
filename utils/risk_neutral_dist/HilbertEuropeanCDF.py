import numpy as np    
from fypy.pricing.fourier.HilbertEuropeanPricer import HilbertEuropeanPricer


class HilbertEuropeanPricer_cdf(HilbertEuropeanPricer):
    def risk_neutral_cdf(self, T, K, is_call):
        gridL = np.arange(-int(self._N / 2), 0)
        gridR = -gridL[::-1]
        H = (
            np.sum(
                self._g(self._h * gridL, T, K) * (np.cos(np.pi * gridL) - 1) / gridL
                + self._g(self._h * gridR, T, K) * (np.cos(np.pi * gridR) - 1) / gridR
            )
            / np.pi
        )
        return -np.real(1j * H)

    def _g(self, xi: np.ndarray, T: float, K: float):
        return np.exp(-1j * xi * np.log(K / self._model.spot())) * (
            self._model.chf(T, xi)
        )
    
    def get_r(self, T:float):
        return self._model.discountCurve.implied_rate(T)