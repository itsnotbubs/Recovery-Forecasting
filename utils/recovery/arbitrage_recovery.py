import numpy as np


def get_arbitrage_based_SDF(x):
    """
    This uses the no-dividend (gross-dividend yield matrix F_{i,j}=1) form of the arbitrage-based
    recovery fomulation. This is computed relative to the minimum of the future price states such that
    m^{T}_{0}=1 for easier comparison with the base method of genralized recovery.
    
    Inputs:
        x: equally spaced range of states for arrow-debreau (discretization for x-axis of S_{t + times_{i}} PDF),
            normalized to S_{t}=1
    Outputs:
        m^{T}: estimation of transitory component of stochastic discount function
    """
    return np.amin(x) / x