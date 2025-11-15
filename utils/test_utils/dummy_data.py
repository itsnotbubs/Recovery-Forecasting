import numpy as np
import scipy.stats as ss


def make_gen_recov_arrow_data(x, means, stds):
    """
    Creates a sample arrow-debreau matrix for the generalized recovery method.
    
    It is not required that the states range is the same for each time in the
    generalized recovery method, however for this dummy data generator that is
    the assumed case.
    
    Inputs:
        x: equally spaced range of states for arrow-debreau (discretization for x-axis of S_{t + times_{i}} PDF),
            normalized to S_{t}=1
        means: lognormal loc parameter for each position along time axis
        stds: lognormal s parameter for each position along the time axis
    Outputs:
        dist_data: discretized risk-neutral PDF of S_{t + times_{i}} for each i,
            np.float64 matrix of shape (times, states)
    """
    dist_data = np.empty((means.shape[0], x.shape[0]), dtype=np.float64)
    
    for i, (mean, std) in enumerate(zip(means, stds)):
        dist_data[i,:] = ss.lognorm.pdf(x, loc=np.log(mean), s=std) * (x[1] - x[0])
    return dist_data


def make_sample_1():
    """
    Simple testing data generator
    
    Outputs:
        x: equally spaced range of states for arrow-debreau (discretization for x-axis of S_{t + times_{i}} PDF),
            normalized to S_{t}=1
        times: set of times to expiry in current market snapshot
        dist_data: discretized risk-neutral PDF of S_{t + times_{i}} for each i,
            np.float64 matrix of shape (times, states)
    """
    x = np.linspace(np.exp(0.8-1), np.exp(1.2-1), 1002) # return levels
    times = np.array([5, 10, 20, 40, 60, 120, 240])
    means = 1.0004 ** times
    # generous guess at variance growth as a harmonic number scaling of 1 step variance
    # will make computations more testable
    euler_mascheroni = 0.57721566490153286060651209008240243104215933593992
    stds = 0.008 * (np.log(times) + euler_mascheroni) ** 1.4
    dist_data = make_gen_recov_arrow_data(x, means, stds)
    return x, times, dist_data