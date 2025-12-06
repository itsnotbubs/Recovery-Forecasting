import numpy as np
import scipy.optimize as so


def piecewise_constant_mat(N, l):
    r"""
    Generates pricing kernel with piecewise constant curvatures X of shape (l, N + 1).
    
    This allows us to approximate h^{-1} as (X[::-1,1:] @ \theta) + 1 where we can estimate \theta (N params)
    rather than h^{-1} (l params).
    """
    assert ((l - 2) // N) * N == l - 2, "l - 2 must be divisible by N"
    X = np.zeros((l, N + 1), dtype=np.float64)
    X[:,0] = 1
    k = (l - 2) // N + 1
    tmp = np.arange(k)[::-1] / k
    X[-k:, 1] = tmp
    X[:-k, 1] = 1
    for n in range(2, N+1):
        X[-(k * n) + n-1:-(k * (n - 1)) + n-1, n] = tmp
        X[:-(k * n) + n-1, n] = 1
    return X


def generalized_recovery_piecewise_approx_delta(dist_data, times, delta_0=0.9999, pieces=10):
    """
    piecewise and delta approximation
    
    This formulation requires piecewise parameters to be monotonic,
    and therefore must use nonnegative least squares algorithm
    
    Inputs:
        dist_data: discretized risk-neutral PDF of S_{t + times_{i}} for each i,
            np.float64 matrix of shape (times, states)
        times: set of times to expiry in current market snapshot
    Outputs:
        f: discretized physical PDF of S_{t + times_{i}} for each i, np.float64 matrix of shape (times, states)
        D_inv: time discount vector of shape (times)
        H_inv: price discount vector of shape (states), m^{T}
    """
    P_1 = dist_data[:,0]
    P_2 = dist_data[:,1:]
    
    pieces = piecewise_constant_mat(pieces, dist_data.shape[1])[::-1,1:]
    # H_inv = np.concatenate([[1], pieces[1:,:] @ theta + 1]) where (pieces[:,:] @ theta + 1)[0] == 1 by definition
    # where theta = beta[1:]
    offset = np.ones(P_2.shape[1])
    
    
    a = -(times - 1) * delta_0 ** times
    b = times * delta_0 ** (times - 1)
    
    X = np.column_stack([-b, P_2 @ pieces[1:,:]])
    y = a - P_1 - (P_2 @ offset)
    # \beta = [delta, 1/h_2, ..., 1/h_n]

    beta = so.nnls(X, y)[0]

    D_inv = (beta[0] ** (-1 / times[0])) ** times
    H_inv = np.concatenate([[1], pieces[1:,:] @ beta[1:] + 1])

    f = np.diag(D_inv) @ dist_data @ np.diag(H_inv)
    return f, D_inv, H_inv


def generalized_recovery_piecewise_ridge_approx_delta(dist_data, times,
                                                      delta_0=0.9999, pieces=10, lmbda=1e-1, lmbda_delta=1e2):
    """
    piecewise and delta approximation
    
    This formulation requires piecewise parameters to be monotonic,
    and therefore must use nonnegative least squares algorithm
    
    Inputs:
        dist_data: discretized risk-neutral PDF of S_{t + times_{i}} for each i,
            np.float64 matrix of shape (times, states)
        times: set of times to expiry in current market snapshot
    Outputs:
        f: discretized physical PDF of S_{t + times_{i}} for each i, np.float64 matrix of shape (times, states)
        D_inv: time discount vector of shape (times)
        H_inv: price discount vector of shape (states), m^{T}
    """
    P_1 = dist_data[:,0]
    P_2 = dist_data[:,1:]
    
    pieces = piecewise_constant_mat(pieces, dist_data.shape[1])[::-1,1:]
    # H_inv = np.concatenate([[1], pieces[1:,:] @ theta + 1]) where (pieces[:,:] @ theta + 1)[0] == 1 by definition
    # where theta = beta[1:]
    offset = np.ones(P_2.shape[1])
    
    
    a = -(times - 1) * delta_0 ** times
    b = times * delta_0 ** (times - 1)
    
    X = np.column_stack([-b, P_2 @ pieces[1:,:]])
    y = a - P_1 - (P_2 @ offset)
    # \beta = [delta, 1/h_2, ..., 1/h_n]
    
    row_aug_X = np.eye(X.shape[1] - 1 + 1, dtype=np.float64)
    row_aug_X[1:,1:] = row_aug_X[1:,1:] * np.sqrt(lmbda)
    row_aug_X[0,0] = row_aug_X[0,0] * np.sqrt(lmbda_delta)
    row_aug_y = np.ones(X.shape[1] - 1 + 1, dtype=np.float64)
    row_aug_y[1:] = row_aug_y[1:] * np.sqrt(lmbda) * 0
    row_aug_y[0] = row_aug_y[0] * np.sqrt(lmbda_delta) * (delta_0 ** times[0])
    
    X = np.row_stack([
        X,
        row_aug_X
    ])
    y = np.concatenate([y, row_aug_y])

    beta = so.nnls(X, y)[0]

    D_inv = (beta[0] ** (-1 / times[0])) ** times
    H_inv = np.concatenate([[1], pieces[1:,:] @ beta[1:] + 1])

    f = np.diag(D_inv) @ dist_data @ np.diag(H_inv)
    return f, D_inv, H_inv


def generalized_recovery_ridge_approx_delta_h1(dist_data, times, delta_0=0.9999, lmbda=1e-2, lmbda_delta=1e6):
    """
    Ridge regression row augmentation and delta approximation
    
    This formulation assumes transitory component has been removed,
    therefore we can assume remaining component is 1 centered
    and we do not require that component to be monotonic.
    
    Inputs:
        dist_data: discretized risk-neutral PDF of S_{t + times_{i}} for each i,
            np.float64 matrix of shape (times, states)
        times: set of times to expiry in current market snapshot
    Outputs:
        f: discretized physical PDF of S_{t + times_{i}} for each i, np.float64 matrix of shape (times, states)
        D_inv: time discount vector of shape (times)
        H_inv: price discount vector of shape (states), m^{T}*m^{P} (or only m^{P} if m^{T} is fully removed)
    """
    P_1 = dist_data[:,0]
    P_2 = dist_data[:,1:]
    
    a = -(times - 1) * delta_0 ** times
    b = times * delta_0 ** (times - 1)
    
    X = np.column_stack([-b, P_2])
    y = a - P_1
    # \beta = [delta, 1/h_2, ..., 1/h_n]
    
    row_aug_X = np.eye(P_2.shape[1] + 1, dtype=np.float64)
    row_aug_X[1:,1:] = row_aug_X[1:,1:] * np.sqrt(lmbda)
    row_aug_X[0,0] = row_aug_X[0,0] * np.sqrt(lmbda_delta)
    row_aug_y = np.ones(P_2.shape[1] + 1, dtype=np.float64)
    row_aug_y[1:] = row_aug_y[1:] * np.sqrt(lmbda)
    row_aug_y[0] = row_aug_y[0] * np.sqrt(lmbda_delta) * (delta_0 ** times[0])
    
    X = np.row_stack([
        X,
        row_aug_X
    ])
    y = np.concatenate([y, row_aug_y])

    beta = so.nnls(X, y)[0]

    D_inv = (beta[0] ** (-1 / times[0])) ** times
    H_inv = np.concatenate([[1], beta[1:]])

    f = np.diag(D_inv) @ dist_data @ np.diag(H_inv)
    return f, D_inv, H_inv


def generalized_recovery_ridge_approx_delta(dist_data, times, delta_0=0.9999, lmbda=1e-2, lmbda_delta=1e6):
    """
    Ridge regression row augmentation and delta approximation
    
    This formulation assumes transitory component has been removed,
    therefore we can assume remaining component is 1 centered
    and we do not require that component to be monotonic.
    We also allow h_1 to take on some value other than 1.
    
    Inputs:
        dist_data: discretized risk-neutral PDF of S_{t + times_{i}} for each i,
            np.float64 matrix of shape (times, states)
        times: set of times to expiry in current market snapshot
    Outputs:
        f: discretized physical PDF of S_{t + times_{i}} for each i, np.float64 matrix of shape (times, states)
        D_inv: time discount vector of shape (times)
        H_inv: price discount vector of shape (states), m^{T}*m^{P} (or only m^{P} if m^{T} is fully removed)
    """
    P = dist_data[:,:]
    
    a = -(times - 1) * delta_0 ** times
    b = times * delta_0 ** (times - 1)
    
    X = np.column_stack([-b, P])
    y = a
    # \beta = [delta, 1/h_2, ..., 1/h_n]
    
    row_aug_X = np.eye(P.shape[1] + 1, dtype=np.float64)
    row_aug_X[1:,1:] = row_aug_X[1:,1:] * np.sqrt(lmbda)
    row_aug_X[0,0] = row_aug_X[0,0] * np.sqrt(lmbda_delta)
    row_aug_y = np.ones(P.shape[1] + 1, dtype=np.float64)
    row_aug_y[1:] = row_aug_y[1:] * np.sqrt(lmbda)
    row_aug_y[0] = row_aug_y[0] * np.sqrt(lmbda_delta) * (delta_0 ** times[0])
    
    X = np.row_stack([
        X,
        row_aug_X
    ])
    y = np.concatenate([y, row_aug_y])

    beta = so.nnls(X, y)[0]

    D_inv = (beta[0] ** (-1 / times[0])) ** times
    H_inv = beta[1:]

    f = np.diag(D_inv) @ dist_data @ np.diag(H_inv)
    return f, D_inv, H_inv


def compute_physical_dist(adp, key='pw_ridge', N=None):
    """
    key can be one of : 'pw' 'ridge' 'ridge_h1' 'pw_ridge'"
    N is supplied if using 'pw', or 'pw_ridge
    """
    fns = {
        'pw': generalized_recovery_piecewise_approx_delta,
        'pw_ridge': generalized_recovery_piecewise_ridge_approx_delta,
        'ridge': generalized_recovery_ridge_approx_delta,
        'ridge_h1': generalized_recovery_ridge_approx_delta_h1,
    }
    fn = fns[key]
    if N is not None:
        f_dist, D_inv, H_inv = fn(adp.T.to_numpy(), adp.columns, pieces=N)
    else:
        f_dist, D_inv, H_inv = fn(adp.T.to_numpy(), adp.columns)
    return f_dist, D_inv, H_inv


