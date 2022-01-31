import numpy as np
import sys
from scipy.stats import pearsonr


# linear normalization
def linear_normalization(matrix, types):
    ind_profit = np.where(types == 1)
    ind_cost = np.where(types == -1)
    nmatrix = np.zeros(np.shape(matrix))
    nmatrix[:, ind_profit] = matrix[:, ind_profit] / (np.amax(matrix[:, ind_profit], axis = 0))
    nmatrix[:, ind_cost] = np.amin(matrix[:, ind_cost], axis = 0) / matrix[:, ind_cost]
    return nmatrix


# min-max normalization
def minmax_normalization(X, criteria_type):
    x_norm = np.zeros((X.shape[0], X.shape[1]))
    ind_profit = np.where(criteria_type == 1)
    ind_cost = np.where(criteria_type == -1)

    x_norm[:, ind_profit] = (X[:, ind_profit] - np.amin(X[:, ind_profit], axis = 0)
                             ) / (np.amax(X[:, ind_profit], axis = 0) - np.amin(X[:, ind_profit], axis = 0))

    x_norm[:, ind_cost] = (np.amax(X[:, ind_cost], axis = 0) - X[:, ind_cost]
                           ) / (np.amax(X[:, ind_cost], axis = 0) - np.amin(X[:, ind_cost], axis = 0))

    return x_norm


# max normalization
def max_normalization(X, criteria_type):
    maximes = np.amax(X, axis=0)
    ind = np.where(criteria_type == -1)
    X = X/maximes
    X[:,ind] = 1-X[:,ind]
    return X


# sum normalization
def sum_normalization(X, criteria_type):
    x_norm = np.zeros((X.shape[0], X.shape[1]))
    ind_profit = np.where(criteria_type == 1)
    ind_cost = np.where(criteria_type == -1)

    x_norm[:, ind_profit] = X[:, ind_profit] / np.sum(X[:, ind_profit], axis = 0)

    x_norm[:, ind_cost] = (1 / X[:, ind_cost]) / np.sum((1 / X[:, ind_cost]), axis = 0)

    return x_norm


# equal weighting
def mean_weighting(X):
    N = np.shape(X)[1]
    return np.ones(N) / N


# entropy weighting
def entropy_weighting(X):
    # normalization for profit criteria
    criteria_type = np.ones(np.shape(X)[1])
    pij = sum_normalization(X, criteria_type)
    m, n = np.shape(pij)

    H = np.zeros((m, n))
    for j in range(n):
        for i in range(m):
            if pij[i, j] != 0:
                H[i, j] = pij[i, j] * np.log(pij[i, j])

    h = np.sum(H, axis = 0) * (-1 * ((np.log(m)) ** (-1)))
    d = 1 - h
    w = d / (np.sum(d))

    return w


# standard deviation weighting
def std_weighting(X):
    stdv = np.std(X, axis = 0)
    return stdv / np.sum(stdv)


# CRITIC weighting
def critic_weighting(X):
    # normalization for profit criteria
    criteria_type = np.ones(np.shape(X)[1])
    x_norm = minmax_normalization(X, criteria_type)
    std = np.std(x_norm, axis = 0)
    n = np.shape(x_norm)[1]
    correlations = np.zeros((n, n))
    for i in range(0, n):
        for j in range(0, n):
            correlations[i, j], _ = pearsonr(x_norm[:, i], x_norm[:, j])

    difference = 1 - correlations
    suma = np.sum(difference, axis = 0)
    C = std * suma
    w = C / (np.sum(C, axis = 0))
    return w


# weighted spearman coefficient rw
def weighted_spearman(R, Q):
    N = len(R)
    denom = N**4 + N**3 - N**2 - N
    reszta = (N-R+1)+(N-Q+1)
    suma = 6*sum((R-Q)**2*reszta)
    rW = 1-(suma/denom)
    return rW

# rank similarity coefficient WS
def coeff_WS(R, Q):
    sWS = 0
    N = len(R)
    for i in range(N):
        sWS += 2**(-int(R[i]))*(abs(R[i]-Q[i])/max(abs(R[i] - 1), abs(R[i] - N)))
    WS = 1 - sWS
    return WS
