import numpy as np
import sys

from additions import *


def VIKOR(matrix, weights, types, norm_method):
    v = 0.5
    nmatrix = norm_method(matrix, types)
    fstar = np.amax(nmatrix, axis = 0)
    fminus = np.amin(nmatrix, axis = 0)
    weighted_matrix = weights * ((fstar - nmatrix) / (fstar - fminus))
    S = np.sum(weighted_matrix, axis = 1)
    R = np.amax(weighted_matrix, axis = 1)
    Sstar = np.min(S)
    Sminus = np.max(S)
    Rstar = np.min(R)
    Rminus = np.max(R)
    Q = v * (S - Sstar) / (Sminus - Sstar) + (1 - v) * (R - Rstar) / (Rminus - Rstar)
    return Q