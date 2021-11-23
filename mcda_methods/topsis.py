import numpy as np
import sys

from additions import *

def TOPSIS(matrix, weights, types, norm_method):
    # Normalize matrix using chosen normalization (for example linear normalization)
    nmatrix = norm_method(matrix, types)

    # Multiplicate all rows of normalized matrix by weights
    weighted_matrix = nmatrix * weights

    # Calculate vectors of PIS (ideal solution) and NIS (anti-ideal solution)
    pis = np.max(weighted_matrix, axis=0)
    nis = np.min(weighted_matrix, axis=0)

    # Calculate chosen distance of every alternative from PIS and NIS
    Dp = (np.sum((weighted_matrix - pis)**2, axis = 1))**0.5
    Dm = (np.sum((weighted_matrix - nis)**2, axis = 1))**0.5

    return Dm / (Dm + Dp)
