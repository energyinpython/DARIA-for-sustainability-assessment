import numpy as np
import sys
import pandas as pd

from additions import *

def TOPSIS(matrix, weights, types, norm_method, list_of_ind, list_of_cols, year):
    # Normalize matrix using chosen normalization (for example linear normalization)
    nmatrix = norm_method(matrix, types)

    # save normalized matrix nmatrix as df in csv in aim top display it in paper
    df_nmatrix = pd.DataFrame(data = nmatrix, index = list_of_ind, columns = list_of_cols)
    df_weights = pd.DataFrame(data = weights.reshape(1, -1), index = ['Weights'], columns = list_of_cols)
    df_types = pd.DataFrame(data = types.reshape(1, -1), index = ['Types'], columns = list_of_cols)
    df_nmatrix = pd.concat([df_nmatrix, df_weights, df_types], axis = 0)
    df_nmatrix.to_csv('output/df_nmatrix_topsis_' + str(year) + '.csv')
    print(df_weights)

    # Multiplicate all rows of normalized matrix by weights
    weighted_matrix = nmatrix * weights

    
    # Calculate vectors of PIS (ideal solution) and NIS (anti-ideal solution)
    pis = np.max(weighted_matrix, axis=0)
    nis = np.min(weighted_matrix, axis=0)

    # Calculate chosen distance of every alternative from PIS and NIS
    Dp = (np.sum((weighted_matrix - pis)**2, axis = 1))**0.5
    Dm = (np.sum((weighted_matrix - nis)**2, axis = 1))**0.5

    return Dm / (Dm + Dp), weighted_matrix
