import numpy as np
import pandas as pd
from itertools import product
from topsis import *
from additions import *
import copy




# procedures for COMET method
def tfn(x, a, m, b):
    if x < a or x > b:
        return 0
    elif a <= x < m:
        return (x-a) / (m-a)
    elif m < x <= b:
        return (b-x) / (b-m)
    elif x == m:
        return 1


def evaluate_alternatives(C, x, ind):
    if ind == 0:
        return tfn(x, C[ind], C[ind], C[ind + 1])
    elif ind == len(C) - 1:
        return tfn(x, C[ind - 1], C[ind], C[ind])
    else:
        return tfn(x, C[ind - 1], C[ind], C[ind + 1])


#create characteristic values
def get_characteristic_values(matrix):
    cv = np.zeros((matrix.shape[1], 3))
    for j in range(matrix.shape[1]):
        cv[j, 0] = np.min(matrix[:, j])
        cv[j, 1] = np.mean(matrix[:, j])
        cv[j, 2] = np.max(matrix[:, j])
    return cv


#comet algorithm
def COMET(matrix, weights, criteria_types, norm_method):
    # generate characteristic values
    cv = get_characteristic_values(matrix)
    df_char_val = pd.DataFrame(cv)
    #df_char_val.to_csv('Char_val_' + str(year) + '.csv')
    #print(cv)
    # generate matrix with COs
    # cartesian product of characteristic values for all criteria
    co = product(*cv)
    co = np.array(list(co))

    # calculate vector SJ using chosen MCDA method
    sj = TOPSIS(co, weights, criteria_types, norm_method)

    # calculate vector P
    k = np.unique(sj).shape[0]
    p = np.zeros(sj.shape[0], dtype=float)

    for i in range(1, k):
        ind = sj == np.max(sj)
        p[ind] = (k - i) / (k - 1)
        sj[ind] = 0

    # inference and obtaining preference for alternatives
    preferences = []

    for i in range(len(matrix)):
        alt = matrix[i, :]
        W = []
        score = 0

        for i in range(len(p)):
            for j in range(len(co[i])):
                ind = int(np.where(cv[j] == co[i][j])[0])

                W.append(evaluate_alternatives(cv[j], alt[j], ind))
            score += np.product(W) * p[i]
            W = []
        preferences.append(score)
    preferences = np.asarray(preferences)

    rankingPrep = np.argsort(-preferences)
    rank = np.argsort(rankingPrep) + 1

    return preferences, rank
