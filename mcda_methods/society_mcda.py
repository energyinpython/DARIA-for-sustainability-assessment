import numpy as np
import pandas as pd
import os

from topsis import TOPSIS
from vikor import VIKOR
from comet import COMET
from additions import *

path = 'DATASET'
years = [2015, 2016, 2017, 2018, 2019]
#years = [2015]
mcda_methods = ['TOPSIS', 'VIKOR', 'COMET']
norm_method = minmax_normalization

list_alt_names = []
for i in range(1, 26 + 1):
    list_alt_names.append(r'$A_{' + str(i) + '}$')

df_writer_pref = pd.DataFrame()
df_writer_pref['Ai'] = list_alt_names

df_writer_rank = pd.DataFrame()
df_writer_rank['Ai'] = list_alt_names

df_country = pd.DataFrame()
df_country['Ai'] = list_alt_names

for year in years:
    
    file = 'data_' + str(year) + '.csv'
    pathfile = os.path.join(path, file)
    data = pd.read_csv(pathfile, index_col = 'Country')

    df_data = data.iloc[:len(data) - 1, :]
    types = data.iloc[len(data) - 1, :].to_numpy()

    df_data = df_data.dropna()
    
    print(df_data)
    df_country.to_csv('alts.csv')
    df_data.to_csv('names' + str(year) + '.csv')

    
    matrix = df_data.to_numpy()
    
    
    # mcda
    weights = critic_weighting(matrix)

    # TOPSIS
    pref = TOPSIS(matrix, weights, types, norm_method)
    rankingPrep = np.argsort(-pref)
    rank = np.argsort(rankingPrep) + 1

    df_writer_pref['TOPSIS'] = pref
    df_writer_rank['TOPSIS'] = rank

    # VIKOR
    pref = VIKOR(matrix, weights, types, norm_method)
    rankingPrep = np.argsort(pref)
    rank = np.argsort(rankingPrep) + 1

    df_writer_pref['VIKOR'] = pref
    df_writer_rank['VIKOR'] = rank


    # COMET
    pref, rank = COMET(matrix, weights, types, norm_method)

    df_writer_pref['COMET'] = pref
    df_writer_rank['COMET'] = rank


    df_writer_pref.to_csv('results_pref_' + str(year) + '.csv')
    df_writer_rank.to_csv('results_rank_' + str(year) + '.csv')
    