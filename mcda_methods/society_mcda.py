import numpy as np
import pandas as pd
import os

from topsis import TOPSIS
from vikor import VIKOR
from comet import COMET
from additions import *


def main():
    path = 'DATASET'
    years = [2015, 2016, 2017, 2018, 2019]
    methods = [
        'topsis',
        'vikor',
        'comet',
        ]
    norm_method = minmax_normalization

    list_alt_names = []
    for i in range(1, 26 + 1):
        list_alt_names.append(r'$A_{' + str(i) + '}$')


    for method in methods:
        df_writer_pref = pd.DataFrame()
        df_writer_pref['Ai'] = list_alt_names

        df_writer_rank = pd.DataFrame()
        df_writer_rank['Ai'] = list_alt_names

        for year in years:
            file = 'data_' + str(year) + '.csv'
            pathfile = os.path.join(path, file)
            data = pd.read_csv(pathfile, index_col = 'Country')

            df_data = data.iloc[:len(data) - 1, :]
            types = data.iloc[len(data) - 1, :].to_numpy()

            df_data = df_data.dropna()
            print(df_data)
            matrix = df_data.to_numpy()
    
            # mcda
            weights = critic_weighting(matrix)

            # TOPSIS
            if method == 'topsis':
                pref = TOPSIS(matrix, weights, types, norm_method)
                rankingPrep = np.argsort(-pref)
                rank = np.argsort(rankingPrep) + 1

            # VIKOR
            elif method == 'vikor':
                pref = VIKOR(matrix, weights, types, norm_method)
                rankingPrep = np.argsort(pref)
                rank = np.argsort(rankingPrep) + 1

            # COMET
            # be patient as the program may take longer to execute
            elif method == 'comet':
                pref, rank = COMET(matrix, weights, types, norm_method)

            df_writer_pref[str(year)] = pref
            df_writer_rank[str(year)] = rank

        df_writer_pref = df_writer_pref.set_index('Ai')
        df_writer_rank = df_writer_rank.set_index('Ai')
        df_writer_pref.to_csv('output/' + method + '_pref.csv')
        df_writer_rank.to_csv('output/' + method + '_rank.csv')

if __name__ == "__main__":
    main()
    