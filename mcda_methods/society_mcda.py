import numpy as np
import pandas as pd
import os
import copy

from topsis import TOPSIS
from vikor import VIKOR
from comet import COMET
from additions import *
from rank_preferences import *


def main():
    path = 'DATASET'
    years = [
        2015,
        2016,
        2017,
        2018,
        2019,
    ]
    str_years = []
    for y in years:
        str_years.append(str(y))
    methods = [
        'topsis',
        #'vikor',
        #'comet',
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

        for el, year in enumerate(years):
            file = 'data_' + str(year) + '.csv'
            pathfile = os.path.join(path, file)
            data = pd.read_csv(pathfile, index_col = 'Country')
            
            df_data = data.iloc[:len(data) - 1, :]
            types = data.iloc[len(data) - 1, :].to_numpy()

            df_data = df_data.dropna()
            print(df_data)
            list_of_cols = list(df_data.columns)
            matrix = df_data.to_numpy()
    
            # mcda
            weights = critic_weighting(matrix)

            if el == 0:
                saved_weights = copy.deepcopy(weights)
            else:
                saved_weights = np.vstack((saved_weights, weights))

            # TOPSIS
            if method == 'topsis':
                pref, weighted_matrix = TOPSIS(matrix, weights, types, norm_method, list_alt_names, list_of_cols, year)
                rankingPrep = np.argsort(-pref)
                rank = np.argsort(rankingPrep) + 1

                df_weights = pd.DataFrame(data = weights.reshape(1, -1), index = ['Weights'], columns = list_of_cols)
                df_types = pd.DataFrame(data = types.reshape(1, -1), index = ['Types'], columns = list_of_cols)

                # save weighted_matrix as df in csv
                df_weighted_matrix = pd.DataFrame(data = weighted_matrix, index = list_alt_names, columns = list_of_cols)
                df_weighted_matrix = pd.concat([df_weighted_matrix, df_weights, df_types], axis = 0)
                df_weighted_matrix['Efficiency'] = np.append(pref, np.repeat(np.nan, 2))
                df_weighted_matrix['Rank'] = np.append(rank, np.repeat(np.nan, 2))
                df_weighted_matrix.to_csv('output/df_weighted_matrix_topsis_' + str(year) + '.csv')


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

        df_saved_weights = pd.DataFrame(data = saved_weights, columns = list_of_cols)
        df_saved_weights.index = str_years
        df_saved_weights.index.name = 'Years'
        df_saved_weights.to_csv('output/all_weights_' + method + '.csv')

        print('Sortowanie wag:')
        df_rank_weights = pd.DataFrame()
        df_saved_weights = df_saved_weights.T
        for col in df_saved_weights.columns:
            rank = rank_preferences(df_saved_weights[col].to_numpy(), reverse = True)
            df_rank_weights[col] = rank

        df_rank_weights.index = list_of_cols
        df_rank_weights.columns = str_years
        df_rank_weights.index.name = 'Criteria'
        print(df_rank_weights)


if __name__ == "__main__":
    main()
    