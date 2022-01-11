import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import copy

import sys

sys.path.append('../../daria')

from daria import DARIA

def main():
    diff_measure = 'Gini coefficient'

    df_country_names = pd.read_csv('data/country_names.csv')
    # list with wanted countries names in rows of dataframe
    list_country_names = list(df_country_names['Name'])

    data = pd.read_csv('data/dataset.csv')
    data = data.set_index('Country')
    types = data.iloc[len(data) - 1, :-1].to_numpy()
    df_data = data.iloc[:-1, :-1]
    df_data = df_data.dropna()

    # ready dataframe
    print(df_data)
    print(types)

    # names of columns in dataframe
    print(list(df_data.columns))
    list_of_cols = []
    for i in range(len(list(df_data.columns))):
        list_of_cols.append(r'$C_{' + str(i + 1) + '}$')


    df_countries = pd.DataFrame()

    #dataframe with variability values without sorting
    df_varia = pd.DataFrame()
    df_varia['Ci'] = list_of_cols

    # for given subsequent countries
    for el, country in enumerate(list_country_names):
        # choose country
        df_data_country = df_data.loc[country]
        matrix = df_data_country.to_numpy()

        # calculate criteria variability using DARIA methodology
        daria = DARIA()
        # calculate variability values
        var = daria._gini_criteria(matrix)
        # calcultate variability direction
        dir_var = daria._direction_criteria(matrix, types)

        # for plot
        df_varia[r'$A_{' + str(el + 1) + '}$'] = list(var)

        # single dataframe for given country
        df_countries_part = pd.DataFrame()

        new_list_of_cols = []
        for i, col in enumerate(list_of_cols):
            new_list_of_cols.append(col + dir_var[i])

        df_countries_part['Ci'] = new_list_of_cols
        df_countries_part[country] = list(var)

        # sort results for presentation
        df_countries_part = df_countries_part.sort_values(by = country, ascending = False)

        # joining results for given countries to one dataframe
        df_countries[r'$A_{' + str(el + 1) + '} crit$'] = list(df_countries_part['Ci'])
        df_countries[r'$A_{' + str(el + 1) + '}$'] = list(df_countries_part[country])

    # save final results to csv
    print(df_countries)
    df_countries.to_csv('output/dataset_' + diff_measure + '.csv')

    # for plot results (column chart)
    # final variability results to plot
    df_varia = df_varia.set_index('Ci')
    print(df_varia)

    #normalization with sum method
    #todraw = df_varia / df_varia.sum(axis = 0)
    # but here we do not use normalization
    todraw = copy.deepcopy(df_varia)
    todraw = todraw.T

    ax = todraw.plot(kind='bar', stacked=True, edgecolor = 'black', figsize = (15,5))
    ax.set_xlabel('Countries', fontsize = 14)
    ax.set_ylabel('Values of ' + diff_measure, fontsize = 14)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
    ncol=10, mode="expand", borderaxespad=0., edgecolor = 'black', fontsize = 14)
    ax.set_xticklabels(list(df_varia.columns), rotation = 'horizontal')
    ax.set_xlim(-1.5, len(todraw) + 0.5)
    ax.tick_params(axis='both', labelsize=14)
    ax.grid(True)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig('output/dataset_' + diff_measure + '.png')
    plt.show()

if __name__ == '__main__':
    main()
