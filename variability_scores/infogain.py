import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import matplotlib

import seaborn as sns
import copy
from scipy.stats import pearsonr

import sys

sys.path.append('../../daria')

from daria import DARIA

# for plotting correlation between variability of different methods
def draw_heatmap(df_new_heatmap, title):
    #plt.figure(figsize = (8,5))
    sns.set(font_scale=1.4)
    heatmap = sns.heatmap(df_new_heatmap, annot=True, fmt=".4f", cmap="BuPu",
                          linewidth=0.5, linecolor='w')
    plt.yticks(va="center")
    plt.xlabel('Methods')
    plt.title('Correlation: ' + title)
    plt.tight_layout()



diff_measure = 'Gini coefficient'
methods = ['topsis',
           'vikor',
           'comet',
           ]

list_alt_names = []
for i in range(1, 26 + 1):
    list_alt_names.append(r'$A_{' + str(i) + '}$')

df_final = pd.DataFrame()
# for plot
df_varia = pd.DataFrame()
df_varia['Ai'] = list_alt_names

df_varia_fin = pd.DataFrame()
df_varia_fin['Ai'] = list_alt_names

for el, met in enumerate(methods):
    df_data = pd.read_csv('data/' + met + '.csv')
    df_data = df_data.set_index('Ai')

    # names of columns in dataframe
    print(list(df_data.columns))

    # dataframe for variability evaluation (transposed)
    df = df_data.T
    print(df)
    matrix = df.to_numpy()
    # calculate variability
    

    descending = True
    if met == 'vikor':
        descending = False

    # calculate efficiencies variability using DARIA methodology
    daria = DARIA()
    # calculate variability values
    var = daria._gini(matrix)
    # calculate variability directions
    dir_list, dir_class = daria._direction(matrix, descending)

    # for plot
    df_varia[met.upper()] = list(var)
    # for next study
    df_varia_fin[met.upper()] = list(var)
    df_varia_fin[met.upper() + ' dir'] = dir_class

    df_results = pd.DataFrame()
    df_results['Ai'] = list(df.columns)
    df_results['Variability'] = list(var)
    # list of directions
    df_results['dir list'] = dir_list
    
    # sorting result dataframe
    df_results = df_results.sort_values(by = 'Variability', ascending = False)

    df_final[r'$A_{' + str(el + 1) + '}$'] = list(df_results['Ai'])
    df_final[met.upper()] = list(df_results['Variability'])
    # list of directions
    df_final[r'$A_{' + str(el + 1) + '} dir$'] = list(df_results['dir list'])

# final results to csv
print(df_final)
df_final.to_csv('output/rankings_' + diff_measure + '.csv')

df_varia_fin = df_varia_fin.set_index('Ai')
df_varia_fin.to_csv('output/FINAL_' + diff_measure + '.csv')

# final results for plot
#todraw = df_varia / df_varia.sum(axis = 0)
df_varia = df_varia.set_index('Ai')


# plot bar plot stacked
# transpose
df_varia = df_varia.T
# normalization with sum method
#df_varia = df_varia / df_varia.sum(axis = 0)
df_varia = df_varia.fillna(0)
todraw = df_varia.T
#
print(todraw)

matplotlib.rc_file_defaults()
ax = todraw.plot(kind='bar', stacked=True, edgecolor = 'black', figsize = (15,5))
ax.set_xlabel('Countries', fontsize = 14)
#ax.set_ylabel('Normalized values of ' + diff_measure, fontsize = 14)
ax.set_ylabel('Values of ' + diff_measure, fontsize = 14)
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
ncol=3, mode="expand", borderaxespad=0., edgecolor = 'black', fontsize = 14)
ax.set_xticklabels(list(todraw.index), rotation = 'horizontal')
ax.tick_params(axis='both', labelsize=14)
ax.set_xlim(-1, len(todraw) + 0.5)
ax.grid(True)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig('output/rankings_' + diff_measure + '.pdf')
plt.show()
