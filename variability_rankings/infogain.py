import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import matplotlib

import seaborn as sns
import copy
from scipy.stats import pearsonr

# for plotting correlation between variability of different methods
def draw_heatmap(df_new_heatmap, title):
    sns.set(font_scale=1.4)
    heatmap = sns.heatmap(df_new_heatmap, annot=True, fmt=".4f", cmap="BuPu",
                          linewidth=0.5, linecolor='w')
    plt.yticks(va="center")
    plt.xlabel('Methods')
    plt.title('Correlation: ' + title)
    plt.tight_layout()


#gini coefficient
def gini(R):
    t, m = np.shape(R)
    G = np.zeros(m)
    # iteration over alternatives i=1, 2, ..., m
    for i in range(0, m):
        # iteration over periods p=1, 2, ..., t
        Yi = np.zeros(t)
        if np.mean(R[:, i]) != 0:
            for p in range(0, t):
                for k in range(0, t):
                    Yi[p] += np.abs(R[p, i] - R[k, i]) / (2 * t**2 * (np.sum(R[:, i]) / t))
        else:
            for p in range(0, t):
                for k in range(0, t):
                    Yi[p] += np.abs(R[p, i] - R[k, i]) / (t**2 - t)
        G[i] = np.sum(Yi)
    return G


# direction of variability
def direction(R):
    t, m = np.shape(R)
    direction_list = []
    dir_class = []
    # iteration over alternatives i=1, 2, ..., m
    for i in range(m):
        thresh = 0
        # iteration over periods p=1, 2, ..., t
        for p in range(1, t):
            thresh += R[p, i] - R[p - 1, i]
        # there are rankings so crits are cost type
        if thresh < 0:
            direction_list.append(r'$\uparrow$')
            dir_class.append(1)
        elif thresh > 0:
            direction_list.append(r'$\downarrow$')
            dir_class.append(-1)
        else:
            direction_list.append(r'$=$')
            dir_class.append(0)
    return direction_list, dir_class


diff_measure = 'gini coeff'
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
    var = gini(matrix)

    dir_list, dir_class = direction(matrix)

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
df_varia_fin.to_csv('output/FINAL_' + diff_measure + '.csv')

# correlations of variabilities - plot
# final results for plot
#todraw = df_varia / df_varia.sum(axis = 0)
df_varia = df_varia.set_index('Ai')

method_types = list(df_varia.columns)

dict_new_heatmap_pearson = {'TOPSIS': [], 
                        'VIKOR': [], 
                        'COMET': [],
                        }

# heatmaps for correlations coefficients
for i in method_types[::-1]:
    for j in method_types:
        corr_p, _ = pearsonr(df_varia[i], df_varia[j])
        dict_new_heatmap_pearson[j].append(corr_p)

df_new_heatmap_pearson = pd.DataFrame(dict_new_heatmap_pearson, index = method_types[::-1])
df_new_heatmap_pearson.columns = method_types

# correlation matrix with pearson coefficient
draw_heatmap(df_new_heatmap_pearson, r'$Pearson$')
plt.savefig('output/pearson_' + diff_measure + '.pdf')
plt.show()


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
