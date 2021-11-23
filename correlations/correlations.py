import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#from itertools import product, combinations
import seaborn as sns
#from pandas.plotting import scatter_matrix
#import os
import copy
#import matplotlib
#import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr


# weighted Spearman rW
def weighted_spearman(R, Q):
    N = len(R)
    denom = N**4 + N**3 - N**2 - N
    reszta = (N-R+1)+(N-Q+1)
    suma = 6*sum((R-Q)**2*reszta)
    rW = 1-(suma/denom)
    return rW

# WS Similarity coefficient
def coeff_WS(R, Q):
    sWS = 0
    N = len(R)
    for i in range(N):
        sWS += 2**(-int(R[i]))*(abs(R[i]-Q[i])/max(abs(R[i] - 1), abs(R[i] - N)))
    WS = 1 - sWS
    return WS


def draw_heatmap(df_new_heatmap, title):
    #plt.figure(figsize = (8,5))
    sns.set(font_scale=1.4)
    heatmap = sns.heatmap(df_new_heatmap, annot=True, fmt=".4f", cmap="BuPu",
                          linewidth=0.5, linecolor='w')
    plt.yticks(va="center")
    plt.xlabel('Methods')
    plt.title('Correlation: ' + title)
    plt.tight_layout()

years = ['2015',
         '2016',
         '2017',
         '2018',
         '2019',
         ]

#years = ['2015']

for year in years:
    data = pd.read_csv('results_rank_' + year + '.csv')
    data = data.iloc[:, 1:]
    data = data.set_index('Ai')
    print(data)

    method_types = list(data.columns)

    dict_new_heatmap_rw = {'TOPSIS': [], 
                            'VIKOR': [], 
                            'COMET': [],
                            }

    #dict_new_heatmap_ws = copy.deepcopy(dict_new_heatmap_rw)

    dict_new_heatmap_pearson = copy.deepcopy(dict_new_heatmap_rw)


    # heatmaps for correlations coefficients
    for i in method_types[::-1]:
        for j in method_types:
            print('i: ', i, ' j: ', j)
            #dict_new_heatmap_rw[j].append(weighted_spearman(data[i], data[j]))
            #dict_new_heatmap_ws[j].append(coeff_WS(data[i], data[j]))
            corr_p, _ = pearsonr(data[i], data[j])
            dict_new_heatmap_pearson[j].append(corr_p)
        

    '''
    df_new_heatmap_rw = pd.DataFrame(dict_new_heatmap_rw, index = method_types[::-1])
    df_new_heatmap_rw.columns = method_types

    df_new_heatmap_ws = pd.DataFrame(dict_new_heatmap_ws, index = method_types[::-1])
    df_new_heatmap_ws.columns = method_types
    '''

    df_new_heatmap_pearson = pd.DataFrame(dict_new_heatmap_pearson, index = method_types[::-1])
    df_new_heatmap_pearson.columns = method_types

    # correlation matrix with pearson coefficient
    draw_heatmap(df_new_heatmap_pearson, r'$Pearson$' + ', Year: ' + year)
    plt.savefig('pearson_' + str(year) + '.pdf')
    plt.show()

    '''
    # correlation matrix with rw coefficient
    draw_heatmap(df_new_heatmap_rw, r'$r_w$')
    plt.savefig('rw_' + str(year) + '.pdf')
    plt.show()

    # correlation matrix with WS coefficient
    draw_heatmap(df_new_heatmap_ws, r'$WS$')
    plt.savefig('ws_' + str(year) + '.pdf')
    plt.show()
    '''