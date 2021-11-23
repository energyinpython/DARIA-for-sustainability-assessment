import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import copy
from scipy.stats import pearsonr


def draw_heatmap(df_new_heatmap, title):
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

for year in years:
    data = pd.read_csv('results_rank_' + year + '.csv')
    data = data.iloc[:, 1:]
    data = data.set_index('Ai')
    print(data)

    method_types = list(data.columns)

    dict_new_heatmap_pearson = {'TOPSIS': [], 
                            'VIKOR': [], 
                            'COMET': [],
                            }

    # heatmaps for correlations coefficients
    for i in method_types[::-1]:
        for j in method_types:
            corr_p, _ = pearsonr(data[i], data[j])
            dict_new_heatmap_pearson[j].append(corr_p)
        
    df_new_heatmap_pearson = pd.DataFrame(dict_new_heatmap_pearson, index = method_types[::-1])
    df_new_heatmap_pearson.columns = method_types

    # correlation matrix with pearson coefficient
    draw_heatmap(df_new_heatmap_pearson, r'$Pearson$' + ', Year: ' + year)
    plt.savefig('pearson_' + str(year) + '.pdf')
    plt.show()
