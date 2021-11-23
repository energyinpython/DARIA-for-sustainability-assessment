import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import copy
from scipy.stats import pearsonr

def draw_heatmap(df_new_heatmap, title):
    #plt.figure(figsize = (8,5))
    sns.set(font_scale=1.4)
    heatmap = sns.heatmap(df_new_heatmap, annot=True, fmt=".4f", cmap="BuPu",
                          linewidth=0.5, linecolor='w')
    plt.yticks(va="center")
    plt.xlabel('Methods')
    plt.title('Correlation: ' + title)
    plt.tight_layout()

methods = [
    'TOPSIS',
    'VIKOR',
    'COMET',
    ]

# data with alternatives' rankings' variability values calculated with Gini coeff and directions
G_df = pd.read_csv('data/FINAL_gini coeff.csv')
G_df = G_df.iloc[:, 1:]
G_df = G_df.set_index('Ai')

# data with alternatives' efficiency of performance calculated for the recent period
E_df = pd.read_csv('data/results_pref_2019.csv')
E_df = E_df.iloc[:, 1:]
E_df = E_df.set_index('Ai')

df_final_E = pd.DataFrame()
df_final_E['Ai'] = list(E_df.index)

df_final_ranks = pd.DataFrame()
df_final_ranks['Ai'] = list(E_df.index)

for met in methods:
    E = E_df[met].to_numpy()
    # VIKOR has ascending ranking from prefs
    if met == 'VIKOR':
        E = 1 - E
    G = G_df[met].to_numpy()
    dir = G_df[met + ' dir'].to_numpy()

    fin_E = E + G * dir

    if met == 'VIKOR':
        fin_E = 1 - fin_E
        rankingPrep = np.argsort(fin_E)
    else:
        rankingPrep = np.argsort(-fin_E)
    df_final_E[met] = list(fin_E)

    rank = np.argsort(rankingPrep) + 1

    df_final_ranks[met] = list(rank)

# final efficiencies
df_final_E = df_final_E.set_index('Ai')
# final rankings
df_final_ranks = df_final_ranks.set_index('Ai')

df_writer_all = pd.concat([df_final_E, df_final_ranks], axis = 1)
df_writer_all.to_csv('output/results_all.csv')


#plot
#plots final rankings
#radar
dane = copy.deepcopy(df_final_ranks)
fig=plt.figure()
ax = fig.add_subplot(111, polar=True)

for col in list(dane.columns):
    labels=np.array(list(dane.index))
    stats = dane.loc[labels, col].values

    angles=np.linspace(0, 2*np.pi, len(labels), endpoint=False)
    # close the plot
    stats=np.concatenate((stats,[stats[0]]))
    angles=np.concatenate((angles,[angles[0]]))
    
    lista = list(dane.index)
    lista.append(dane.index[0])
    labels=np.array(lista)

    ax.plot(angles, stats, '-', linewidth=1)
    ax.fill_between(angles, stats, alpha=0.05)
    
ax.set_thetagrids(angles * 180/np.pi, labels)
ax.grid(True)
ax.set_axisbelow(True)
plt.legend(dane.columns, bbox_to_anchor=(1.1, 0.95, 0.3, 0.2), loc='upper left')
plt.title('Final rankings')
plt.tight_layout()
plt.savefig('output/radar_' + 'rankings' + '.pdf')
plt.show()

# correlations
data = copy.deepcopy(df_final_ranks)
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
draw_heatmap(df_new_heatmap_pearson, r'$Pearson$')
plt.savefig('output/pearson_final_ranks' + '.pdf')
plt.show()
