import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


#labels are alternatives (dane.index)
#lines are methods (dane.columns)

methods = ['topsis',
           'vikor',
           'comet',
           ]

for met in methods:
    dane = pd.read_csv(met + '.csv', index_col='Ai')

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
    plt.title(met.upper())
    plt.tight_layout()
    plt.savefig('radar_' + met + '.pdf')
    plt.show()
