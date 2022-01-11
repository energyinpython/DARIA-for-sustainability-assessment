import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    '''
    #https://www.kaggle.com/typewind/draw-a-radar-chart-with-python-in-a-simple-way
    # https://www.codegrepper.com/code-examples/python/python+radar+chart
    met = 'topsis'
    dane = pd.read_csv(met + '.csv', index_col='Ai')

    # first approach
    fig=plt.figure()
    ax = fig.add_subplot(111, polar=True)
    for ind in list(dane.index):
        # set labels - points on the perimeter of the chart
        # labels - years
        labels=np.array(list(dane.columns))
        # alternatives
        stats=dane.loc[ind, labels].values

        angles=np.linspace(0, 2*np.pi, len(labels), endpoint=False)
        # close the plot
        stats=np.concatenate((stats,[stats[0]]))
        angles=np.concatenate((angles,[angles[0]]))
    
        lista = list(dane.columns)
        lista.append(dane.columns[0])
        labels=np.array(lista)

        ax.plot(angles, stats, 'o-', linewidth=2)
        #ax.fill(angles, stats, alpha=0.25)
    ax.set_thetagrids(angles * 180/np.pi, labels)
    ax.grid(True)
    #plt.legend(dane.index, loc = 'right')
    plt.legend(dane.index, bbox_to_anchor=(1.1, 0.95, 0.3, 0.2), loc='upper left')
    plt.show()
    '''


    # second approach
    #labels were columns (years) but now we want alternatives
    #linie to byly alternatives a chcemy metody czyli labele - kolumny - dane.columns
    #as labels we want dane.index

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

if __name__ == '__main__':
    main()