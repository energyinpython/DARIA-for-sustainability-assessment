import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import copy


#gini coefficient
def gini(DM):
    t, n = np.shape(DM)
    G = np.zeros(n)
    # iteration over criteria j=1, 2, ..., n
    for j in range(0, n):
        Yj = np.zeros(t)
        # iteration over periods p=1, 2, ..., t
        if np.mean(DM[:, j]) != 0:
            for p in range(0, t):
                for k in range(0, t):
                    Yj[p] += np.abs(DM[p, j] - DM[k, j]) / (2 * t**2 * (np.sum(DM[:, j]) / t))
        else:
            for p in range(0, t):
                for k in range(0, t):
                    Yj[p] += np.abs(DM[p, j] - DM[k, j]) / (t**2 - t)
        G[j] = np.sum(Yj)
    return G


# direction of variability
def direction(DM, crit_types):
    t, n = np.shape(DM)
    direction_list = []
    # iteration over criteria j=1, 2, ..., n
    for j in range(n):
        thresh = 0
        # iteration over periods p=1, 2, ..., t
        for p in range(1, t):
            thresh += DM[p, j] - DM[p - 1, j]
        # if criterion is cost type
        if crit_types[j] == -1:
            if thresh < 0:
                direction_list.append(r' $\uparrow$')
            elif thresh > 0:
                direction_list.append(r' $\downarrow$')
            else:
                direction_list.append(r' $=$')
        # if criterion is profit type
        else:
            if thresh > 0:
                direction_list.append(r' $\uparrow$')
            elif thresh < 0:
                direction_list.append(r' $\downarrow$')
            else:
                direction_list.append(r' $=$')
    return direction_list



# entropy gini coeff std
diff_measure = 'gini coeff'

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
    # calculate variability
    var = gini(matrix)
    dir_var = direction(matrix, types)

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
# plot
# final variability results to plot
df_varia = df_varia.set_index('Ci')
print(df_varia)

#normalization with sum method
#todraw = df_varia / df_varia.sum(axis = 0)
# we do not normalize
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
plt.savefig('output/dataset_' + diff_measure + '.pdf')
plt.show()
