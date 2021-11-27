import numpy as np
import pandas as pd
import os

#main
# name of folder with datasets
path = 'DATASET'
# file with wanted countries names in rows of dataframe
file_country_names = 'Country_names.csv'
path_country_names = os.path.join(path, file_country_names)
df_country_names = pd.read_csv(path_country_names)
# list with wanted countries names in rows of dataframe
list_country_names = list(df_country_names['Name'])
# list with wanted years in columns of dataframe
list_years = ['TIME', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020']


# dataframe for all results
df_all = pd.DataFrame()
# there are 10 files with 10 criteria (C1 - C10)
for i in range(1, 11):
    file = 'sdg11_C' + str(i) + '.csv'
    pathfile = os.path.join(path, file)
    data = pd.read_csv(pathfile)
    
    # columns in given files with dataset for given criterion
    current_columns = list(data.columns)
    
    # selection of wanted columns in given dataset
    wanted_columns = []
    for el in current_columns:
        if el in list_years:
            wanted_columns.append(el)

    df = data[wanted_columns]
    df = df.reset_index()
    df = df.drop(['index'], axis = 1)
    df = df.set_index('TIME')
    # selection of wanted countries in given dataset
    df = df.loc[list_country_names]

    # prepare list of columns with given criterion name
    crits = []
    for j in range(df.shape[1]):
        crits.append('C' + str(i))

    df_crit = pd.DataFrame([crits])
    
    df_crit.columns = wanted_columns[1:]
    
    new_df = pd.concat([df_crit, df])
    # write to csv cured dataset for given criterion
    df.to_csv('DATASET/output_cured/' + 'data_C' + str(i) + '.csv')
    # write to dataframe each cured dataset for subsequent criteria
    df_all = pd.concat([df_all, new_df], axis = 1)

# final dataframe with complete results
print(df_all)
df_all.to_csv('DATASET/output_all/' + 'data_all.csv')