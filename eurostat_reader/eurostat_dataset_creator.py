import numpy as np
import pandas as pd
import os
import copy

def main():
    # here select the year for which you want to create the decision matrix
    year = '2015'
    path = 'output_all'
    file_name = 'data_all.csv'
    path_file = os.path.join(path, file_name)

    df = pd.read_csv(path_file, header = None)
    rows, cols = df.shape

    for j in range(1, cols):
        if df.iloc[0, j] == year:
            print(df.iloc[0, j])
            for i in range(2, rows):
                if pd.isna(df.iloc[i, j]):
                    print('NaN')
                    for k in range(j, 0, -1):
                        if df.iloc[1, j] != df.iloc[1, k]:
                            break
                        if pd.notna(df.iloc[i, k]):
                            df.iloc[i, j] = df.iloc[i, k]
                            break

    # df.to_csv('output_all/filled_' + year + '.csv')
    new_df = copy.deepcopy(df)
    # rename columns by year names
    new_df.columns = new_df.iloc[0, :]
    # remove row with year names
    new_df = new_df.drop([0])
    # select only columns with the year that you are interested in
    new_df = new_df[['Country', year]]
    # set names of countries as the index of dataframe
    new_df = new_df.set_index('Country')
    # rename columns by criteria symbols
    new_df.columns = new_df.iloc[0, :]
    # remove the old unnecessary row with criteria symbols
    new_df = new_df.iloc[1:, :]
    print(new_df)

    new_df.to_csv('output_all/data_' + year + '.csv')

if __name__ == '__main__':
    main()