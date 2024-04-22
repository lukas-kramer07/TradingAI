import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import seaborn as sns
import os

FILENAME = 'data/GenElectric'

def plot(df, date_time):
    plot_cols = ['open', 'change', 'volume']
    plot_features = df[plot_cols]
    plot_features.index = date_time
    _ = plot_features.plot(subplots=True)

    plot_features = df[plot_cols][:20]
    plot_features.index = date_time[:20]
    _ = plot_features.plot(subplots=True)
    plt.show()
def return_data(filename = FILENAME):
    df = pd.read_pickle(filename)
    df.pop('label')
    date_time = pd.to_datetime(df.pop('date'), format='%Y-%m-%d')
    #plot(df, date_time)
    #print(df.describe().transpose())

    # Split the data
    column_indices = {name: i for i, name in enumerate(df.columns)}

    n = len(df)
    train_df = df[0:int(n*0.8)]
    val_df = df[int(n*0.8):]
    test_df = val_df

    num_features = df.shape[1]



    return train_df, val_df, test_df, column_indices, num_features

def iterate_files(folder_path):
    for filename in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, filename)):
            yield filename
def concat_data(folder):
    # concat all data
    standard, train_df, val_df, test_df= None,[],[],[]
    for filename in iterate_files(folder):
        train_df1, val_df1, test_df1, column_indices, num_features = return_data(filename=f'{folder}/{filename}')
        standard = pd.concat([standard, train_df1])
        train_df.append(train_df1)
        val_df.append(train_df1)
        test_df.append(train_df1)
    
    #standardize
    mean = standard.mean()
    std = standard.std()

    for df in train_df: 
        df = (df - mean)/std
    for df in val_df:
        df = (df - mean)/std
    for df in test_df:
        df = (df-mean)/std
        
    """# plot 
    df_std = (val_df - train_mean) / train_std
    df_std = df_std.melt(var_name='Column', value_name='Normalized')
    plt.figure(figsize=(12, 6))
    ax = sns.violinplot(x='Column', y='Normalized', data=df_std)
    _ = ax.set_xticklabels(val_df.keys(), rotation=90)     
    plt.show()"""

    return train_df, val_df, test_df, column_indices, num_features
def main():
    return_data()
    plt.show()
if __name__ == '__main__':
    main()