import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import seaborn as sns

FILENAME = 'data/apple'

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
    train_df = df[0:int(n*0.7)]
    val_df = df[int(n*0.7):int(n*0.9)]
    test_df = df[int(n*0.9):]

    num_features = df.shape[1]

    # Standardize
    train_mean = train_df.mean()
    train_std = train_df.std()

    train_df = (train_df - train_mean) / train_std
    val_df = (val_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std

    # plot 
    df_std = (df - train_mean) / train_std
    df_std = df_std.melt(var_name='Column', value_name='Normalized')
    plt.figure(figsize=(12, 6))
    ax = sns.violinplot(x='Column', y='Normalized', data=df_std)
    _ = ax.set_xticklabels(df.keys(), rotation=90)

    return train_df, val_df, test_df, column_indices, num_features

def main():
    return_data()
if __name__ == '__main__':
    main()