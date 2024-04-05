import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from WindowGen import WindowGenerator

FILENAME = 'data/apple'

def plot(df):
    plot_cols = ['open', 'change', 'volume']
    df.pop('label')
    date_time = pd.to_datetime(df.pop('date'), format='%Y-%m-%d')
    plot_features = df[plot_cols]
    plot_features.index = date_time
    _ = plot_features.plot(subplots=True)

    plot_features = df[plot_cols][:20]
    plot_features.index = date_time[:20]
    _ = plot_features.plot(subplots=True)
    plt.show()

def main():
    df = pd.read_pickle(FILENAME)
    plot(df)
    print(df.describe().transpose())

    # Split the data
    column_indices = {name: i for i, name in enumerate(df.columns)}

    n = len(df)
    train_df = df[0:int(n*0.7)]
    val_df = df[int(n*0.7):int(n*0.9)]
    test_df = df[int(n*0.9):]

    num_features = df.shape[1]

    w2 = WindowGenerator(input_width=6, label_width=1, shift=1,
                         train_df=train_df, val_df=val_df, test_df=test_df,
                            label_columns=['open'])
    w2.plot()

if __name__ == '__main__':
    main()