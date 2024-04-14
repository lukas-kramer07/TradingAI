## A number of single step models, prediciting one day into the future based on the last day of data / the last days of data
## They predict the closing_value of the day

#imports
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from utils import WindowGenerator
from utils import concat_data

import os


MAX_EPOCHS = 60
CONV_WIDTH = 10
VAL_PERFORMANCE = {}
PERFORMANCE = {}
INIT = tf.initializers.zeros()
## Models
class Baseline(tf.keras.Model):
  def __init__(self, label_index=None):
    super().__init__()
    self.label_index = label_index

  def call(self, inputs):
    if self.label_index is None:
      return inputs
    result = inputs[:, :, self.label_index]
    return result[:, :, tf.newaxis]


def compile_and_fit(model, window, patience=10):
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')

  model.compile(loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=[tf.keras.metrics.MeanAbsoluteError()])

  history = model.fit(window.train, epochs=MAX_EPOCHS,
                      validation_data=window.val,
                      callbacks=[early_stopping])
  return history

def test(model, window, name):
    VAL_PERFORMANCE[name] = model.evaluate(window.val, return_dict=True)
    PERFORMANCE[name] = model.evaluate(window.test, verbose=0, return_dict=True)



def main():
    #get data
    train_df, val_df, test_df, column_indices, num_features = concat_data('data')
    single_step_window = WindowGenerator(
        train_df=train_df, val_df=val_df, test_df=test_df,
        input_width=1, label_width=1, shift=1,
        label_columns=['close'])
    #single_step_window.plot()
    #plt.suptitle("Given 1 day of inputs, predict 1 day into the future")

    wide_window = WindowGenerator(
        train_df=train_df, val_df=val_df, test_df=test_df,
        input_width=20, label_width=20, shift=1,
        label_columns=['close'])
    
    conv_window = WindowGenerator(
        train_df=train_df, val_df=val_df, test_df=test_df,
        input_width=CONV_WIDTH,
        label_width=1,
        shift=1,
        label_columns=['close'])
    #conv_window.plot()
    #plt.suptitle("Given 3 days of inputs, predict 1 day into the future.")
    # Train the different single_step models

    #Baseline
    print('Baseline model')
    baseline = Baseline(label_index=column_indices['close'])
    baseline_history = compile_and_fit(baseline, single_step_window)
    test(baseline, single_step_window, 'baseline')

    #Linear
    print('linear model')
    linear = tf.keras.Sequential([tf.keras.layers.Dense(units=1)])
    linear_history = compile_and_fit(linear, single_step_window)
    test(linear, single_step_window, 'linear')

    # Dense Deep
    print('dense_model')
    dense = tf.keras.Sequential([
      tf.keras.layers.Dense(units=64, activation='relu'),
      tf.keras.layers.Dense(units=64, activation='relu'),
      tf.keras.layers.Dense(units=1)
      ])
    dense_history = compile_and_fit(dense, single_step_window, patience=15)
    test(dense, single_step_window, 'dense')


    # Train the different multi_step models

    #Multi_Dense
    print('multi Dense')
    multi_step_dense = tf.keras.Sequential([
        # Shape: (time, features) => (time*features)
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dense(units=1),
        # Add back the time dimension.
        # Shape: (outputs) => (1, outputs)
        tf.keras.layers.Reshape([1,-1]), #-1 is used for shape infrence
    ])
    multi_step_dense_history = compile_and_fit(multi_step_dense, conv_window)
    test(multi_step_dense, conv_window,'multi_step_dense')

    # Conv Model
    conv_model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(filters=32,
                              kernel_size=(CONV_WIDTH,),
                              activation='relu'),
        tf.keras.layers.Dense(units=32, activation='relu'),
        tf.keras.layers.Dense(units=1),
    ])
    conv_history = compile_and_fit(conv_model, conv_window)
    test(conv_model, conv_window, 'conv')

    # LSTM
    lstm_model = tf.keras.models.Sequential([
        # Shape [batch, time, features] => [batch, time, lstm_units]
        tf.keras.layers.LSTM(64, return_sequences=True),
        #tf.keras.layers.LSTM(64, return_sequences=True),
        # Shape => [batch, time, features]
        tf.keras.layers.Dense(units=1)
    ])
    lstm_history = compile_and_fit(lstm_model, conv_window)
    test(lstm_model, conv_window, 'lstm')

    # Plot
    x = np.arange(len(PERFORMANCE))
    width = 0.3
    metric_name = 'mean_absolute_error'
    val_mae = [v[metric_name] for v in VAL_PERFORMANCE.values()]
    test_mae = [v[metric_name] for v in PERFORMANCE.values()]

    plt.ylabel('mean_absolute_error [T (degC), normalized]')
    plt.bar(x - 0.17, val_mae, width, label='Validation')
    plt.bar(x + 0.17, test_mae, width, label='Test')
    plt.xticks(ticks=x, labels=PERFORMANCE.keys(),
              rotation=45)
    _ = plt.legend()

if __name__ == '__main__':
    main()
    plt.show()