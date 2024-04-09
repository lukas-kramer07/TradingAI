## A number of single step models, prediciting one day into the future based on the last day of data
## They predict the closing_value of the day

#imports
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from utils.WindowGen import WindowGenerator
from utils.dataengineering import return_data

MAX_EPOCHS = 60
VAL_PERFORMANCE = {}
PERFORMANCE = {}

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


def compile_and_fit(model, window, patience=5):
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
    train_df, val_df, test_df, column_indices, num_features = return_data()
    single_step_window = WindowGenerator(
        train_df=train_df, val_df=val_df, test_df=test_df,
        input_width=1, label_width=1, shift=1,
        label_columns=['close'])
    wide_window = WindowGenerator(
        train_df=train_df, val_df=val_df, test_df=test_df,
        input_width=20, label_width=20, shift=1,
        label_columns=['close'])

    # Train the different models

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

if __name__ == '__main__':
    main()
    plt.show()