## A number of single step models, prediciting one day into the future based on the last day of data / the last days of data
## They predict the closing_value of the day

#imports
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from utils import WindowGenerator
from utils import concat_data, compile_and_fit, plot
import os

RETRAIN = True
SHIFT = 150
MAX_EPOCHS = 60
CONV_WIDTH = 10
VAL_PERFORMANCE = {}
PERFORMANCE = {}
HISTORY = {}
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

class ResidualWrapper(tf.keras.Model):
  def __init__(self, model, label_index=None):
    super().__init__()
    self.model = model
    self.label_index = label_index

  def call(self, inputs, *args, **kwargs):
    delta = self.model(inputs, *args, **kwargs)

    # The prediction for each time step is the input
    # from the previous time step plus the delta
    # calculated by the model.
    if self.label_index: 
      return inputs[:, :, self.label_index][:,:,tf.newaxis] + delta
    return inputs + delta

def test(model, window, name):
    VAL_PERFORMANCE[name] = model.evaluate(window.val, return_dict=True)
    PERFORMANCE[name] = model.evaluate(window.test, verbose=0, return_dict=True)

def train_and_test(model, window, model_name, patience=5, retrain = RETRAIN):
  if model_name not in os.listdir('Training/Models') or retrain:
    HISTORY[model_name] = compile_and_fit(model, window, patience)
    model.save(f'Training/Models/{model_name}')
  else:
     model = tf.keras.models.load_model(f'Training/Models/{model_name}')
  test(model,window,model_name)

def main():
    #get data
    train_df, val_df, test_df, column_indices, num_features = concat_data('data', standard=False)
    # define windows
    single_step_window = WindowGenerator(
        train_df=train_df, val_df=val_df, test_df=test_df,
        input_width=1, label_width=1, shift=SHIFT,
        label_columns=['close'])

    multi_output_single_step_window = WindowGenerator(
        train_df=train_df, val_df=val_df, test_df=test_df,
        input_width=1, label_width=1, shift=SHIFT)
    
    wide_window = WindowGenerator(
        train_df=train_df, val_df=val_df, test_df=test_df,
        input_width=20, label_width=20, shift=SHIFT,
        label_columns=['close'])
    
    multi_output_wide_window = WindowGenerator(
        train_df=train_df, val_df=val_df, test_df=test_df,
        input_width=20, label_width=20, shift=SHIFT)
    
    conv_window = WindowGenerator(
        train_df=train_df, val_df=val_df, test_df=test_df,
        input_width=CONV_WIDTH,
        label_width=1,
        shift=SHIFT,
        label_columns=['close'])


    # Train the different single_input models

    #Baseline
    print('Baseline model')
    baseline = Baseline(label_index=column_indices['close'])
    train_and_test(baseline, single_step_window, 'single/baseline')
    print('Input shape:', wide_window.example[0].shape)
    print('Output shape:', baseline(wide_window.example[0]).shape)
    plt.show()
    #Linear
    print('linear model')
    linear = tf.keras.Sequential([tf.keras.layers.Dense(units=1)])
    train_and_test(linear, single_step_window, 'single/linear')


    # Dense Deep
    print('dense_model')
    dense = tf.keras.Sequential([
      tf.keras.layers.Dense(units=64, activation='relu'),
      tf.keras.layers.Dense(units=64, activation='relu'),
      tf.keras.layers.Dense(units=1)
      ])
    train_and_test(dense, single_step_window, 'single/dense')

    # Train the different multi_input models

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
    train_and_test(multi_step_dense, conv_window,'single/multi_step_dense')

    # Conv Model
    print('conv_model')
    conv_model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(filters=32,
                              kernel_size=(CONV_WIDTH,),
                              activation='relu'),
        tf.keras.layers.Dense(units=32, activation='relu'),
        tf.keras.layers.Dense(units=1),
    ])
    train_and_test(conv_model, conv_window, 'single/conv')

    # LSTM
    print('lstm')
    lstm_model = tf.keras.models.Sequential([
        # Shape [batch, time, features] => [batch, time, lstm_units]
        tf.keras.layers.LSTM(64, return_sequences=True),
        #tf.keras.layers.LSTM(64, return_sequences=True),
        # Shape => [batch, time, features]
        tf.keras.layers.Dense(units=1)
    ])
    train_and_test(lstm_model, wide_window, 'single/lstm')

    # res Net with single_output
    print('residual_lstm')
    residual_lstm_single = ResidualWrapper(
        tf.keras.Sequential([
        tf.keras.layers.LSTM(32, return_sequences=True),
        tf.keras.layers.Dense(
            1,
            # The predicted deltas should start small.
            # Therefore, initialize the output layer with zeros.
            kernel_initializer=tf.initializers.zeros())
    ]), label_index=column_indices['close'])
    train_and_test(residual_lstm_single, wide_window, 'single/residual_lstm_single', retrain=True)

    plot(val_performance=VAL_PERFORMANCE, performance=PERFORMANCE, plotname='single_step_single_output_models')
    VAL_PERFORMANCE.clear()
    PERFORMANCE.clear()

    # Train the different multi_Output models

    # baseline
    print('mulit_output_baseline')
    multi_baseline = Baseline()
    train_and_test(multi_baseline, multi_output_single_step_window, 'single/multi_baseline')

    # Dense
    print('multi_output_Dense')
    multi_output_dense = tf.keras.Sequential([
       tf.keras.layers.Flatten(),
       tf.keras.layers.Dense(64, activation='relu'),
       tf.keras.layers.Dense(64, activation='relu'),
       tf.keras.layers.Dense(num_features),
       tf.keras.layers.Reshape([1,-1])
    ])
    train_and_test(multi_output_dense, multi_output_wide_window, 'single/single_multi_output_dense')

    # Res Net with multiple outputs
    print('residual_lstm')
    residual_lstm = ResidualWrapper(
        tf.keras.Sequential([
        tf.keras.layers.LSTM(32, return_sequences=True),
        tf.keras.layers.Dense(
            num_features,
            # The predicted deltas should start small.
            # Therefore, initialize the output layer with zeros.
            kernel_initializer=tf.initializers.zeros())
    ]))
    train_and_test(residual_lstm, multi_output_wide_window, 'single/residual_lstm', retrain=True)

    plot(VAL_PERFORMANCE, PERFORMANCE, 'all_standard_single_step_multi_output_models')

if __name__ == '__main__':
    main()
    plt.show()