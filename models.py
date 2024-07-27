# Advanced LSTM model specialized on prediciting stock prices

#imports
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from single_step_models import ResidualWrapper
from utils import WindowGenerator
from utils import concat_data, plot
from multi_step_models import LastStepBaseline
import os
import keras
import pprint
RETRAIN = False
VAL_PERFORMANCE = {}
PERFORMANCE = {}
HISTORY = {}
IN_STEPS=1000

class Baseline(keras.Model):
  def __init__(self, output_arr):
    super().__init__()
    self.output_arr = output_arr
  def call(self, inputs):
    batch_size = tf.shape(inputs)[0]
    return tf.tile(tf.convert_to_tensor([self.output_arr], dtype=tf.float32), [batch_size,1])


def main():
    #get data
    train_df, val_df, test_df, column_indices, num_features = concat_data('data', standard=False)
    # define windows
    window = WindowGenerator(train_df=train_df, val_df = val_df, test_df=test_df,
                                    input_width=IN_STEPS,
                                    shift=150, label_columns=['c'])
    print(window.example)

    # Training
    # Baseline Models
    baselines = {
        'Baseline_hold': [0, 0, 1, 0, 0],
        'Baseline_buy': [0, 1, 0, 0, 0],
        'Baseline_sell': [0, 0, 0, 1, 0],
        'Baseline_strong_buy': [1, 0, 0, 0, 0],
        'Baseline_strong_sell': [0, 0, 0, 0, 1],
        'Baseline_equal': [0.2,0.2,0.2,0.2,0.2]
    }
    
    for name, output_arr in baselines.items():
        print(name)
        baseline_model = Baseline(output_arr)
        train_and_test(baseline_model, window, name, epochs=1)

    print('linear model')
    linear_model = keras.Sequential([
        keras.layers.Flatten(),
        keras.layers.Dense(units=5, activation='softmax')
      ])
    train_and_test(linear_model, window, 'Linear')

    print('deep model')
    deep_model = keras.Sequential([
       keras.layers.Flatten(),
       keras.layers.Dense(128, activation='relu'),
       keras.layers.Dropout(0.3),
       keras.layers.Dense(64, activation='relu'),
       keras.layers.Dense(5, activation='softmax')
    ])
    train_and_test(deep_model, window, 'Deep')

    print('conv_model')
    conv_model = keras.Sequential([
       keras.layers.GaussianNoise(stddev=0.2),
       keras.layers.Conv1D(32, 3, activation='relu', padding='same', kernel_regularizer=keras.regularizers.L2(0.01)),
       keras.layers.Dropout(0.2),
       keras.layers.Flatten(),
       keras.layers.Dense(64, activation='relu'),
       keras.layers.Dense(5, activation='softmax')
    ])
    train_and_test(conv_model, window, 'Conv')
    print(conv_model.summary())

    print('LSTM')
    lstm = keras.Sequential([
       keras.layers.LSTM(64,return_sequences=False),
       keras.layers.Dense(5, activation='softmax')
    ])
    train_and_test(lstm, window, 'LSTM', retrain=True)
    pprint.pprint(PERFORMANCE, VAL_PERFORMANCE)

def test(model, window, name):
    VAL_PERFORMANCE[name] = model.evaluate(window.val, return_dict=True)
    PERFORMANCE[name] = model.evaluate(window.test, verbose=0, return_dict=True)

def train_and_test(model, window, model_name, patience=5 ,retrain = RETRAIN, epochs=30):
  if model_name not in os.listdir('Training/Models') or retrain:
    HISTORY[model_name] = compile_and_fit(model, window, patience, epochs=epochs)
    model.save(f'Training/Models/{model_name}')
  else:
     model = keras.models.load_model(f'Training/Models/{model_name}')
  test(model,window,model_name)

def compile_and_fit(model, window, patience, epochs):
  early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                  patience=patience,
                                                  mode='min',
                                                  min_delta=0.001,
                                                  verbose=1)

  model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
                loss='categorical_crossentropy',
                metrics=[keras.metrics.CategoricalAccuracy()])

  history = model.fit(window.train, epochs=epochs,
                      validation_data=window.val,
                      callbacks=[early_stopping])
  return history

if __name__ == '__main__':
   main()