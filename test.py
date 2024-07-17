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

RETRAIN = True 
VAL_PERFORMANCE = {}
PERFORMANCE = {}
HISTORY = {}
IN_STEPS=1000

class Baseline(keras.Model):
  def __init__(self, output_arr):
    super().__init__()
    self.output_arr = output_arr
  def call(self, inputs):
    return tf.convert_to_tensor([self.output_arr], dtype=tf.float32)


def main():
    #get data
    train_df, val_df, test_df, column_indices, num_features = concat_data('data', standard=False)
    # define windows
    window = WindowGenerator(train_df=train_df, val_df = val_df, test_df=test_df,
                                    input_width=IN_STEPS,
                                    shift=100, label_columns=['c'])
    print(window.example)

    # Training
    # Baseline Models
    baselines = {
        'Baseline_hold': [0, 0, 1, 0, 0],
        'Baseline_buy': [0, 1, 0, 0, 0],
        'Baseline_sell': [0, 0, 0, 1, 0],
        'Baseline_strong_buy': [1, 0, 0, 0, 0],
        'Baseline_strong_sell': [0, 0, 0, 0, 1]
    }
    
    for name, output_arr in baselines.items():
        print(name)
        baseline_model = Baseline(output_arr)
        train_and_test(baseline_model, window, name)

    print('linear model')
    linear_model = keras.Sequential([
        keras.layers.Flatten(),
        keras.layers.Dense(units=64, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(units=5, activation='softmax')
      ])
    train_and_test(linear_model, window, 'Linear')
    print(linear_model(window.example))
    m = keras.metrics.Accuracy()
    print(baseline_model(window.example), window.example[1])
def test(model, window, name):
    VAL_PERFORMANCE[name] = model.evaluate(window.val, return_dict=True)
    PERFORMANCE[name] = model.evaluate(window.test, verbose=0, return_dict=True)

def train_and_test(model, window, model_name, patience=3 ,retrain = RETRAIN):
  if True:#model_name not in os.listdir('Training/Models/multi') or retrain:
    HISTORY[model_name] = compile_and_fit(model, window, patience)
    model.save(f'Training/Models/multi/{model_name}')
  else:
     model = keras.models.load_model(f'Training/Models/multi/{model_name}')
  test(model,window,model_name)

def compile_and_fit(model, window, patience, epochs=30):
  early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                  patience=patience,
                                                  mode='min',
                                                  min_delta=0.001,
                                                  verbose=1)

  model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
                loss='categorical_crossentropy',
                metrics=[keras.metrics.Accuracy()])

  history = model.fit(window.train, epochs=epochs,
                      validation_data=window.val,
                      callbacks=[early_stopping])
  return history

if __name__ == '__main__':
   main()