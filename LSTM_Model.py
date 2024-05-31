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

RETRAIN = True
VAL_PERFORMANCE = {}
PERFORMANCE = {}
HISTORY = {}
OUT_STEPS = 1000
IN_STEPS=1000
def main():
    #get data
    train_df, val_df, test_df, column_indices, num_features = concat_data('data', standard=False)
    # define windows
    multi_window = WindowGenerator(train_df=train_df, val_df = val_df, test_df=test_df,
                                    input_width=IN_STEPS,
                                    label_width=OUT_STEPS,
                                    shift=1, label_columns=['c'])
    last_baseline = LastStepBaseline(label_index=column_indices['c'])
    train_and_test(last_baseline, multi_window, 'lastBaseline')

    LSTM_model = tf.keras.Sequential([
      tf.keras.layers.LSTM(256, return_sequences=True, input_shape=(1000, 1)),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.LSTM(128, return_sequences=True),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.LSTM(64),
      tf.keras.layers.Dense(OUT_STEPS * 1),
      tf.keras.layers.Reshape([OUT_STEPS, 1])
    ])
    train_and_test(LSTM_model, multi_window, 'improved_LSTM')

    plot(VAL_PERFORMANCE, PERFORMANCE)

def test(model, window, name):
    VAL_PERFORMANCE[name] = model.evaluate(window.val, return_dict=True)
    PERFORMANCE[name] = model.evaluate(window.test, verbose=0, return_dict=True)

def train_and_test(model, window, model_name, patience=3 ,retrain = RETRAIN):
  if model_name not in os.listdir('Training/Models/multi') or retrain:
    HISTORY[model_name] = compile_and_fit(model, window, patience)
    model.save(f'Training/Models/multi/{model_name}')
  else:
     model = tf.keras.models.load_model(f'Training/Models/multi/{model_name}')
  test(model,window,model_name)

def compile_and_fit(model, window, patience, epochs=200):
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')

  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
                loss='mean_squared_error',
                metrics=[tf.keras.metrics.MeanAbsoluteError()])

  history = model.fit(window.train, epochs=epochs,
                      validation_data=window.val,
                      callbacks=[early_stopping])
  return history

if __name__ == '__main__':
    main()