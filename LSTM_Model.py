# Advanced LSTM model specialized on prediciting stock prices

#imports
from cgitb import small
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

# Improved LSTM Model with Regularization
def build_model():
    regularizer = tf.keras.regularizers.l2(1e-4)  # L2 regularization factor
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(256, return_sequences=True, input_shape=(1000, 1),
                             kernel_regularizer=regularizer, bias_regularizer=regularizer),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(128, return_sequences=True,
                             kernel_regularizer=regularizer, bias_regularizer=regularizer),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(64, kernel_regularizer=regularizer, bias_regularizer=regularizer),
        tf.keras.layers.Dense(OUT_STEPS * 1, kernel_regularizer=regularizer),
        tf.keras.layers.Reshape([OUT_STEPS, 1])
    ])
    return model
# Smaller LSTM Model with Regularization
def build_small_model():
    regularizer = tf.keras.regularizers.l2(1e-4)  # L2 regularization factor
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, return_sequences=False, input_shape=(1000, 1),
                             kernel_regularizer=regularizer, bias_regularizer=regularizer),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(OUT_STEPS * 1, kernel_regularizer=regularizer),
        tf.keras.layers.Reshape([OUT_STEPS, 1])
    ])
    return model
def main():
    #get data
    train_df, val_df, test_df, column_indices, num_features = concat_data('data', standard=False)
    # define windows
    multi_window = WindowGenerator(train_df=train_df, val_df = val_df, test_df=test_df,
                                    input_width=IN_STEPS,
                                    label_width=OUT_STEPS,
                                    shift=1, label_columns=['c'])
    
    # Check your training data (example: multi_window.train)
    check_for_nan(multi_window.train)
    print('BASELINE')
    last_baseline = LastStepBaseline(label_index=column_indices['c'])
    train_and_test(last_baseline, multi_window, 'lastBaseline')

    print('SMALL_LSTM')
    small_LSTM_model = build_small_model()
    train_and_test(small_LSTM_model, multi_window, 'small_LSTM')


    print('LARGE_LSTM')
    LSTM_model = build_model()
    train_and_test(LSTM_model, multi_window, 'improved_LSTM')

    plot(VAL_PERFORMANCE, PERFORMANCE, metric_name='accuracy')

def test(model, window, name):
    VAL_PERFORMANCE[name] = model.evaluate(window.val, return_dict=True)
    PERFORMANCE[name] = model.evaluate(window.test, verbose=0, return_dict=True)

def train_and_test(model, window, model_name, patience=15 ,retrain = RETRAIN):
  if model_name not in os.listdir('Training/Models/multi') or retrain:
    HISTORY[model_name] = compile_and_fit(model, window, patience)
    model.save(f'Training/Models/multi/{model_name}')
  else:
     model = tf.keras.models.load_model(f'Training/Models/multi/{model_name}')
  test(model,window,model_name)

def compile_and_fit(model, window, patience=15, epochs=20):
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')

  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
                loss='mean_squared_error',
                metrics=[tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.MeanAbsolutePercentageError(), tf.keras.metrics.Accuracy()])

  history = model.fit(window.train, epochs=epochs,
                      validation_data=window.val,
                      callbacks=[early_stopping])
  return history
# Ensure no NaN values in input data
def check_for_nan(dataset):
    for batch in dataset:
        inputs, targets = batch
        if tf.reduce_any(tf.math.is_nan(inputs)) or tf.reduce_any(tf.math.is_nan(targets)):
            raise ValueError("Input data contains NaN values")
if __name__ == '__main__':
    main()