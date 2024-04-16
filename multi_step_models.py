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
MAX_EPOCHS = 60
CONV_WIDTH = 10
VAL_PERFORMANCE = {}
PERFORMANCE = {}
HISTORY = {}
INIT = tf.initializers.zeros()


def test(model, window, name):
    VAL_PERFORMANCE[name] = model.evaluate(window.val, return_dict=True)
    PERFORMANCE[name] = model.evaluate(window.test, verbose=0, return_dict=True)

def train_and_test(model, window, model_name, patience=5):
  if model_name not in os.listdir('Training/Models') or RETRAIN:
    HISTORY[model_name] = compile_and_fit(model, window, patience)
    model.save(f'Training/Models/{model_name}')
  else:
     model = tf.keras.models.load_model(f'Training/Models/{model_name}')
  test(model,window,model_name)