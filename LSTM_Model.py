# Advanced LSTM model specialized on prediciting stock prices

#imports
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from single_step_models import ResidualWrapper
from utils import WindowGenerator
from utils import concat_data, compile_and_fit, plot
from multi_step_models import HISTORY, PERFORMANCE, RETRAIN, VAL_PERFORMANCE, LastStepBaseline
import os

RETRAIN = True
VAL_PERFORMANCE = {}
PERFORMANCE = {}
HISTORY={}

def main():
    pass

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


if __name__ == '__main__':
    main()