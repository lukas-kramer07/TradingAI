## A number of multi step models, prediciting one day into the future based on the last day of data / the last days of data
## They predict the closing_value of the day

#imports
from cProfile import label
from operator import mul
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from utils import WindowGenerator
from utils import concat_data, compile_and_fit, plot
import os

RETRAIN = True
MAX_EPOCHS = 60
VAL_PERFORMANCE = {}
PERFORMANCE = {}
HISTORY = {}
INIT = tf.initializers.zeros()
OUT_STEPS = 50

# Models
class LastStepBaseline(tf.keras.Model):
   def __init__(self, label_index=None):
      super().__init__()
      self.label_index = label_index
   def call(self,inputs):
      if self.label_index:
         res = inputs[:, -1:, self.label_index]
         return tf.tile(res[:, :, tf.newaxis], [1, OUT_STEPS, 1])
      return tf.tile(inputs[:, -1:, :], [1, OUT_STEPS, 1])

class RepeatBaseline(tf.keras.Model):
   def __init__(self, label_index=None):
      super().__init__()
      self.label_index = label_index
   def call(self,inputs):
      
      delta = inputs[:, -1, :] - inputs[:, 0,:]
      inputs+=delta[:, tf.newaxis, :]
      if self.label_index:
         return inputs[:,:,self.label_index][:,:,tf.newaxis]
      return inputs

"""class DeltaBaseline(tf.keras.Model):
   def call(inputs):
      delta = inputs[:, 0:, :] - inputs[:, -1:, :]""" # TODO


def test(model, window, name):
    VAL_PERFORMANCE[name] = model.evaluate(window.val, return_dict=True)
    PERFORMANCE[name] = model.evaluate(window.test, verbose=0, return_dict=True)

def train_and_test(model, window, model_name, patience=3):
  if model_name not in os.listdir('Training/Models') or RETRAIN:
    HISTORY[model_name] = compile_and_fit(model, window, patience)
    model.save(f'Training/Models/{model_name}')
  else:
     model = tf.keras.models.load_model(f'Training/Models/{model_name}')
  test(model,window,model_name)

def main():
   #get data
   train_df, val_df, test_df, column_indices, num_features = concat_data('data')
   # define windows
   multi_window = WindowGenerator(train_df=train_df, val_df = val_df, test_df=test_df,
                                 input_width=50,
                                 label_width=OUT_STEPS,
                                 shift=OUT_STEPS, label_columns=['close'])

   # train the models (single output)

   # Baseline 1
   """print('baselineLastStep')
   last_baseline = LastStepBaseline(label_index=column_indices['close'])
   train_and_test(last_baseline, multi_window, 'lastBaseline')"""
   
   # Baseline 2
   print('baselineRepeat')
   repeat_baseline = RepeatBaseline(label_index=column_indices['close'])
   train_and_test(repeat_baseline, multi_window, 'repeatBaseline')
   multi_window.plot(repeat_baseline)

if __name__ == '__main__':
   main()
   plt.show()