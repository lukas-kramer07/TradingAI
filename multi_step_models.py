## A number of multi step models, prediciting one day into the future based on the last day of data / the last days of data
## They predict the closing_value of the day

#imports
from cProfile import label
from operator import mul
from pickle import TRUE
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from single_step_models import CONV_WIDTH
from utils import WindowGenerator
from utils import concat_data, compile_and_fit, plot
import os

RETRAIN = False
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

def train_and_test(model, window, model_name, patience=3 ,retrain = RETRAIN):
  if model_name not in os.listdir('Training/Models') or retrain:
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
                                 input_width=OUT_STEPS,
                                 label_width=OUT_STEPS,
                                 shift=OUT_STEPS, label_columns=['close'])

   # train the models (single output)
   # Baseline 1
   print('baselineLastStep')
   last_baseline = LastStepBaseline(label_index=column_indices['close'])
   train_and_test(last_baseline, multi_window, 'lastBaseline')
   
   # Baseline 2
   print('baselineRepeat')
   repeat_baseline = RepeatBaseline(label_index=column_indices['close'])
   train_and_test(repeat_baseline, multi_window, 'repeatBaseline')

   # Multilinear
   print('multilinear')
   multi_linear_model = tf.keras.Sequential([
      # Take the last time-step.
      # Shape [batch, time, features] => [batch, 1, features]
      tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
      # Shape => [batch, 1, out_steps*features]
      tf.keras.layers.Dense(OUT_STEPS,),#kernel_initializer=tf.initializers.zeros()),
      # Shape => [batch, out_steps, features]
      tf.keras.layers.Reshape([OUT_STEPS, 1])
   ])
   train_and_test(multi_linear_model, multi_window, 'multi_linear') 

   # MultiDense
   print('multi dense')
   multi_dense_model = tf.keras.Sequential([
      tf.keras.layers.Lambda(lambda x : x[:, -1:, :]),
      tf.keras.layers.Dense(1024, activation = 'relu'),
      tf.keras.layers.Dense(OUT_STEPS),#,kernel_initializer=tf.initializers.zeros()),
      tf.keras.layers.Reshape([OUT_STEPS,1])
   ])
   print(multi_dense_model(multi_window.example[0]).shape)
   train_and_test(multi_dense_model, multi_window, 'multi_dense', retrain=True, patience=10)
   multi_window.plot(multi_dense_model)
   plt.show()

   # Conv Model
   print('conv_model')
   CONV_WIDTH = OUT_STEPS
   multi_conv_model = tf.keras.Sequential([
      # Shape [batch, time, features] => [batch, CONV_WIDTH, features]
      tf.keras.layers.Lambda(lambda x: x[:, -CONV_WIDTH:, :]),
      # Shape => [batch, 1, conv_units]
      tf.keras.layers.Conv1D(256, activation='relu', kernel_size=(CONV_WIDTH)),
      # Shape => [batch, 1,  out_steps*features]
      tf.keras.layers.Dense(OUT_STEPS*num_features,
                           kernel_initializer=tf.initializers.zeros()),
      # Shape => [batch, out_steps, features]
      tf.keras.layers.Reshape([OUT_STEPS, num_features])
   ])
   

   plot(VAL_PERFORMANCE, PERFORMANCE, 'multi_step_performances')


if __name__ == '__main__':
   main()
   plt.show()