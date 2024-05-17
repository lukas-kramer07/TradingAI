## A number of multi step models, prediciting one day into the future based on the last day of data / the last days of data
## They predict the closing_value of the day

#imports
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from single_step_models import ResidualWrapper
from utils import WindowGenerator
from utils import concat_data, compile_and_fit, plot
import os

RETRAIN = True
VAL_PERFORMANCE = {}
PERFORMANCE = {}
HISTORY = {}
INIT = tf.initializers.zeros()
OUT_STEPS = 200

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
      delta = inputs[:, 0:, :] - inputs[:, -1:, :]""" # TODO: create Baseline

class FeedBack(tf.keras.Model):
   def __init__(self, units, out_steps):
      super().__init__()
      self.out_steps = out_steps
      self.units = units
      self.lstm_cell = tf.keras.layers.LSTMCell(units)
      # Also wrap the LSTMCell in an RNN to simplify the `warmup` method.
      self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state=True)
      self.dense = tf.keras.layers.Dense(11)
   def warmup(self, inputs):
      # inputs.shape => (batch, time, features)
      # x.shape => (batch, lstm_units)
      x, *state = self.lstm_rnn(inputs)

      # predictions.shape => (batch, features)
      prediction = self.dense(x)
      return prediction, state
   def call(self, inputs, training=None):
      # Use a TensorArray to capture dynamically unrolled outputs.
      predictions = []
      # Initialize the LSTM state.
      prediction, state = self.warmup(inputs)

      # Insert the first prediction.
      predictions.append(prediction)

      # Run the rest of the prediction steps.
      for n in range(1, self.out_steps):
         # Use the last prediction as input.
         x = prediction
         # Execute one lstm step.
         x, state = self.lstm_cell(x, states=state,
                                    training=training)
         # Convert the lstm output to a prediction.
         prediction = self.dense(x)
         # Add the prediction to the output.
         predictions.append(prediction)

      # predictions.shape => (time, batch, features)
      predictions = tf.stack(predictions)
      # predictions.shape => (batch, time, features)
      predictions = tf.transpose(predictions, [1, 0, 2])
      return predictions


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
   train_df, val_df, test_df, column_indices, num_features = concat_data('data', standard=True)
   # define windows
   multi_window = WindowGenerator(train_df=train_df, val_df = val_df, test_df=test_df,
                                 input_width=OUT_STEPS,
                                 label_width=OUT_STEPS,
                                 shift=OUT_STEPS, label_columns=['c'])

   # train the models (single output)
   # Baseline 1
   print('baselineLastStep')
   last_baseline = LastStepBaseline(label_index=column_indices['c'])
   train_and_test(last_baseline, multi_window, 'multi/lastBaseline')
   
   # Baseline 2
   print('baselineRepeat')
   repeat_baseline = RepeatBaseline(label_index=column_indices['c'])
   train_and_test(repeat_baseline, multi_window, 'multi/repeatBaseline')

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
   train_and_test(multi_linear_model, multi_window, 'multi/multi_linear') 

   # MultiDense
   print('multi dense')
   multi_dense_model = tf.keras.Sequential([
      tf.keras.layers.Lambda(lambda x : x[:, -1:, :]),
      tf.keras.layers.Dense(1024, activation = 'relu'),
      tf.keras.layers.Dense(OUT_STEPS),#,kernel_initializer=tf.initializers.zeros()),
      tf.keras.layers.Reshape([OUT_STEPS,1])
   ])
   print(multi_dense_model(multi_window.example[0]).shape)
   train_and_test(multi_dense_model, multi_window, 'multi/multi_dense')
   

   # Conv Model
   print('conv_model')
   CONV_WIDTH = OUT_STEPS//2
   multi_conv_model = tf.keras.Sequential([
      # Shape [batch, time, features] => [batch, CONV_WIDTH, features]
      tf.keras.layers.Lambda(lambda x: x[:, -CONV_WIDTH:, :]),
      # Shape => [batch, 1, conv_units]
      tf.keras.layers.Conv1D(256, activation='relu', kernel_size=(CONV_WIDTH)),
      # Shape => [batch, 1,  out_steps*features]
      tf.keras.layers.Dense(OUT_STEPS),#kernel_initializer=tf.initializers.zeros()),
      # Shape => [batch, out_steps, features]
      tf.keras.layers.Reshape([OUT_STEPS, 1])
   ])
   train_and_test(multi_conv_model, multi_window, 'multi/multi_conv', patience=10)

   # lstm Model
   print('lstm Model')
   multi_lstm_model = tf.keras.Sequential([
      # Shape [batch, time, features] => [batch, lstm_units].
      # Adding more `lstm_units` just overfits more quickly.
      tf.keras.layers.LSTM(32, return_sequences=False),
      # Shape => [batch, out_steps*features].
      tf.keras.layers.Dense(OUT_STEPS),
      # Shape => [batch, out_steps, features].
      tf.keras.layers.Reshape([OUT_STEPS, 1])
   ])
   train_and_test(multi_lstm_model, multi_window, 'multi/multi_lstm')

   # res Net lstm
   print('residual_lstm')
   residual_lstm_multi = ResidualWrapper(
        tf.keras.Sequential([
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.Dense(
            OUT_STEPS,
            kernel_initializer=tf.initializers.zeros())
    ]), label_index=column_indices['c'])
   train_and_test(residual_lstm_multi, multi_window, 'multi/residual_lstm_multi')
   
   
   # autoregreassive model
   feedback_model = FeedBack(units=32, out_steps=OUT_STEPS)
   train_and_test(feedback_model, multi_window, 'autoregressive', retrain=True)
   multi_window.plot(feedback_model, max_subplots=15)


   plt.show()
   plot(VAL_PERFORMANCE, PERFORMANCE, 'all_standard_multi_step_performances')


if __name__ == '__main__':
   main()
   plt.show()