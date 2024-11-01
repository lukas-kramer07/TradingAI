# Advanced LSTM model specialized on prediciting stock prices

#imports
import tensorflow as tf
from utils import WindowGenerator
from utils import concat_data
import os 
import keras
from keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau
import datetime

RETRAIN = False
VAL_PERFORMANCE = {}
PERFORMANCE = {}
HISTORY = {}
IN_STEPS=1000

class AttentionLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1]),
                                 initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight(shape=(input_shape[-1],), initializer="zeros", trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        u = keras.backend.tanh(keras.backend.dot(x, self.W) + self.b)
        a = keras.backend.exp(u) / keras.backend.sum(keras.backend.exp(u), axis=1, keepdims=True)
        return keras.backend.sum(a * x, axis=1)
    

class Baseline(keras.Model):
  def __init__(self, output_arr):
    super().__init__()
    self.output_arr = output_arr
  def call(self, inputs):
    batch_size = tf.shape(inputs)[0]
    return tf.tile(tf.convert_to_tensor([self.output_arr], dtype=tf.float32), [batch_size,1])

#TODO: create dynamic Baseline
class DynamicBaseline(keras.Model):
  def __init__(self):
    super().__init__()
  def call(self,inputs):
    c_values_end = inputs[: ,-1, 0]
    c_values_in = inputs[:,1000-150, 0]
    #tf.print('in')
    #tf.print(c_values_in)
    #tf.print('end')
    #tf.print(c_values_end)
    res = tf.divide(c_values_end, c_values_in)
    # use tf.where to mask on label values
    #tf.print(res)
    # conditions
    strong_buy = res>1.05
    buy = tf.logical_and(res >= 1.015, res <= 1.05)
    hold = tf.logical_and(res > 0.985, res < 1.015)
    sell = tf.logical_and(res >= 0.95, res <= 0.985)
    strong_sell = 0.95>res
    res = tf.where(strong_buy, float(0), res) 
    res = tf.where(buy, float(1), res)
    res = tf.where(hold, float(2), res)
    res = tf.where(sell, float(3), res)
    res = tf.where(strong_sell, float(4), res)
    
    # one_hot encode
    res=tf.cast(res, dtype=tf.int32)
    res = tf.one_hot(res, depth = 5, dtype=tf.float32)
    return res

from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras import backend as K

@register_keras_serializable()
def custom_accuracy(y_true, y_pred):
    # Convert predictions to one-hot format (argmax)
    pred_class = K.argmax(y_pred, axis=-1)
    true_class = K.argmax(y_true, axis=-1)
    
    # Exact match (full accuracy)
    exact_match = K.cast(K.equal(pred_class, true_class), dtype=tf.float32)
    
    # 1-off match (half accuracy)
    one_off_match = K.cast(K.equal(K.abs(pred_class - true_class), 1), dtype=tf.float32) * 0.5
    
    # Combine exact and 1-off matches
    accuracy = exact_match + one_off_match
    
    # Return mean accuracy over all samples
    return K.mean(accuracy)

def main():
    #get data
    train_df, val_df, test_df, column_indices, num_features = concat_data('data', standard=False)
    # define windows
    window = WindowGenerator(train_df=train_df, val_df = val_df, test_df=test_df,
                                    input_width=IN_STEPS,
                                    shift=150, label_columns=['c']) # shift in hours (32.5 Trading hours in a week, so 150h~5weeks)
    # Training
    # Baseline Models
    print(window.example)

    '''print('dynamic baseline')
    dynamic_baseline = DynamicBaseline()
    print(dynamic_baseline.predict(window.example[0]))
    train_and_test(dynamic_baseline, window, 'dynamic_baseline', epochs=1)

    baselines = {
        'Baseline_hold': [0, 0, 1, 0, 0],
        'Baseline_buy': [0, 1, 0, 0, 0],
        'Baseline_sell': [0, 0, 0, 1, 0],
        'Baseline_strong_buy': [1, 0, 0, 0, 0],
        'Baseline_strong_sell': [0, 0, 0, 0, 1],
        'Baseline_equal': [0.2,0.2,0.2,0.2,0.2],
        'Baseline_normal': [0.1,0.2,0.4,0.2,0.1]
    }
    
    for name, output_arr in baselines.items():
        print(name)
        baseline_model = Baseline(output_arr)
        train_and_test(baseline_model, window, name, epochs=1)'''

    
    print('linear model')
    linear_model = keras.Sequential([
        keras.layers.Flatten(),
        keras.layers.Dense(units=5, activation='softmax')
      ])
    train_and_test(linear_model, window, 'Linear')

    print('deep model')
    deep_model = keras.Sequential([
      keras.layers.BatchNormalization(),
      keras.layers.Flatten(),
      keras.layers.Dense(128, activation='relu'),
      #keras.layers.Dropout(0.3),
      keras.layers.Dense(64, activation='relu'),
      keras.layers.Dense(5, activation='softmax')
    ])
    train_and_test(deep_model, window, 'Deep')

    print('conv_model')
    conv_model = keras.Sequential([
      keras.layers.BatchNormalization(),
      keras.layers.GaussianNoise(stddev=0.2),
      keras.layers.Conv1D(32, 3, activation='relu', padding='same', kernel_regularizer=keras.regularizers.L2(0.01)),
      #keras.layers.BatchNormalization(),
      keras.layers.Dropout(0.3),
      #keras.layers.Flatten(),
      keras.layers.Flatten(),
      keras.layers.Dense(64, activation='relu'),
      keras.layers.Dense(5, activation='softmax')
    ])
    train_and_test(conv_model, window, 'Conv', retrain=True)

    print('improved_conv_model')
    improved_conv_model = keras.Sequential([
        keras.layers.GaussianNoise(stddev=0.2),
        keras.layers.Conv1D(32, 5, activation='relu', padding='same'),# kernel_regularizer=keras.regularizers.L2(0.01)),
        keras.layers.Dropout(0.3),
        keras.layers.Conv1D(64, 3, activation='relu', padding='same'),# kernel_regularizer=keras.regularizers.L2(0.01))
        keras.layers.Dropout(0.3),
        keras.layers.Conv1D(128, 3, activation='relu', padding='same'),# kernel_regularizer=keras.regularizers.L2(0.01))
        keras.layers.Dropout(0.3),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        #keras.layers.Dropout(0.3),
        keras.layers.Dense(5, activation='softmax')
    ])

    train_and_test(improved_conv_model, window, 'Improved_Conv', retrain=True)

    print('LSTM')
    lstm = keras.Sequential([
      keras.layers.BatchNormalization(),
      #keras.layers.GaussianNoise(stddev=0.2),
    
      keras.layers.LSTM(64,return_sequences=False),
      keras.layers.Dense(64, activation='relu'),
      keras.layers.Dense(5, activation='softmax', kernel_initializer="zeros")
    ])
    train_and_test(lstm, window, 'LSTM')
    
    print('Improved LSTM with Bidirectional and Dropout')  
    improved_lstm_1 = keras.Sequential([
        keras.layers.BatchNormalization(),
        keras.layers.GaussianNoise(stddev=0.2),

        keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True, kernel_regularizer=keras.regularizers.l2(0.01))),
        keras.layers.Dropout(0.3),
        keras.layers.Bidirectional(keras.layers.LSTM(32, return_sequences=False)),
        
        keras.layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(5, activation='softmax')
    ])
    train_and_test(improved_lstm_1, window, 'Improved_LSTM_1', retrain=True, epochs=3)

    print('Improved LSTM with More Units and Layer Normalization')  
    improved_lstm_2 = keras.Sequential([
        keras.layers.BatchNormalization(),
        keras.layers.GaussianNoise(stddev=0.2),

        keras.layers.LSTM(128, return_sequences=True, kernel_regularizer=keras.regularizers.l2(0.01)),
        keras.layers.LayerNormalization(),
        keras.layers.LSTM(64, return_sequences=False),
        
        keras.layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(5, activation='softmax')
    ])
    train_and_test(improved_lstm_2, window, 'Improved_LSTM_2', retrain=True, epochs=3)

    print("improved LSTM with attention layer")
    improved_lstm_3 = keras.Sequential([
        keras.layers.BatchNormalization(),
        keras.layers.GaussianNoise(stddev=0.2),

        keras.layers.LSTM(64, return_sequences=True, kernel_regularizer=keras.regularizers.l2(0.01)),
        AttentionLayer(),

        keras.layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(5, activation='softmax')
    ])
    train_and_test(improved_lstm_3, window, 'Improved_LSTM_3', retrain=True, epochs=3)
    for model, *performance in VAL_PERFORMANCE.items():
      print(f'{model}: {performance}\n')


def test(model, window, name):
    VAL_PERFORMANCE[name] = model.evaluate(window.val, return_dict=True)
    PERFORMANCE[name] = model.evaluate(window.test, verbose=0, return_dict=True)

def train_and_test(model, window, model_name, patience=5 ,retrain = RETRAIN, epochs=30):
  if f'{model_name}.keras' not in os.listdir('Training/Models') or retrain:
    HISTORY[model_name] = compile_and_fit(model, window, patience, epochs=epochs,model_name=model_name)
    model.save(f'Training/Models/{model_name}.keras')
  else:
     model = keras.models.load_model(f'Training/Models/{model_name}.keras')
  test(model,window,model_name)

def compile_and_fit(model, window, patience, epochs, model_name):

  #CALLBACKS
  lr_callback= ReduceLROnPlateau(patience=patience//2)
  early_stopping = EarlyStopping(monitor='val_loss',
                                                  patience=patience,
                                                  mode='min',
                                                  min_delta=0.001,
                                                  verbose=1,
                                                  restore_best_weights=True)
  
  log_dir = f'Training/logs/{model_name}/' +datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1, #write_images=True, write_graph=True, )
                            profile_batch='500,800')
  
  #COMPILE
  model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
                loss='categorical_crossentropy',
                metrics=[keras.metrics.CategoricalAccuracy(), custom_accuracy, keras.metrics.Precision(), keras.metrics.Recall()])

  #FIT
  history = model.fit(window.train, epochs=epochs,
                      validation_data=window.val,
                      callbacks=[early_stopping, tensorboard])
  return history

def sort_plugin_files(dir = "Training/logs"):
  '''
  redirect plugin subdir to train subdir; fixes a tensorboard bug
  '''
  for models in os.listdir(dir):
    f = os.path.join(dir, models)
    for logs in os.listdir(f):
      log = os.path.join(f, logs)
      if 'plugins' in os.listdir(log):
        os.rename(log+'/plugins', log+'/train/plugins')

if __name__ == '__main__':
  main()
  sort_plugin_files()