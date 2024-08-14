import tensorflow as tf

import numpy as np
import os
import time
import keras
gpus = tf.config.list_physical_devices("GPU")
print(gpus)
if gpus:
    # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        logical_gpus = tf.config.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)
gpus


tensor = tf.convert_to_tensor([4])
print(tf.one_hot(tensor, depth = 5, dtype=tf.int32))

print(keras.metrics.CategoricalAccuracy()([0,0,1], [0.3,0.3,0.4]))