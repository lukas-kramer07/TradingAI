import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def compile_and_fit(model, window, patience, epochs=60):
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')

  model.compile(loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=[tf.keras.metrics.MeanAbsoluteError()])

  history = model.fit(window.train, epochs=epochs,
                      validation_data=window.val,
                      callbacks=[early_stopping])
  return history

def plot(val_performance, performance, plotname = 'NONE'):
    # Plot models' performances
    x = np.arange(len(performance))
    width = 0.3
    metric_name = 'mean_absolute_error'
    val_mae = [v[metric_name] for v in val_performance.values()]
    test_mae = [v[metric_name] for v in performance.values()]

    plt.ylabel('mean_absolute_error [close normalized]')
    plt.bar(x - 0.17, val_mae, width, label='Validation')
    plt.bar(x + 0.17, test_mae, width, label='Test')
    plt.xticks(ticks=x, labels=performance.keys(),
              rotation=45)
    _ = plt.legend()
    plt.savefig(f'Training/plots/{plotname}')
    plt.close()