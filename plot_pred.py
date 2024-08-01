 # Script to predict a possilbe movement using the single_step/multi_step models 150 days into the future
from utils import getdata
from datetime import datetime, timedelta
import pandas as pd
from utils import standardize
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def main(len = 1000):
    end = datetime.now()
    start = end - timedelta(hours=len*5)
    end = end.strftime('%Y-%m-%dT00')
    start = start.strftime('%Y-%m-%dT00')
    symbol = input('Symbol to predict: ')
    data = getdata(start, end, data_name=symbol).tail(len)
    date_time = pd.to_datetime(data.pop('t'), format='%Y-%m-%dT%H:%M:%SZ')
    mean,mean0 = data.mean(), data['c'].mean()
    std,std0 = data.std(), data['c'].std()
    standard_data = standardize(data,mean,std)

    tensor = tf.expand_dims(tf.convert_to_tensor(data.values), 0)
    
    model = tf.keras.models.load_model('Training/Models/multi/multi_dense')
    prediction = model.predict(tensor)
    print(prediction.shape)
    #plot
    # Inverse the normalization
    original_scale_pred = prediction.flatten()#((prediction * std0) + mean0).flatten()
    original_scale_input = tf.convert_to_tensor(data['c'].values)#tf.convert_to_tensor(((data['c']*std0)+mean0).values)
    full_pred = tf.concat([original_scale_input, original_scale_pred],0)
    # Convert the tensor to a 1D array for plotting (if necessary)
    pred_flattened = full_pred

    # Plotting the values
    plt.plot(pred_flattened)
    plt.title(f"Plot of {symbol}'s predicted stock")
    plt.xlabel("future days")
    plt.ylabel("Value")
    plt.show()

if __name__ == '__main__':
    main()
'''
# Existing data
np.random.seed(1)  # For reproducibility
x = np.linspace(0, 10, 100)
y = np.random.random(100)+10

# Forecast triangles
forecast_length = 100
x_forecast_start = x[-1]
x_forecast_end = x_forecast_start + 10
y_forecast_center = y[-1]

# Define forecast windows with different percentage deviations and opacities
windows = [
    {'deviation': 0.02, 'alpha': 0.2},
    {'deviation': 0.015, 'alpha': 0.4},
    {'deviation': 0.01, 'alpha': 0.6},
    {'deviation': 0.005, 'alpha': 0.8}
]

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(x, y, label='Existing Data')

for window in windows:
    deviation = window['deviation']
    y_forecast_top = y_forecast_center * (1 + deviation)
    y_forecast_bottom = y_forecast_center * (1 - deviation)
    plt.fill([x_forecast_start, x_forecast_end, x_forecast_end], 
             [y_forecast_center, y_forecast_top, y_forecast_bottom], 
             color='gray', alpha=window['alpha'], label=f'Forecast ±{deviation*100}%')

plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Data with Percentage Deviation Forecast Triangles')
plt.legend()
plt.show()
'''
