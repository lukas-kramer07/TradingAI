 # Script to predict a possilbe movement using the single_step/multi_step models 150 days into the future
from utils import getdata
from datetime import datetime, timedelta
import pandas as pd
from utils import standardize
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def main(len = 1000):

    #get data specified by user
    end = datetime.now()
    start = end - timedelta(hours=len*5)
    end = end.strftime('%Y-%m-%dT00')
    start = start.strftime('%Y-%m-%dT00')
    symbol = input('Symbol to predict: ')
    try:
        data = getdata(start, end, data_name=symbol).tail(len)
    except:
        raise Exception('Wrong Data Name or too little Data available')


    date_time = pd.to_datetime(data.pop('t'), format='%Y-%m-%dT%H:%M:%SZ')
    mean = data.mean()
    std = data.std()
    standard_data = standardize(data,mean,std)

    tensor = tf.expand_dims(tf.convert_to_tensor(data.values), 0)
    
    model = tf.keras.models.load_model('Training/Models/LSTM')
    prediction = model.predict(tensor)
    print(prediction.shape, prediction)

    data = np.array(data.pop('c'))
    plot(data, prediction[0])


def plot(data, prediction):

    # Forecast triangles
    forecast_length = 150
    x_forecast_start = 1000
    x_forecast_end = x_forecast_start + forecast_length
    y_forecast_center = data[-1]

    # Define forecast windows with different percentage deviations and opacities
    windows = [
        {'deviation': 0.02, 'alpha': prediction[0]},
        {'deviation': 0.015, 'alpha': prediction[1]},
        {'deviation': 0.01, 'alpha': prediction[2]},
        {'deviation': 0.005, 'alpha': prediction[3]}
    ]

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(data, label='Existing Data')

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
    
if __name__ == '__main__':
    main()