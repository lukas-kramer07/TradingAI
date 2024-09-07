 # Script to predict a possilbe movement using the single_step/multi_step models 150 trading hours into the future
from utils import getdata
from datetime import datetime, timedelta
import pandas as pd
from utils import standardize
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
LEN = 1000
def main(len = LEN):
    
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
    print(data)
    
    date_time = pd.to_datetime(data.pop('t'), format='%Y-%m-%dT%H:%M:%SZ')
    mean = data.mean()
    std = data.std()
    standard_data = standardize(data,mean,std)

    tensor = tf.expand_dims(tf.convert_to_tensor(data.values), 0)
    data = np.array(data.pop('c'))
    

    for modelname in os.listdir('Training/Models'):
        dir = os.path.join('Training/Models', modelname)
        model = tf.keras.models.load_model(dir)
        prediction = model.predict(tensor)
        print(modelname)
        print(prediction.shape, prediction)

        
        plot(data, prediction[0], symbol, name=modelname+' predicts: ')


def plot(data, prediction, symbol, name='Prediction- '):

    # Forecast triangles
    forecast_length = 150
    x_forecast_start = 0
    x_forecast_end = x_forecast_start + forecast_length
    y_forecast_center = data[-1]
    x_labels = np.arange(len(data)) - len(data)+1
    print(x_labels)
    # Define forecast windows with different percentage deviations and opacities
    windows = [
{'lower': 0.05, 'upper': 0.1, 'alpha': prediction[0], 'label': f'strong buy -{prediction[0]*100:.1f}%', 'color': '#006400', 'dotted': True},  # darkgreen
    {'lower': 0.015, 'upper': 0.05, 'alpha': prediction[1], 'label': f'buy -{prediction[1]*100:.1f}%', 'color': '#90EE90'},  # lightgreen
    {'lower': -0.015, 'upper': 0.015, 'alpha': prediction[2], 'label': f'hold -{prediction[2]*100:.1f}%', 'color': '#BDB76B'},  # darkyellow
    {'lower': -0.05, 'upper': -0.015, 'alpha': prediction[3], 'label': f'sell -{prediction[3]*100:.1f}%', 'color': '#FFA07A'},  # lightred
    {'lower': -0.1, 'upper': -0.05, 'alpha': prediction[4], 'label': f'strong sell -{prediction[4]*100:.1f}%', 'color': '#8B0000', 'dotted': True}  # darkred
    ]

    # Plotting
    fig = plt.figure(figsize=(20, 15))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(223)
    ax1.plot(x_labels, data, label='Existing Data')

    for window in windows:
        linestyle = ':' if 'dotted' in window else '-'
        y_forecast_top = y_forecast_center * (1 + window['upper'])
        y_forecast_bottom = y_forecast_center * (1 + window['lower'])
        ax1.fill([x_forecast_start, x_forecast_end, x_forecast_end], 
                [y_forecast_center, y_forecast_top, y_forecast_bottom], 
                color=window['color'], alpha=window['alpha'], linestyle=linestyle, linewidth=2.5, label=window['label'])

    ax1.set_xlabel('Trading hour Data points')
    ax1.set_ylabel('Close value $')
    fig.set_title(name+ symbol)
    ax1.legend()

    ax2.pie(prediction, labels=[window['label'] for winodw in windows], colors=[window['color'] for window in windows])
    plt.show()
    
if __name__ == '__main__':
    main()