 # Script to predict a possilbe movement using the single_step/multi_step models 150 days into the future
from utils import getdata
from datetime import datetime, timedelta
import pandas as pd
from utils import standardize
import tensorflow as tf
import matplotlib.pyplot as plt
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