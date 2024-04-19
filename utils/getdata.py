import pandas as pd
import requests
from configparser import ConfigParser
from datetime import datetime

config = ConfigParser()
config.read('config.cfg')
api_key = config['API']['api_key']
# Define the URL of the API endpoint you want to request
DATA_NAMES = ['AAPL', 'JNJ', 'V', 'KO', 'XOM', 'WMT', 'GOOGL', 'PFE', 'JPM', 'PG', 'AMZN', 'CVX', 'COST', 'FB', 'T']
url = f'https://financialmodelingprep.com/api/v3/historical-price-full/'

NUM_DATA = 2 # number of 5-years data stacked in a df 


def getdata(date_end, data_name):

    date_start = date_end.replace(year=date_end.year-5, day=date_end.day+1) # data range is 5 years-1 day

    #format dates
    date_end = date_end.strftime('%Y-%m-%d')
    date_start = date_start.strftime('%Y-%m-%d')
    # Make a GET request to the API endpoint
    response = requests.get(url+data_name+'?from='+date_start+'&to='+date_end+'&apikey='+api_key)
    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Extract the JSON data from the response
        data = response.json()
        
    else:
        # If the request was not successful, print an error message
        print(f"Error: {response.status_code}")
        
    # Convert JSON data to pandas DataFrame
    df = pd.json_normalize(data['historical']).iloc[::-1]
    return df

def main():
    for data_name in DATA_NAMES:
        dataframes = []
        for n in range(NUM_DATA):
            curr_year = datetime.now().year
            df = getdata(datetime.now().replace(year=curr_year-5*n), data_name)
            # Display the DataFrame
            dataframes.append(df)
        
        #stack dataframes
        dataframes.reverse()
        df = pd.concat(dataframes).reset_index(drop=True)
        # Save data in data folder
        df.to_pickle(f'data/{data_name}')
        print(df)

if __name__ == '__main__':
    main()