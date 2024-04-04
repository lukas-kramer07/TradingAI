import pandas as pd
import requests
from configparser import ConfigParser
from datetime import datetime

config = ConfigParser()
config.read('config.cfg')
api_key = config['API']['api_key']
# Define the URL of the API endpoint you want to request
url = 'https://financialmodelingprep.com/api/v3/historical-price-full/AAPL'

NUM_DATA = 3 # number of 5-years data stacked in a df 
DATA_NAME = 'apple'

def getdata(date_end):

    date_start = date_end.replace(year=date_end.year-5, day=date_end.day+1) # data range is 5 years-1 day

    #format dates
    date_end = date_end.strftime('%Y-%m-%d')
    date_start = date_start.strftime('%Y-%m-%d')
    # Make a GET request to the API endpoint
    response = requests.get(url+'?from='+date_start+'&to='+date_end+'&apikey='+api_key)
    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Extract the JSON data from the response
        data = response.json()
        
    else:
        # If the request was not successful, print an error message
        print(f"Error: {response.status_code}")
        
    # Convert JSON data to pandas DataFrame
    df = pd.json_normalize(data['historical'])
    return df

def main():
    dataframes = []
    for n in range(NUM_DATA):
        curr_year = datetime.now().year
        df = getdata(datetime.now().replace(year=curr_year-5*n))
        # Display the DataFrame
        dataframes.append(df)
        
    # Save data in data folder
    #df.to_pickle('data/apple')

if __name__ == '__main__':
    main()