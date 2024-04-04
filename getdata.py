import pandas as pd
import requests
from configparser import ConfigParser
from datetime import datetime

config = ConfigParser()
config.read('config.cfg')
api_key = config['API']['api_key']

# Define the URL of the API endpoint you want to request
url = 'https://financialmodelingprep.com/api/v3/historical-price-full/AAPL'

def getdata():
    # Make a GET request to the API endpoint
    response = requests.get(url+'?apikey='+api_key)
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
    
    df = getdata()
    # Display the DataFrame
    print(df)
    
    # Save data in data folder
    df.to_pickle('data/apple')

if __name__ == '__main__':
    main()