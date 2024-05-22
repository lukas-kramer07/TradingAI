import pandas as pd
import requests
from configparser import ConfigParser
from datetime import datetime

config = ConfigParser()
config.read('config.cfg')
api_key = config['API']['api_key']
secret_api_key = config['API']['secret_api_key']
url = config['API']['base_url']

data_name = 'MSFT'
"""
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
    main()"""
def getdata(start='2018-01-01T00', end='2023-08-08T00', data_name=data_name):
    url = f"https://data.alpaca.markets/v2/stocks/bars?symbols={data_name}&timeframe=1H&start={start}%3A00%3A00Z&end={end}%3A00%3A00Z&limit=10000&adjustment=raw&feed=sip&sort=asc"

    headers = {
        "accept": "application/json",
        "APCA-API-KEY-ID": "PKKCGDFQEWT19TPVRLRD",
        "APCA-API-SECRET-KEY": "CWBpLu9t485amAEWYt8JSFao2KAR0kLBPvNeaM7Q"
    }

    response = requests.get(url, headers=headers).json()
    df = pd.DataFrame(response['bars'][data_name])
    page_token = response["next_page_token"]
    progress = 0
    while page_token:
        progress +=1
        print(f"progress:{progress}/77")
        url=f"https://data.alpaca.markets/v2/stocks/bars?symbols={data_name}&timeframe=1H&start={start}%3A00%3A00Z&end={end}%3A00%3A00Z&limit=10000&adjustment=raw&feed=sip&page_token={page_token}&sort=asc"
        response = requests.get(url, headers=headers).json()
        df1 = pd.DataFrame(response['bars'][data_name])
        df = pd.concat([df, df1])
        page_token = response["next_page_token"]
    return df
    
if __name__=='__main__':
    df = getdata()
    #Save data in folder
    df.to_pickle(f'data/{data_name}')
