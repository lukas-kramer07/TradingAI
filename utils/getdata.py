import pandas as pd
import requests
from configparser import ConfigParser
from datetime import datetime

config = ConfigParser()
config.read('config.cfg')
api_key = config['API']['api_key']
secret_api_key = config['API']['secret_api_key']
url = config['API']['base_url']

data_names = ['MSFT']

def main():
    for data_name in data_names:
        df = getdata(data_name)
        df.to_pickle(f'data/{data_name}')
def getdata(start='2018-01-01T00', end='2024-05-18T00', data_name='AAPL'):
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
    main()
