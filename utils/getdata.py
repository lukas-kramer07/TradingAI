import pandas as pd
import requests
from configparser import ConfigParser
from datetime import datetime

config = ConfigParser()
config.read('config.cfg')
api_key = config['API']['api_key']
secret_api_key = config['API']['secret_api_key']
url = config['API']['base_url']

# Technology
technology_stocks = [
    "AAPL",  # Apple
    "MSFT",  # Microsoft
    "GOOGL", # Alphabet
    "AMZN",  # Amazon
    "NVDA"   # NVIDIA
]

# Healthcare
healthcare_stocks = [
    "JNJ",   # Johnson & Johnson
    "PFE",   # Pfizer
    "MRNA"   # Moderna
]

# Finance
finance_stocks = [
    "JPM",   # JPMorgan Chase
    "GS",    # Goldman Sachs
    "BAC"    # Bank of America
]

# Consumer Goods
consumer_goods_stocks = [
    "PG",    # Procter & Gamble
    "KO",    # Coca-Cola
    "WMT"    # Walmart
]

# Energy
energy_stocks = [
    "XOM",   # ExxonMobil
    "CVX"    # Chevron
]

# Utilities
utilities_stocks = [
    "NEE",   # NextEra Energy
    "DUK"    # Duke Energy
]

# Combined list of all stocks
all_stocks = (
    technology_stocks +
    healthcare_stocks +
    finance_stocks +
    consumer_goods_stocks +
    energy_stocks +
    utilities_stocks
)


def getdata(start='2018-01-01T00', end='2024-05-05T00', data_name='AAPL'):
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

def main():
    for data_name in all_stocks:
        print(data_name)
        df = getdata(data_name=data_name)
        df.to_pickle(f'data/{data_name}') 
if __name__=='__main__':
    main()
