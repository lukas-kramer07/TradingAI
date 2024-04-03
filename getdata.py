import pandas as pd

# JSON data sample
json_data = {
    "symbol": "AAPL",
    "historical": [
        {
            "date": "2023-10-06",
            "open": 173.8,
            "high": 176.61,
            "low": 173.18,
            "close": 176.53,
            "adjClose": 176.53,
            "volume": 21712747,
            "unadjustedVolume": 21712747,
            "change": 2.73,
            "changePercent": 1.57077,
            "vwap": 175.44,
            "label": "October 06, 23",
            "changeOverTime": 0.0157077
        },
        {
            "date": "2023-10-05",
            "open": 173.79,
            "high": 175.45,
            "low": 172.68,
            "close": 174.91,
            "adjClose": 174.91,
            "volume": 48251046,
            "unadjustedVolume": 47369743,
            "change": 1.12,
            "changePercent": 0.64446,
            "vwap": 174.23,
            "label": "October 05, 23",
            "changeOverTime": 0.0064446
        }
    ]
}

# Convert JSON data to pandas DataFrame
df_apple = pd.json_normalize(json_data['historical'])

# Display the DataFrame
print(df_apple)

# Save data in data folder
df_apple.to_pickle('data/apple')