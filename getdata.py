import pandas as pd
import requests
from configparser import ConfigParser

config = ConfigParser()
config.read('config.cfg')

api_key = config['API']['api_key']
print(api_key)
# Define the URL of the API endpoint you want to request
url = 'https://api.example.com/data'


def main():
    # Make a GET request to the API endpoint
    response = requests.get(url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Extract the JSON data from the response
        data = response.json()
        # Process the data as needed
        print(data)
    else:
        # If the request was not successful, print an error message
        print(f"Error: {response.status_code}")

    # Convert JSON data to pandas DataFrame
    df_apple = pd.json_normalize(data['historical'])

    # Display the DataFrame
    print(df_apple)

    # Save data in data folder
    df_apple.to_pickle('data/apple')
