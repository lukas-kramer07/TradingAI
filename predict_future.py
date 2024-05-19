 # Script to predict a possilbe movement using the single_step/multi_step models 150 days into the future
from utils import getdata
from datetime import datetime, timedelta

def main(len = 356):
    end = datetime.now()
    start = end - timedelta(hours=len)
    end = end.strftime('%Y-%m-%dT00')
    start = start.strftime('%Y-%m-%dT00')
    symbol = input('Symbol to predict: ')
    data = getdata(start, end, data_name=symbol)
    print(data)

if __name__ == '__main__':
    main()