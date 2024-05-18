 # Script to predict a possilbe movement using the single_step/multi_step models 150 days into the future
from utils import getdata
from datetime import datetime

def main(len = 356):
    end = datetime.now()
    start = end.replace(year=end.year-1, )#day=end.day+1)
    end = end.strftime('%Y-%m-%dT00')
    start = start.strftime('%Y-%m-%dT00')
    symbol = input('Symbol to predict: ')
    data = getdata(start, end, data_name=symbol)
    print(data)

if __name__ == '__main__':
    main()