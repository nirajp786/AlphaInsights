import time
import datetime
import pandas as pd
import csv
import requests

def extract(ticker):
    start_period = int(time.mktime(datetime.datetime(2020, 1, 1, 23, 59).timetuple()))
    end_period = int(time.mktime(datetime.datetime(2021, 3, 13, 23, 59).timetuple()))

    interval = '1d' #setting the interval to be 1 week

    query = f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={start_period}&period2={end_period}&interval={interval}&events=history&includeAdjustedClose=true"
    # df = pd.read_csv(query)
    # df.to_csv(ticker + 'test' +'.csv')
    # return df

    response = requests.get(query)
    with open(ticker+'.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for line in response.iter_lines():
            writer.writerow(line.decode('utf-8').split(','))