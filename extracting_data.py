import time
import datetime
import csv
import requests
import pandas as pd

# ----------------------------------------------------------------------------
def extract(ticker):  
    """
    extract obtains stock data from a URL (Yahoo Finance) and generates a CSV.
    --------------------------------------------------------------------------
    :param csv: a stock symbol
    :return: None (a csv is generated)
    """
    start_period = int(time.mktime(datetime.datetime(2020, 1, 1, 23, 59).timetuple()))
    end_period = int(time.mktime(datetime.datetime(2021, 3, 13, 23, 59).timetuple()))

    interval = '1d' #setting the interval to be 1 day (?)

    query = f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={start_period}&period2={end_period}&interval={interval}&events=history&includeAdjustedClose=true"

    response = requests.get(query)
    with open(ticker+'.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for line in response.iter_lines():
            writer.writerow(line.decode('utf-8').split(','))

# ----------------------------------------------------------------------------            
def csv_to_df(csv):
    """
    csv_to_df reads a .csv file and converts it into a pandas dataframe
    --------------------------------------------------------------------------
    :param csv: the name of the csv
    :return: a pandas dataframe containing the information from the csv file
    """
    extract(csv)
    file_path = csv + ".csv"
    return pd.read_csv(file_path)