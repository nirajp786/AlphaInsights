import time
import datetime
import pandas as pd

ticker = "MSFT"
start_period = int(time.mktime(datetime.datetime(2021, 1, 1, 23, 59).timetuple()))
end_period = int(time.mktime(datetime.datetime(2021, 3, 13, 23, 59).timetuple()))

interval = '1d' #setting the interval to be 1 week

query = f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={start_period}&period2={end_period}&interval={interval}&events=history&includeAdjustedClose=true"
df = pd.read_csv(query)
print(df)