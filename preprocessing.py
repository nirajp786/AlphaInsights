import pandas as pd

def preprocess_data(data):
    """
    preprocess data prepares a pandas dataframe containing stock data from a CSV
    for analysis using a machine learning model. The pandas dataframe should have 5 
    columns: Date, Close, Volume, Open, High, and Low. The function will add 3 
    additional colummns:
    TenDayChange: The change in stock price after 10 days, with respect to the 
                  current day.
    Direction:    The label (1 for a price increase, 0 otherwise)
    RelativeDate: An integer representing the date, with 0 representing the earliest 
                  date available.
    --------------------------------------------------------------------------
    :param data: a dataframe containing stock data
    :return: None (the dataframe object is modified as per the description above)
    """     

    # Convert date from a string to a datetime object   
    data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')
    
    # Create a date column that can be used as a feature
    relative_date = []            
    for value in range(len(data)):
        relative_date.append(value)
    data.insert(len(data.columns), "RelativeDate", relative_date)
    
    # Create a new column showing how the price will change in 10 days
    ten_day_change = []
    for index, price in enumerate(data['Close']):
        if index < len(data) - 10:
            current_price = data.iloc[index][4] # Column 4 is the 'Close' column
            future_price = data.iloc[index + 10][4]
            price_change = future_price - current_price
            ten_day_change.append(price_change)
        else:
            ten_day_change.append(0)
    data.insert(len(data.columns), "TenDayChange", ten_day_change)

    # Create labels column (1 if price has increased, 0 otherwise)
    direction_of_change = []
    for price_change in ten_day_change:
        if price_change > 0:
            direction_of_change.append(1)
        else:
            direction_of_change.append(0)
    data.insert(len(data.columns), "Direction", direction_of_change)