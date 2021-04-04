import numpy as np

# ----------------------------------------------------------------------------
def moving_average(data, span, feature):
    """
    moving average generates a moving average with length equal to span
    using values from the desired feature (price, volume, etc.).
    -> For volume use 'Volume'
    -> For price use 'Close'
    -> Other options available ('Open' / 'High' / 'Low' / etc.)
    --------------------------------------------------------------------------
    :param data: a dataframe containing stock data
    :param predictions: the desired length for the moving avearge
    :param ticker: e feature used to calculate the average (price of volume)
    :return: Dataframe containing the information as described above
    """    
    # Moving average will store the set of moving average values
    moving_average = []
    # Window stores the span of values that will be used to calculate the MA
    window = np.zeros(shape = span)
    total = 0
    # Calculate the moving averages
    for index, row in data.iterrows():
        i = index % span
        removed = window[i]
        window[i] = row[feature]
        total = total + window[i]
        # The general case
        if (index >= span):
            total = total - removed
            average = total / span
            moving_average.append(average)
        # When the window is first filled we do not need to remove anything
        elif (index >= span - 1):
            average = total / span
            moving_average.append(average)
        # Prior to the window being filled
        else:
            # Should be "None", but causes model to fail, so use average of values
            moving_average.append(total / (i + 1)) 
    # Add a new column to the data frame, before target variable (can add multiple MAs)
    if (feature == 'Close'):
        metric = 'price'
    else:
        metric = 'volume'
    column_name = "MA" + str(span) + metric
    location = data.columns.get_loc('Direction')
    data.insert(location, column_name, moving_average)
