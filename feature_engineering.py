import numpy as np

def moving_average(data, span, style):

    # Moving average will store the set of moving average values
    moving_average = []
    # Window stores the span of values that will be used to calculate the MA
    window = np.zeros(shape = span)
    total = 0
    # Calculate the moving averages
    for index, row in data.iterrows():
        i = index % span
        removed = window[i]
        window[i] = row[style]
        total = total + window[i]
        # The general case
        if (index >= span):
            total = total - removed
            average = total / span
            moving_average.append(average)
        # When span is first filled we do not need to remove anything
        elif (index >= span - 1):
            average = total / span
            moving_average.append(average)
        # Prior to the window being filled
        else:
            moving_average.append(total / (i + 1)) # Should be "None, but causes model to fail
    # Add a new column to the data frame, before target variable (can add multiple MAs)
    if (style == 'Close'):
        metric = 'price'
    else:
        metric = 'volume'
    column_name = "MA" + str(span) + metric
    location = data.columns.get_loc('Direction')
    data.insert(location, column_name, moving_average)
