import pandas as pd
from extracting_data import extract
from preprocessing import preprocess_data
from stock_graph import graph, simulate
from decision_function import validate_model, train_model
from ledger import run_ledger
import feature_engineering as feat

def csv_to_df(csv):
    """
    csv_to_df reads a .csv file and converts it into a pandas dataframe
    ----------------------------------------------------------------------------
    :param csv: the name of the csv
    :return: a pandas dataframe containing the information from the csv file
    """     
    extract(csv)
    file_path = csv + ".csv"
    return pd.read_csv(file_path)

## MAIN / TEST PROGRAM
# Load the data

ticker = input("Enter a ticker symbol: ")
data = csv_to_df(ticker)
# Fix the data
preprocess_data(data)

#Add moving averages
feat.moving_average(data, 10)
feat.moving_average(data, 30)

# Test the model
validate_model(data)

# Store the model for making predictions
knn_model = train_model(data)

# Make a prediction
X = data.iloc[0:, 1: -2]
# Can pass a row of a pandas dataframe directly
predictions = knn_model.predict(X)

#transaction_history = run_ledger(data, predictions, ticker)
print(data)