import pandas as pd
from extracting_data import extract
from stock_graph import graph, simulate
from preprocessing import preprocess_data
from decision_function_knn import validate_model_knn, train_model_knn
from decision_function_svm import validate_model_svm, train_model_svm
from decision_function_dt import validate_model_dt, train_model_dt
from decision_function_rf import validate_model_rf, train_model_rf
from decision_function_mlp import validate_model_mlp, train_model_mlp

SEPERATOR = '=========================================='

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

## MAIN / TEST PROGRAM with KNN Classifier
print(SEPERATOR)
print('KNN Classifier:')
# Load the data
ticker = input("Enter a ticker symbol: ")
data = csv_to_df(ticker)
# Fix the data
preprocess_data(data)
# Test the model
validate_model_knn(data)
# Store the model for making predictions
knn_model = train_model_knn(data)
# Make a prediction
X = data.iloc[0:, 1: -2]
# Can pass a row of a pandas dataframe directly
predictions = knn_model.predict(X)
print(predictions)
graph(data, predictions, ticker)
#simulate(data, predictions, ticker)
print(SEPERATOR)



## MAIN / TEST PROGRAM with SVM Classifier
print('SVM with RBF KERNAL Classifier:')
# Load the data
ticker = input("Enter a ticker symbol: ")
data = csv_to_df(ticker)
# Fix the data
preprocess_data(data)
# Test the model
validate_model_svm(data)
# Store the model for making predictions
SVM_model = train_model_svm(data)
# Make a prediction
X = data.iloc[0:, 1: -2]
# Can pass a row of a pandas dataframe directly
predictions = SVM_model.predict(X)
print(predictions)
graph(data, predictions, ticker)
#simulate(data, predictions, ticker)
print(SEPERATOR)


## MAIN / TEST PROGRAM with Decision Tree Classifier
print('Decision Tree Classifier:')
# Load the data
ticker = input("Enter a ticker symbol: ")
data = csv_to_df(ticker)
# Fix the data
preprocess_data(data)
# Test the model
validate_model_dt(data)
# Store the model for making predictions
DT_model = train_model_dt(data)
# Make a prediction
X = data.iloc[0:, 1: -2]
# Can pass a row of a pandas dataframe directly
predictions = DT_model.predict(X)
print(predictions)
graph(data, predictions, ticker)
#simulate(data, predictions, ticker)
print(SEPERATOR)


## MAIN / TEST PROGRAM with Random Forest Classifier
print('Random Forest Classifier:')
# Load the data
ticker = input("Enter a ticker symbol: ")
data = csv_to_df(ticker)
# Fix the data
preprocess_data(data)
# Test the model
validate_model_rf(data)
# Store the model for making predictions
RF_model = train_model_rf(data)
# Make a prediction
X = data.iloc[0:, 1: -2]
# Can pass a row of a pandas dataframe directly
predictions = RF_model.predict(X)
print(predictions)
graph(data, predictions, ticker)
#simulate(data, predictions, ticker)
print(SEPERATOR)

## MAIN / TEST PROGRAM MLPC Classifier
print('MLP Classifier:')
# Load the data
ticker = input("Enter a ticker symbol: ")
data = csv_to_df(ticker)
# Fix the data
preprocess_data(data)
# Test the model
validate_model_mlp(data)
# Store the model for making predictions
MLP_model = train_model_mlp(data)
# Make a prediction
X = data.iloc[0:, 1: -2]
# Can pass a row of a pandas dataframe directly
predictions = MLP_model.predict(X)
print(predictions)
graph(data, predictions, ticker)
#simulate(data, predictions, ticker)
print(SEPERATOR)


# Or construct one using a numpy array and transforming it
# X2 = np.array([2517, 231.6, 41872770, 229.517, 233.27, 226.46]).reshape(1, -1)
# predictions2 = knn_model.predict(X2)
# Output results
# print(predictions)
# print(predictions2)