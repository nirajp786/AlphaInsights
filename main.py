from ledger import run_ledger
from extracting_data import csv_to_df
from stock_graph import graph, simulate
import feature_engineering as feat
from preprocessing import preprocess_data
from decision_function import validate_model, train_model

# MAIN PROGRAM ***************************************************************
SEPERATOR = '=========================================='

acc_knn, acc_svm, acc_dt, acc_df, acc_mlp = 0, 0 , 0, 0, 0
ppv_knn, ppv_svm, ppv_dt, ppv_rf, ppv_mlp = 0, 0, 0, 0, 0

# LOAD THE DATA **************************************************************
ticker = input("Enter a ticker symbol: ")
data = csv_to_df(ticker)

# PREPROCESSING & FEATURE ENGINEERING ****************************************
preprocess_data(data)
feat.moving_average(data, 10, 'Close')
feat.moving_average(data, 30, 'Close')
feat.moving_average(data, 20, 'Volume')

# DETERMINING THE BEST DECISION FUNCTION *************************************

# KNN Classifier -------------------------------------------------------------
print(SEPERATOR)
print('KNN Classifier:')
# Test the model
acc_knn, ppv_knn, k = validate_model(data, 'KNN')
# Store the model for making predictions
knn_model = train_model(data, 'KNN', k)
# Identify the Test Population
X = data.iloc[0:, 1: -2]
# Store the predictions
predictions_knn = knn_model.predict(X)
# Show Predictions on Chart
graph(data, predictions_knn, ticker,'KNN')

# SVM Classifier -------------------------------------------------------------
print(SEPERATOR)
print('SVM Classifier:')
# Test the model
acc_svm, ppv_svm, k = validate_model(data, 'SVM')
# Store the model for making predictions
svm_model = train_model(data, 'SVM')
# Identify the Test Population
X = data.iloc[0:, 1: -2]
# Store the predictions
predictions_svm = svm_model.predict(X)
graph(data, predictions_svm , ticker, 'SVM')

# DT Classifier --------------------------------------------------------------
print(SEPERATOR)
print('DT Classifier:')
# Test the model
acc_dt, ppv_dt, k = validate_model(data, 'DT')
# Store the model for making predictions
dt_model = train_model(data, 'DT')
# Identify the Test Population
X = data.iloc[0:, 1: -2]
# Store the predictions
predictions_dt = dt_model.predict(X)
graph(data, predictions_dt, ticker, 'DT')

# RF Classifier --------------------------------------------------------------
print(SEPERATOR)
print('RF Classifier:')
# Test the model
acc_rf, ppv_rf, k = validate_model(data, 'RF')
# Store the model for making predictions
rf_model = train_model(data, 'RF')
# Identify the Test Population
X = data.iloc[0:, 1: -2]
# Store the predictions
predictions_rf = rf_model.predict(X)
graph(data, predictions_rf , ticker, 'RF')

# MLP Classifier -------------------------------------------------------------
print(SEPERATOR)
print('MLP Classifier:')
# Test the model
acc_mlp, ppv_mlp, k = validate_model(data, 'MLP')
# Store the model for making predictions
mlp_model = train_model(data, 'MLP')
# Identify the Test Population
X = data.iloc[0:, 1: -2]
# Store the predictions
predictions_mlp = mlp_model.predict(X)
graph(data, predictions_mlp , ticker, 'MLP')

# DETERMINE THE BEST ACCURACY / PRECISION ************************************

# Aaccuracy ------------------------------------------------------------------
print(SEPERATOR)
dict1 = {'KNN': acc_knn, 'SVM': acc_svm, 'DT': acc_dt, 'RF': acc_rf, 'MLP': acc_mlp}
max_key = max(dict1, key=dict1.get)
max_value = max(dict1.values())
print("Best Accuracy Among Five Classifiers")
print("------------------------------------")
print("Max Accuracy achieved by", max_key, "classifier")
print("Max Accuracy = {:.4f}".format(max_value))

# Precision ------------------------------------------------------------------
print(SEPERATOR)
dict2 = {'KNN': ppv_knn, 'SVM': ppv_svm, 'DT': ppv_dt, 'RF':ppv_rf, 'MLP': ppv_mlp}
max_key1 = max(dict2, key=dict2.get)
max_value1 = max(dict2.values())
print("Best Precision Among Five Classifiers")
print("------------------------------------")
print("Max Precision Achieved By", max_key1, "classifier")
print("Max Precision = {:.4f}".format(max_value1))

# SIMULATE RESULT USING BEST MODEL *******************************************
print(SEPERATOR)
if max_key1 == 'KNN':
    transaction_history = run_ledger(data, predictions_knn, ticker)
    simulate(data, predictions_knn, ticker)
elif max_key1 == 'SVM':
    transaction_history = run_ledger(data, predictions_svm, ticker)    
    simulate(data, predictions_svm, ticker)
elif max_key1 == 'DT':
    transaction_history = run_ledger(data, predictions_dt, ticker)    
    simulate(data, predictions_dt, ticker)
elif max_key1 == 'RF':
    transaction_history = run_ledger(data, predictions_rf, ticker)
    simulate(data, predictions_rf, ticker)
else:
    transaction_history = run_ledger(data, predictions_mlp, ticker)
    simulate(data, predictions_mlp, ticker)