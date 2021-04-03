from ledger import run_ledger
from extracting_data import csv_to_df
from stock_graph import graph, simulate
import feature_engineering as feat
from preprocessing import preprocess_data
from decision_function import validate_model, train_model

SEPERATOR = '=========================================='



ticker = input("Enter a ticker symbol: ")

acc_knn=0
acc_svm=0
acc_dt=0
acc4_df=0
acc5_mpl=0

ppv_knn= 0
ppv_svm= 0
ppv_dt=0
ppv_rf=0
ppv_mpl=0


## MAIN / TEST PROGRAM with KNN Classifier
print(SEPERATOR)
print('KNN Classifier:')
# Load the data
data = csv_to_df(ticker)
# Fix the data
preprocess_data(data)
#Add moving averages
feat.moving_average(data, 10, 'Close')
feat.moving_average(data, 30, 'Close')
# Test the model
acc_knn, ppv_knn, k = validate_model(data, 'KNN')
# Store the model for making predictions
knn_model = train_model(data, 'KNN', k)
# Make a prediction
X = data.iloc[0:, 1: -2]
# Can pass a row of a pandas dataframe directly
predictions_knn = knn_model.predict(X)
print(predictions_knn)
graph(data, predictions_knn, ticker,'KNN')
#transaction_history = run_ledger(data, predictions_knn, ticker)
#simulate(data, predictions_knn, ticker)
print(SEPERATOR)

## MAIN / TEST PROGRAM with SVM Classifier
print('SVM with RBF KERNAL Classifier:')
# Load the data
data = csv_to_df(ticker)
# Fix the data
preprocess_data(data)
#Add moving averages
feat.moving_average(data, 10, 'Close')
feat.moving_average(data, 30, 'Close')
# Test the model
acc_svm, ppv_svm, k = validate_model(data, 'SVM')
# Store the model for making predictions
SVM_model = train_model(data, 'SVM')
# Make a prediction
X = data.iloc[0:, 1: -2]
# Can pass a row of a pandas dataframe directly
predictions_svm = SVM_model.predict(X)
print(predictions_svm)
graph(data, predictions_svm , ticker, 'SVM')
#transaction_history = run_ledger(data, predictions_svm, ticker)
#simulate(data, predictions_svm, ticker)
print(SEPERATOR)


## MAIN / TEST PROGRAM with Decision Tree Classifier
print('Decision Tree Classifier:')
# Load the data
data = csv_to_df(ticker)
# Fix the data
preprocess_data(data)
#Add moving averages
feat.moving_average(data, 10, 'Close')
feat.moving_average(data, 30, 'Close')
# Test the model
acc_dt, ppv_dt, k = validate_model(data, 'DT')
# Store the model for making predictions
DT_model = train_model(data, 'DT')
# Make a prediction
X = data.iloc[0:, 1: -2]
# Can pass a row of a pandas dataframe directly
predictions_dt = DT_model.predict(X)
print(predictions_dt)
graph(data, predictions_dt, ticker, 'DT')
#transaction_history = run_ledger(data, predictions_dt, ticker)
#simulate(data, predictions_dt, ticker)
print(SEPERATOR)

## MAIN / TEST PROGRAM with Random Forest Classifier
print('Random Forest Classifier:')
# Load the data
data = csv_to_df(ticker)
# Fix the data
preprocess_data(data)
#Add moving averages
feat.moving_average(data, 10, 'Close')
feat.moving_average(data, 30, 'Close')
# Test the model
acc_rf, ppv_rf, k = validate_model(data, 'RF')
# Store the model for making predictions
RF_model = train_model(data, 'RF')
# Make a prediction
X = data.iloc[0:, 1: -2]
# Can pass a row of a pandas dataframe directly
predictions_rf = RF_model.predict(X)
print(predictions_rf)
graph(data, predictions_rf, ticker, 'RF')
#transaction_history = run_ledger(data, predictions_rf, ticker)
#simulate(data, predictions_rf, ticker)
print(SEPERATOR)

## MAIN / TEST PROGRAM MLPC Classifier
print('MLP Classifier:')
# Load the data
data = csv_to_df(ticker)
# Fix the data
preprocess_data(data)
#Add moving averages
feat.moving_average(data, 10, 'Close')
feat.moving_average(data, 30, 'Close')
# Test the model
acc_mlp, ppv_mlp, k  = validate_model(data, 'MLP')
# Store the model for making predictions
MLP_model = train_model(data, 'MLP')
# Make a prediction
X = data.iloc[0:, 1: -2]
# Can pass a row of a pandas dataframe directly
predictions_mpl = MLP_model.predict(X)
print(predictions_mpl)
graph(data, predictions_mpl, ticker, 'MPL')
#transaction_history = run_ledger(data, predictions_mpl, ticker)
#simulate(data, predictions_mpl, ticker)
print(SEPERATOR)

#getting the best accuracy among five classifiers
dict1 = {'knn': acc_knn, 'svm': acc_svm, 'dt': acc_dt, 'rf': acc_rf, 'mlp': acc_mlp}
max_key = max(dict1, key=dict1.get)
max_value = max(dict1.values())
#print(max_key)
#print(max_value)
print("Best Accuracy Among Five Classifiers")
print("------------------------------------")
print("Max Accuracy achieved by", max_key, "classifier")
print("Max Accuracy = {:.4f}".format(max_value))

#getting the best PPV among five classifiers
dict2 = {'KNN': ppv_knn, 'SVM': ppv_svm, 'DT': ppv_dt, 'RF':ppv_rf, 'MPL': ppv_mlp}

max_key1 = max(dict2, key=dict2.get)
max_value1 = max(dict2.values())
print(SEPERATOR)
print("Best PPV Among Five Classifiers")
print("-------------------------------")
print("Max PPV achieved by", max_key1, "classifier")
print("Max PPV = {:.4f}".format(max_value1))
print(SEPERATOR)

#printing animated graph for classifier with best PPV

if max_key1 == 'KNN':
    print('Message to Carson: PLZ uncomment the next line to print animated graph')
    #simulate(data, predictions_knn, ticker)

elif max_key1 == 'SVM':
    print('Message to Carson: PLZ uncomment the next line to print animated graph')
    #simulate(data, predictions_svm, ticker)

elif max_key1 == 'DT':
    print('Message to Carson: PLZ uncomment the next line to print animated graph')
    #simulate(data, predictions_dt, ticker)

elif max_key1 == 'RF':
    print('Message to Carson: PLZ uncomment the next line to print animated graph')
    #simulate(data, predictions_rf, ticker)

elif max_key1 == 'MPL':
    print('Message to Carson: PLZ uncomment the next line to print animated graph')
    #simulate(data, predictions_mpl, ticker)


# Or construct one using a numpy array and transforming it
# X2 = np.array([2517, 231.6, 41872770, 229.517, 233.27, 226.46]).reshape(1, -1)
# predictions2 = knn_model.predict(X2)
# Output results
# print(predictions)
# print(predictions2)
