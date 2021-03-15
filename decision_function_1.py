import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

pd.set_option('display.max_rows', None)


def csv_to_df(csv):
    """
    csv_to_df reads a .csv file and converts it into a pandas dataframe
    ----------------------------------------------------------------------------
    :param csv: the name of the csv
    :return: a pandas dataframe containing the information from the csv file
    """     
    file_path = csv + ".csv"
    return pd.read_csv(file_path)

def preprocess_data(data):
    """
    preprocess data prepares a pandas dataframe containing stock data from a CSV
    for analysis using a machine learning model. The pandas dataframe should have 5 
    columns: Date, Close/Last, Volume, Open, High, and Low. The function will
    change the 'Close/Last' column to 'Close', remove '$'s from the price data, 
    convert the price data from floats to integers, and add 3 additional colummns:
    TenDayChange: The change in stock price after 10 days, with respect to the 
                  current day.
    Direction:    The label (1 for a price increase, 0 otherwise)
    RelativeDate: An integer representing the date, with 0 representing the earliest 
                  date available.
    ----------------------------------------------------------------------------
    :param data: a dataframe containing stock data
    :return: None(the dataframe object is modified as per the description above)
    """     
    # Fix column names so they only contain alphanumeric characters  
    data.rename(columns = {'Close/Last':'Close'}, inplace = True) 
    # Remove $ from price data
    data.Close = [x.strip('$') for x in data.Close]
    data.Open = [x.strip('$') for x in data.Open]
    data.High = [x.strip('$') for x in data.High]
    data.Low = [x.strip('$') for x in data.Low]
    
    # Convert price data to floats
    data.Close = data.Close.astype(float) 
    data.Open = data.Open.astype(float) 
    data.High = data.High.astype(float) 
    data.Low = data.Low.astype(float) 

    # Create a new column showing how the price will change in 10 days
    ten_day_change = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for index, price in enumerate(data['Close']):
        if index > 9:
            price_today = data.iloc[index][1]
            price_in_10_days = data.iloc[index - 10][1]
            price_change = price_in_10_days - price_today
            ten_day_change.append(price_change)
    data.insert(len(data.columns), "TenDayChange", ten_day_change)

    # Create labels column (1 if price has increased, 0 otherwise)
    direction_of_change = []
    for price_change in ten_day_change:
        if price_change > 0:
            direction_of_change.append(1)
        else:
            direction_of_change.append(0)
    data.insert(len(data.columns), "Direction", direction_of_change)

    # Create a date column that can be used as a feature
    relative_date = []            
    for value in range(2517):
        relative_date.append(abs(value - 2517))
    data.insert(1, "RelativeDate", relative_date)
                
def validate_model(data):
    """
    validate_model runs the KNN model using 10-fold cross-validation and outputs
    the accuracy, as well as a confusion matrix and some other performance measures.
    ----------------------------------------------------------------------------
    :param data: a dataframe containing stock data
    :return: None(performance measures are printed)
    """   
    X = data.iloc[10:, 1: -2]
    y = data.iloc[10:, -1]

    ### SET UP CROSS-VALIDATION (K-FOLD) ###
    k = 10
    kf = KFold(n_splits=k)

    model = SVC() # SVM-L: linear kernel

    y_pred = cross_val_predict(model, X, y, cv = kf)

    cm = confusion_matrix(y, y_pred)
    TP = cm[1, 1] # True Positives
    TN = cm[0, 0] # True Negatives
    FP = cm[0, 1] # False Positives
    FN = cm[1, 0] # False Negatives
    n = len(X)   # Number of Samples
    # Positive Predicted Value
    PPV = TP / (TP + FP)

    # Negative Predicted Value
    NPV = TN / (TN + FN)

    # Specificity
    specificity = TN / (TN + FP)

    # Sensitivity
    sensitivity = TP / (TP + FN)

    # Accuracy
    accuracy = (TP + TN) / n

    #print("Confusion Matrix:")
    print(cm)
    print("PPV = {:.4f}".format(PPV))
    print("NPV = {:.4f}".format(NPV))
    print("Specificity = {:.4f}".format(specificity))
    print("Sensitivity = {:.4f}".format(sensitivity))
    print("Accuracy = {:.4f}".format(accuracy))
    
def train_model(data):
    """
    train_model trains the model using all of the data in the dataframe input. 
    It returns a trained model that make predictions using model.predict(X).
    ----------------------------------------------------------------------------
    :param data: a dataframe containing stock data for training the model
    :return: a trained model that can be used to make predictions
    """       
    X = data.iloc[10:, 1: -2]
    y = data.iloc[10:, -1]
    model = SVC() # SVM-L: linear kernel
    model.fit(X, y)
    return model
  
## MAIN / TEST PROGRAM
# Load the data
data = csv_to_df('microsoft')

# Fix the data
preprocess_data(data)

# Test the model
validate_model(data)

# Store the model for making predictions
knn_model = train_model(data)

# Make a prediction
X = data.iloc[0:1, 1: -2]
# Can pass a row of a pandas dataframe directly
predictions = knn_model.predict(X)
# Or construct one using a numpy array and transforming it
X2 = np.array([2517, 231.6, 41872770, 229.517, 233.27, 226.46]).reshape(1, -1)
predictions2 = knn_model.predict(X2)
# Output results
print(predictions)
print(predictions2)

#Results with %62.2 accuracy
#PPV = 0.6238
#NPV = 0.2500
#Specificity = 0.0021
#Accuracy = 0.6227
