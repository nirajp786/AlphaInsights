import pandas as pd
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

pd.set_option('display.max_rows', None)

	
def validate_model(data):
    """
    validate_model runs the KNN model using 10-fold cross-validation and outputs
    the accuracy, as well as a confusion matrix and some other performance measures.
    ----------------------------------------------------------------------------
    :param data: a dataframe containing stock data
    :return: None(performance measures are printed)
    """  
    global accuracy
    X = data.iloc[:, 1: -2]
    y = data.iloc[:, -1]

    ### SET UP CROSS-VALIDATION (K-FOLD) ###  
    k = 10
    kf = KFold(n_splits=k)

    model = KNN() # SVM-L: linear kernel

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
	
    if accuracy < 0.5500:
       model = KNN() # Defaults to 5 neighbors, Euclidean distance
	   
    #print("Confusion Matrix:")
    print(cm)
    print("PPV = {:.4f}".format(PPV))
    print("NPV = {:.4f}".format(NPV))
    print("Specificity = {:.4f}".format(specificity))
    print("Sensitivity = {:.4f}".format(sensitivity))
    print("Accuracy = {:.4f}".format(accuracy))
	
def get_accuracy ():

    return accuracy
    
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
    model = SVC()
    accuracy = get_accuracy ()
    if accuracy < 0.5500:
       model = KNN() # Defaults to 5 neighbors, Euclidean distance
    model.fit(X, y)
    return model
  