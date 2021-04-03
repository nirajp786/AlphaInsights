from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold, cross_val_predict, GridSearchCV
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

def validate_model(data, classifier):
    """
    validate_model_svm runs the DecisionTree Classifier 10-fold cross-validation and outputs
    the accuracy, as well as a confusion matrix and some other performance measures.
    ----------------------------------------------------------------------------
    :param data: a dataframe containing stock data
    :return: None(performance measures are printed)
    """
    X = data.iloc[:-10, 1: -2]
    y = data.iloc[:-10, -1]

    ### SET UP CROSS-VALIDATION (K-FOLD) ###
    k = 10
    kf = KFold(n_splits=k)
    
    # Select the Model
    if (classifier == 'DT'):
        model = DecisionTreeClassifier()
    elif (classifier == 'SVM'):
        model = SVC(kernel='rbf', gamma='scale')
    elif (classifier == 'MLP'):
        model = MLPClassifier(max_iter=1000)
    elif (classifier == 'RF'):
        model = RandomForestClassifier()   
    else:
        # Find the best value for K
        knn = KNN()
        param_grid = {'n_neighbors': np.arange(1, 21)}
        knn_gscv = GridSearchCV(knn, param_grid, cv=10)
        knn_gscv.fit(X, y)
        k = int(knn_gscv.best_params_['n_neighbors'])
        # Instatiate the model
        model = KNN(n_neighbors = k)

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
    return accuracy, PPV, k


def train_model(data, classifier, k = 0):
    """
    train_model trains the model using all of the data in the dataframe input.
    It returns a trained model that make predictions using model.predict(X).
    ----------------------------------------------------------------------------
    :param data: a dataframe containing stock data for training the model
    :return: a trained model that can be used to make predictions
    """
    X = data.iloc[:-10, 1: -2]
    y = data.iloc[:-10, -1]
    
    # Select the Model
    if (classifier == 'DT'):
        model = DecisionTreeClassifier()
    elif (classifier == 'SVM'):
        model = SVC(kernel='rbf', gamma='scale')
    elif (classifier == 'MLP'):
        model = MLPClassifier(max_iter=1000)
    elif (classifier == 'RF'):
        model = RandomForestClassifier()   
    else:
        model = KNN(n_neighbors = k)    
    
    # Fit and return the model
    model.fit(X, y)
    return model
