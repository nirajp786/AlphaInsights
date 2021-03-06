from sklearn.model_selection import KFold, cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
import pandas as pd

pd.set_option('display.max_rows', None)

def validate_model_svm(data):
    """
    validate_model runs the SVM with defult RBF Kernal model using 10-fold cross-validation 
    and outputs the accuracy, as well as a confusion matrix and some other performance measures.
    ----------------------------------------------------------------------------
    :param data: a dataframe containing stock data
    :return: None(performance measures are printed)
    """   
    X = data.iloc[:-10, 1: -2]
    y = data.iloc[:-10, -1]

    ### SET UP CROSS-VALIDATION (K-FOLD) ###
    k = 10
    kf = KFold(n_splits=k)

    #Create a dictionary called param_grid and fill out some parameters for kernels, C and gamma
    #param_grid = {'C': [0.001, 0.01, 0.1, 1, 10], 
                  #'kernel':['rbf', 'poly', 'sigmoid'],
                  #'degree': [1, 2, 3],
                  #'gamma': [0.001, 0.01, 0.1, 1]}

    #Create a GridSearchCV object and fit it to the training data
    #svc_grid = GridSearchCV(SVC(),param_grid, cv=kf)
    #svc_grid.fit(X, y)
    #best_params = svc_grid.best_params_
    #model = SVC(best_params)

    #model = SVC(kernel='poly', degree=2, gamma='scale', C=1) #took forever and did not do well
    model = SVC(kernel='rbf', gamma='scale') #defult RBF Kernel

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
    return accuracy, PPV

def train_model_svm(data):
    """
    train_model trains the model using all of the data in the dataframe input. 
    It returns a trained model that make predictions using model.predict(X).
    ----------------------------------------------------------------------------
    :param data: a dataframe containing stock data for training the model
    :return: a trained model that can be used to make predictions
    """       
    X = data.iloc[:-10, 1: -2]
    y = data.iloc[:-10, -1]
    #model = GridSearchCV(SVC(), best_params) #I had best_params set as one of the params in this fuction and retruned from validat_model
    #model = SVC(kernel='poly', degree=2, gamma='scale', C=1)  #took forever and did not do well
    model = SVC(kernel='rbf', gamma='scale') #defult RBF Kernel
    model.fit(X, y)
    return model
