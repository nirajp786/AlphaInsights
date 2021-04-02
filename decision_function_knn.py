from sklearn.model_selection import KFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

pd.set_option('display.max_rows', None)

	
def validate_model_knn(data):
    """
    validate_model runs the KNN model using 10-fold cross-validation and GridSearchCV
    output: the accuracy, as well as a confusion matrix and some other performance measures.
    ----------------------------------------------------------------------------
    :param data: a dataframe containing stock data
    :return: None(performance measures are printed)
    """   
    X = data.iloc[10:, 1: -2]
    y = data.iloc[10:, -1]

    ### SET UP CROSS-VALIDATION (K-FOLD) ###
    k = 10
    kf = KFold(n_splits=k)
	
    
	## calculating Error rate
    error = []
    for i in range(1, 21):
     knn = KNN(n_neighbors=i)
     knn.fit(X, y)
     pred_i = knn.predict(X)
     error.append(np.mean(pred_i != y))

    ## ploting error
	## Best K value corresponds to least error rate
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, 21), error, color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
    plt.title('Best k Value',fontsize=20)
    plt.xlabel('K Value')
    plt.ylabel('Mean Error')
    plt.tight_layout()
    plt.savefig('./plots/knn-{:s}.png')
	
	##Applying GridSearch to get best K value
    knn0 = KNN()
    param_grid = {'n_neighbors': np.arange(1, 21)}
    knn_gscv = GridSearchCV(knn0, param_grid, cv=10)
    knn_gscv.fit(X, y)
    k = int(knn_gscv.best_params_['n_neighbors'])
    model = KNN (n_neighbors = k)
	
	
	
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
    
def train_model_knn(data):
    """
    train_model trains the model using all of the data in the dataframe input. 
    It returns a trained model that make predictions using model.predict(X).
    ----------------------------------------------------------------------------
    :param data: a dataframe containing stock data for training the model
    :return: a trained model that can be used to make predictions
    """       
    X = data.iloc[10:, 1: -2]
    y = data.iloc[10:, -1]
    knn0 = KNN()
    param_grid = {'n_neighbors': np.arange(1, 21)}
    knn_gscv = GridSearchCV(knn0, param_grid, cv=10)
    knn_gscv.fit(X, y)
    k = int(knn_gscv.best_params_['n_neighbors'])
    model = KNN (n_neighbors = k)
    model.fit(X, y)
    return model