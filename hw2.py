import numpy as np
import pandas as pd
import cvxpy as cp

# ====================== Utility Functions ======================

def meanSquaredError(predicted, actual):
    ''''
    Function to calculate the mean squared error
    Inputs: 
    predicted: a numpy array of predicted values for our model
    actual: a numpy array of actual values for housing prices

    Ouput: Mean squared error of our predictions
    '''
    
    r = actual - predicted # Residuals
    return np.sum(r**2)/len(predicted)

def preprocessData(filename, trainDataPct):
    '''
    Function to preprocess the data
    Inputs: None
    Output: None
    '''
    data = pd.read_csv(filename, delimiter='\t')

    # Using trainDataPct% of the data for training
    trainDataSize = int(trainDataPct * data.shape[0])
    trainData = data.iloc[:trainDataSize, :]
    X_train = trainData.iloc[:, :-1].values
    Y_train = (trainData.iloc[:, -1].values).reshape(-1, 1)

    # Using testDataPct% of the data for testing 
    testData = data.iloc[trainDataSize:, :]
    X_test = testData.iloc[:, :-1].values
    Y_test = (testData.iloc[:, -1].values).reshape(-1, 1)

    return X_train, Y_train, X_test, Y_test
    
def trainModel(X_train, Y_train):
    '''
    Function to train the model
    Inputs: None
    Output: None
    '''
    # Initilizing decision variables
    alpha = cp.Variable()
    beta = cp.Variable((X_train.shape[1], 1))

    # Defining function for our predictions
    Ytrain_pred = alpha + X_train @ beta

    # Defining objective function
    objective = cp.Minimize(cp.sum_squares(Ytrain_pred - Y_train))

    # Formulating problem
    problem = cp.Problem(objective)

    # solving the problem
    problem.solve()

    return Ytrain_pred.value, beta.value, alpha.value


# ====================== Data Preprocessing =====================

X_train1, Y_train1, X_test1, Y_test1 = preprocessData('housing.txt', 0.3) # Training with 30% of data
X_train2, Y_train2, X_test2, Y_test2 = preprocessData('housing.txt', 0.6) # Training with 60% of data


# ====================== Training the model ======================

Ytrain_pred1, beta1, alpha1 = trainModel(X_train1, Y_train1) # 30% training data
Ytrain_pred2, beta2, alpha2 = trainModel(X_train2, Y_train2) # 60% training data


# ====================== Testing the model ======================

Y_pred1 =  alpha1 + X_test1 @ beta1 # 30% training data
Y_pred2 =  alpha2 + X_test2 @ beta2 # 60% training data


# ====================== Printing Results ======================

# 30% training data
print('Mean squared error for training data (30%): ', round(meanSquaredError(Ytrain_pred1, Y_train1)))
print('Mean squared error for testing data (30%): ', round(meanSquaredError(Y_pred1, Y_test1)))

print('\n') # Creating separation for better readability

# 60% training data
print('Mean squared error for training data (60%): ', round(meanSquaredError(Ytrain_pred2, Y_train2)))
print('Mean squared error for testing data (60%): ', round(meanSquaredError(Y_pred2, Y_test2)))