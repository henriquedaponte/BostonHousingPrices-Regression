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
    return np.sum((actual - predicted)**2)/len(predicted)


# ====================== Data Preprocessing =====================

data = pd.read_csv('housing.txt', delimiter='\t')

# Using 70% of the data for training
trainDataSize = int(0.7*data.shape[0])
trainData = data.iloc[:trainDataSize, :]
X_train = trainData.iloc[:, :-1].values
Y_train = (trainData.iloc[:, -1].values).reshape(-1, 1)

# Using 30% of the data for testing 
testDataSize = int(0.3*data.shape[0])
testData = data.iloc[trainDataSize:, :]
X_test = testData.iloc[:, :-1].values
Y_test = (testData.iloc[:, -1].values).reshape(-1, 1)

# ====================== Training the model ======================

# Initilizing decision variables
alpha = cp.Variable((X_train.shape[0], 1))
beta = cp.Variable((X_train.shape[1], 1))

# Defining function for our predictions
Ytrain_pred = alpha + X_train @ beta

# Defining objective function
objective = cp.Minimize(cp.sum_squares(Ytrain_pred - Y_train))

# Formulating problem
problem = cp.Problem(objective)

# solving the problem
problem.solve()

# Printing results
print('Mean squared error for training data: ', round(meanSquaredError(Ytrain_pred.value, Y_train)))

# ====================== Testing the model ======================

print('alpha value: ', alpha.value)
print('alpha size: ', alpha.value.shape)
print('beta size: ', beta.value.shape)
print('X_test size: ', X_test.shape)

# Defining function for our predictions
Y_pred =  X_test @ beta.value

# Printing results
print('Mean squared error for testing data: ', round(meanSquaredError(Y_pred, Y_test)))
