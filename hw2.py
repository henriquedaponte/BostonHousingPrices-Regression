import numpy as np
import pandas as pd
import cvxpy as cp

def meanSquaredError(predicted, actual):
    ''''
    Function to calculate the mean squared error
    '''
    return np.sum((actual - predicted)**2)/len(predicted)

