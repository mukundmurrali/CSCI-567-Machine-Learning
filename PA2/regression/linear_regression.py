"""
Do not change the input and output format.
If our script cannot run your code or the format is improper, your code will not be graded.

The only functions you need to implement in this template is linear_regression_noreg, linear_regression_invertible，regularized_linear_regression,
tune_lambda, test_error and mapping_data.
"""

import numpy as np
import pandas as pd

###### Q1.1 ######
def mean_absolute_error(w, X, y):
    """
    Compute the mean absolute error on test set given X, y, and model parameter w.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing test feature.
    - y: A numpy array of shape (num_samples, ) containing test label
    - w: a numpy array of shape (D, )
    Returns:
    - err: the mean absolute error
    """
    #####################################################
    # TODO 1: Fill in your code here #
    #####################################################
    err = None
    #compute the result
    result = np.dot(X, w).transpose()
    #find the difference
    diffs = np.absolute(np.subtract(y,result))
    #take the mean
    err = np.mean(diffs)
    return err

###### Q1.2 ######
def linear_regression_noreg(X, y):
  """
  Compute the weight parameter given X and y.
  Inputs:
  - X: A numpy array of shape (num_samples, D) containing feature.
  - y: A numpy array of shape (num_samples, ) containing label
  Returns:
  - w: a numpy array of shape (D, )
  """
  #####################################################
  #	TODO 2: Fill in your code here #
  #####################################################		
  w = None
  xTx = np.dot(X.transpose(), X)
  xTxInv = np.linalg.inv(xTx)
  xTy = np.dot(X.transpose(), y.transpose())
  w = np.dot(xTxInv, xTy)
  return w

###### Q1.3 ######
def linear_regression_invertible(X, y):
    """
    Compute the weight parameter given X and y.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    Returns:
    - w: a numpy array of shape (D, )
    """
    #####################################################
    # TODO 3: Fill in your code here #
    #####################################################

    w = None
    xTx = np.dot(X.transpose(), X)
    eigenValues,_ = np.linalg.eig(xTx)

    while True:
        if all(i >= 10 ** -5 for i in eigenValues):
            break
        identitymatrix = (10**(-1)*np.identity(len(eigenValues)))
        xTx += identitymatrix
        eigenValues,_ = np.linalg.eig(xTx)


    xTxInv = np.linalg.inv(xTx)
    xTy = np.dot(X.transpose(), y.transpose())
    w = np.dot(xTxInv, xTy)
    return w


###### Q1.4 ######
def regularized_linear_regression(X, y, lambd):
    """
    Compute the weight parameter given X, y and lambda.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    - lambd: a float number containing regularization strength
    Returns:
    - w: a numpy array of shape (D, )
    """
  #####################################################
  # TODO 4: Fill in your code here #
  #####################################################		
    w = None
    identitymatrix = np.identity(np.size(X,1))
    regularization = lambd * identitymatrix
    xTx = np.dot(X.transpose(), X)
    xTxregualr = xTx + regularization
    xTxregualrInv = np.linalg.inv(xTxregualr)
    xy = np.dot(X.transpose(), y.transpose())
    w = np.dot(xTxregualrInv,xy)
    return w

###### Q1.5 ######
def tune_lambda(Xtrain, ytrain, Xval, yval):
    """
    Find the best lambda value.
    Inputs:
    - Xtrain: A numpy array of shape (num_training_samples, D) containing training feature.
    - ytrain: A numpy array of shape (num_training_samples, ) containing training label
    - Xval: A numpy array of shape (num_val_samples, D) containing validation feature.
    - yval: A numpy array of shape (num_val_samples, ) containing validation label
    Returns:
    - bestlambda: the best lambda you find in lambds
    """
    #####################################################
    # TODO 5: Fill in your code here #
    #####################################################		
    bestlambda = None
    minError = None
    for i in range(-19, 20):
        exp = 10 ** i
        w = regularized_linear_regression(Xtrain, ytrain, exp)
        currError = mean_absolute_error(w, Xval, yval)
        if minError is None or currError < minError:
            minError = currError
            bestlambda = exp
    return bestlambda

###### Q1.6 ######
def mapping_data(X, power):
    """
    Mapping the data.
    Inputs:
    - X: A numpy array of shape (num_training_samples, D) containing training feature.
    - power: A integer that indicate the power in polynomial regression
    Returns:
    - X: mapped_X, You can manully calculate the size of X based on the power and original size of X
    """
    #####################################################
    # TODO 6: Fill in your code here #
    #####################################################
    X_orig = X;
    for pow in range(2, power + 1, 1):
        powerX = np.power(X_orig, pow)
        X = np.concatenate((X,powerX), axis=1)
    return X


