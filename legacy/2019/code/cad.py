"""
CAD module code main code.
"""

import numpy as np
import cad_util as util


def sigmoid(a):
    # Computes the sigmoid function
    # Input:
    # a - value for which the sigmoid function should be computed
    # Output:
    # s - output of the sigmoid function

    #-------------------------------------------------------------------#
    # TODO: Implement the expression for the sigmoid function. The
    #  function must also work for vector inputs: for example if the input
    #  is [1 -1 2] the output should be a vector of the same size with the
    #  sigmoid values for every element of the input vector.
    #-------------------------------------------------------------------#

    return s


def lr_nll(X, Y, Theta):
    # Computes the negative log-likelihood (NLL) loss for the logistic
    # regression classifier.
    # Input:
    # X - the data matrix
    # Y - targets vector
    # Theta - parameters of the logistic regression model
    # Ouput:
    # L - the negative log-likelihood loss

    # compute the predicted probability by the logistic regression model
    p = sigmoid(X.dot(Theta))

    #-------------------------------------------------------------------#
    # TODO: Implement the expression for the NLL.
    #-------------------------------------------------------------------#

    return L


def lr_agrad(X, Y, Theta):
    # Gradient of the negative log-likelihood for a logistic regression
    # classifier.
    # Input:
    # X - the data matrix
    # Y - targets vector
    # Theta - parameters of the logistic regression model
    # Example inputs:
    # X - training_x_ones.shape=(100, 1729)
    # Y - training_y[idx].shape=(100, 1)
    # Theta - Theta.shape=(1729, 1)
    # Ouput:
    # g - gradient of the negative log-likelihood loss
    #
    a = X.dot(Theta)
    p = sigmoid(a)
    g = np.sum((p - Y)*X, axis=0).reshape(1,-1)

    return g
