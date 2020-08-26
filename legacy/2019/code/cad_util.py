"""
Utility functions for computer-aided diagnosis.
"""

import numpy as np
import cad


def addones(X):
    # Add a column of all ones to a data matrix.
    # Input/output:
    # X - the data matrix

    if len(X.shape) < 2:
        X = X.reshape(-1,1)

    r, c = X.shape
    one_vec = np.ones((r,1))
    X = np.concatenate((X,one_vec), axis=1)

    return X

def plot_regression(X, Y, Theta, ax):
    # Visualize a trained linear regression model.
    # Input:
    # X - features, only works with 1D features
    # Y - targets
    # Theta - parameters of a polynomial regression model

    predictedY = addones(X).dot(Theta)
    
    ax.plot(X[:,0], Y, '*', label='Original data')
    
    plot_curve(X[:,0].reshape(-1,1), Theta, ax)
    
    ax.plot(X[:,0], predictedY, '*', label='Predicted data')

    for k in np.arange(len(X)):
        ax.plot([X[k,0], X[k,0]], [Y[k], predictedY[k]], c='r')


def plot_curve(X, Theta, ax):
    # Helper function for plot_regression.

    N_points = 100
    stretch = 1.05
    order = len(Theta)-1
    
    rangeX = np.linspace(np.min(X)/stretch, np.max(X)*stretch, N_points).reshape(-1,1)
    
    expandedX = []
    expandedRangeX = []

    for k in np.arange(order):
        if k==0:
            expandedX = X**(k+1)
            expandedRangeX = rangeX**(k+1)
        else:
            expandedX = np.concatenate((expandedX, X**(k+1)), axis=1)
            expandedRangeX = np.concatenate((expandedRangeX, rangeX**(k+1)), axis=1)

    ax.plot(rangeX, addones(expandedRangeX).dot(Theta), linewidth=2, label='Regression curve')


def plot_lr(X, Y, Theta, ax):
    # Visualize the training of a logistic regression model.
    # Input:
    # X - features, only works with 1D features
    # Y - targets
    # Theta - parameters of the logistic regression model

    num_range_points = 1000

    mn = np.min(X, axis=0)
    mx = np.max(X, axis=0)

    x1_range = np.linspace(mn[0], mx[0], num_range_points)
    x2_range = np.linspace(mn[1], mx[1], num_range_points)
    extent = np.min(x1_range), np.max(x1_range), np.min(x2_range), np.max(x2_range)

    x1, x2 = np.meshgrid(x1_range, x2_range)

    Xh = np.concatenate((x1.reshape(-1,1), x2.reshape(-1,1)), axis=1)

    Xh_ones = addones(Xh)
    ph = cad.sigmoid(Xh_ones.dot(Theta)) > 0.5

    decision_map = ph.reshape(num_range_points, num_range_points)

    im1 = ax.imshow(decision_map, cmap='coolwarm', alpha=0.5, interpolation='nearest', extent=extent, aspect='auto')

    return im1, Xh_ones, num_range_points


def montageRGB(X, ax):
    # Creates a 2D RGB montage of image slices from a 4D matrix
    # Input:
    # X - 4D matrix containing multiple 2D image slices in RGB format
    #     to be displayed as a montage / mosaic
    #
    # Adapted from http://www.datawrangling.org/python-montage-code-for-displaying-arrays/

    m, n, RGBval, count = X.shape
    mm = int(np.ceil(np.sqrt(count)))
    nn = mm
    M = np.zeros((mm * m, nn * n, 3))
    image_id = 0

    for j in np.arange(mm):
        for k in np.arange(nn):
            if image_id >= count:
                break
            sliceM, sliceN = j * m, k * n
            M[sliceN:sliceN + n, sliceM:sliceM + m, :] = X[:, :, :, image_id]
            image_id += 1

    M = np.flipud(np.rot90(M)).astype(np.uint8)
    ax.imshow(M)

    return M
