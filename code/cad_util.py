"""
Utility functions for computer-aided diagnosis.
"""

import numpy as np
import cad

import matplotlib.pyplot as plt
import matplotlib.cm as cm

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

def scatter_data(X, Y, feature0=0, feature1=1, ax=None):
    # scater_data displays a scatterplot of at most 1000 samples from dataset X, and gives each point
    # a different color based on its label in Y

    k = 1000
    if len(X) > k:
        idx = np.random.randint(len(X), size=k)
        X = X[idx,:]
        Y = Y[idx]

    class_labels, indices1, indices2 = np.unique(Y, return_index=True, return_inverse=True)
    if ax is None:
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111)
        ax.grid()

    colors = cm.rainbow(np.linspace(0, 1, len(class_labels)))
    for i, c in zip(np.arange(len(class_labels)), colors):
        idx2 = indices2 == class_labels[i]
        lbl = 'X, class '+str(i)
        ax.scatter(X[idx2,feature0], X[idx2,feature1], color=c, label=lbl)

    return ax

def generate_gaussian_data(N=100, mu1=[0,0], mu2=[2,0], sigma1=[[1,0],[0,1]], sigma2=[[1,0],[0,1]]):
    # Generates a 2D toy dataset with 2 classes, N samples per class. 
    # Class 1 is Gaussian distributed with mu1 and sigma2
    # Class 2 is Gaussian distributed with mu2 and sigma2. 
    # By default, N=100, mu1=[0 0], mu2=[2,0], sigma1=sigma2 = [1 0; 0 1]
    #
    # Input:
    # N             - Number of samples per class (2N in total)
    # mu1           - 1x2 vector, mean of class 1 
    # mu2           - 1x2 vector, mean of class 2 
    # sigma1        - 2x2 matrix, covariance of class 1 
    # sigma2        - 2x2 matrix, covariance of class 2 
    
    # Generate class 1
    A = np.linalg.cholesky(sigma1)
    data1 = np.random.randn(N,2).dot(A) + mu1     #Rotate data according to covariance matrix (must be positive definite!), and add the mean
    # Generate class 2
    B = np.linalg.cholesky(sigma2)
    data2 = np.random.randn(N,2).dot(B) + mu2
    
    # Put the data together
    X = np.concatenate((data1, data2), axis=0)
    # Create labels
    Y = np.concatenate((np.zeros((N,1)), np.ones((N,1))), axis=0)

    return X, Y
