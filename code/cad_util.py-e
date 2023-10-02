"""
Utility functions for CAD.
"""

import numpy as np
import cad
import random
random.seed(0)

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

def reshape_and_normalize(training_images, validation_images, test_images):
    # Images are reshaped to be 1D vectors of size 24x24x3 and subsequently
    # pixel intensities are normalized to zero mean unit variance
    #
    # Input:
    # images         - Matrix of images in shape (24, 24, 3, n_images)
    
    ## dataset preparation
    imageSize = training_images.shape
    # every pixel is a feature so the number of features is:
    # height x width x color channels
    numFeatures = imageSize[0]*imageSize[1]*imageSize[2]
    training_x = training_images.reshape(numFeatures, training_images.shape[3]).T.astype(float)        # (14607, 1728)
    validation_x = validation_images.reshape(numFeatures, validation_images.shape[3]).T.astype(float)  # (7303, 1728)
    test_x = test_images.reshape(numFeatures, test_images.shape[3]).T.astype(float)                    # (20730, 1728)

    # the training will progress much better if we
    # normalize the features
    meanTrain = np.mean(training_x, axis=0).reshape(1,-1)
    stdTrain = np.std(training_x, axis=0).reshape(1,-1)

    training_x = training_x - np.tile(meanTrain, (training_x.shape[0], 1))
    training_x = training_x / np.tile(stdTrain, (training_x.shape[0], 1))

    validation_x = validation_x - np.tile(meanTrain, (validation_x.shape[0], 1))
    validation_x = validation_x / np.tile(stdTrain, (validation_x.shape[0], 1))

    test_x = test_x - np.tile(meanTrain, (test_x.shape[0], 1))
    test_x = test_x / np.tile(stdTrain, (test_x.shape[0], 1))
    return training_x, validation_x, test_x


def shuffle_training_x(x, y):
    # Shuffle training data x with corresponding label y
    #
    # Input:
    # x             - matrix with training data of shape (n_images, 1728)
    # y             - matrix with training labels of shape (n_images, 1)
    indices = list(range(x.shape[0]))
    random.shuffle(indices)
    
    new_image = np.zeros(x.shape).astype(x.dtype)
    new_y = np.zeros(y.shape).astype(y.dtype)
    
    for original_index in range(x.shape[0]):
        new_index = indices[original_index]
        
        new_image[new_index,:] = x[original_index,:]
        new_y[new_index, :] = y[original_index, :]
    
    return new_image, new_y
        

def visualize_big_small_images(images, y, n=10):
    # Randomly select images to be either visualized as large or small
    #
    # Input:
    # images        - matrix with training data of shape (24, 24, 3, n_images)
    # y             - matrix with training labels of shape (n_images, 1)
    # n             - create an image with nxn tiles
    
    # randomly sample images
    n = n*n
    indices = list(range(images.shape[-1]))
    random.shuffle(indices)
    
    shape = images.shape
    big = np.zeros((shape[0], shape[1], shape[2], 0)).astype(images.dtype)
    small = np.zeros((shape[0], shape[1], shape[2], 0)).astype(images.dtype)
    n_big, n_small, i = 0, 0, 0
    while n_big < n or n_small < n:
        tile = indices[i]
        thislabel = y[tile, 0]
        this_x = images[:,:,:,tile][:,:,:,None]
        this_x = (this_x - this_x.min()) / (this_x.max() - this_x.min())
        this_x = (255*this_x).astype(np.uint8)
        if thislabel == 1 and n_big < n:
            big = np.concatenate((big, this_x), axis=3)
            n_big += 1
        elif thislabel == 0 and n_small < n:
            small = np.concatenate((small, this_x), axis=3)
            n_small += 1
        i += 1
    fig = plt.figure(figsize=(16,8))
    ax1  = fig.add_subplot(121)
    montageRGB(big, ax1)
    plt.title('large')
    ax1  = fig.add_subplot(122)
    montageRGB(small, ax1)
    plt.title('small')
    plt.show()    

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

def sigmoid(x):
    # Calculate sigmoid for given x
    return 1.0/(1+ np.exp(-x))

def sigmoid_derivative(sig_x):
    # Caluclate the derivative of sigmoid for given sigmoid(x)
    return sig_x * (1.0 - sig_x)

def loss(prediction, label):
    # L2 loss for notebook 2.5 implementation
    return np.mean((prediction - label)**2)

def init_model(w1_shape, w2_shape):
    # Model initialization for notebook 2.5
    w1 = np.random.rand(w1_shape[0], w1_shape[1])*2-1
    w2 = np.random.rand(w2_shape[0], w2_shape[1])*2-1
    return {'w1': w1,
            'w2': w2}