"""
Test code for segmentation.
"""

import numpy as np
import segmentation_util as util
import registration_util as reg_util
import matplotlib.pyplot as plt
import segmentation as seg
from scipy import ndimage, stats
import scipy
import scipy.io
from sklearn.neighbors import KNeighborsClassifier
import timeit
from IPython.display import display, clear_output
plt.rcParams['image.cmap'] = 'gray'


# SECTION 1. Segmentation in feature space


def scatter_data_test(showFigs=True):
    I = plt.imread('../data/dataset_brains/1_1_t1.tif')
    X1 = I.flatten().T
    X1 = X1.reshape(-1, 1)
    GT = plt.imread('../data/dataset_brains/1_1_gt.tif')
    gt_mask = GT>0
    Y = gt_mask.flatten() # labels

    I_blurred = ndimage.gaussian_filter(I, sigma=2)
    X2 = I_blurred.flatten().T
    X2 = X2.reshape(-1, 1)
    X_data = np.concatenate((X1, X2), axis=1)

    # Keep track of features you added
    features = ('T1 intensity', 'T1 gauss 2')

    if showFigs:
        util.scatter_data(X_data,Y,0,1)

    #------------------------------------------------------------------#
    # TODO: Implement a few test cases of with different features
    #!studentstart
    I_blurred = ndimage.gaussian_filter(I, sigma=4)
    X3 = I_blurred.flatten().T
    X3 = X3.reshape(-1, 1)
    I_blurred = ndimage.gaussian_filter(I, sigma=6)
    X4 = I_blurred.flatten().T
    X4 = X4.reshape(-1, 1)
    X_data = np.concatenate((X1, X2, X3, X4), axis=1)
    # Keep track of features you added
    features = ('T1 intensity', 'T1 gauss 2', 'T1 gauss 4', 'T1 gauss 6')
    if showFigs:
        util.scatter_data(X_data,Y,0,2)
        util.scatter_data(X_data,Y,0,3)
    #!studentend
    #------------------------------------------------------------------#

    return X_data, Y


def scatter_t2_test(showFigs=True):
    I1 = plt.imread('../data/dataset_brains/1_1_t1.tif')
    X1 = I1.flatten().T
    X1 = X1.reshape(-1, 1)
    I2 = plt.imread('../data/dataset_brains/1_1_t2.tif')
    X2 = I2.flatten().T
    X2 = X2.reshape(-1, 1)

    GT = plt.imread('../data/dataset_brains/1_1_gt.tif')
    gt_mask = GT>0
    Y = gt_mask.flatten() # labels

    I1_blurred = ndimage.gaussian_filter(I1, sigma=4)
    X12 = I1_blurred.flatten().T
    X12 = X12.reshape(-1, 1)
    X_data = np.concatenate((X1, X12), axis=1)

    # Keep track of features you added
    features = ('T1 intensity', 'T1 gauss 2')

    if showFigs:
        util.scatter_data(X_data,Y,0,1)

    #------------------------------------------------------------------#
    # TODO: Extract features from the T2 image and compare them to the T1 features
    #!studentstart
    I2_blurred = ndimage.gaussian_filter(I2, sigma=4)
    X22 = I2_blurred.flatten().T
    X22 = X22.reshape(-1, 1)
    X_data = np.concatenate((X1, X2, X12, X22), axis=1)

    # Keep track of features you added
    features = ('T1 intensity', 'T1 gauss 2', 'T2 intensity', 'T2 gauss 2')

    if showFigs:
        util.scatter_data(X_data,Y,0,2)
        util.scatter_data(X_data,Y,1,3)
    #!studentend
    #------------------------------------------------------------------#

    return X_data, Y


def extract_coordinate_feature_test():
    I = plt.imread('../data/dataset_brains/1_1_t1.tif')
    c, coord_im = seg.extract_coordinate_feature(I)
    fig = plt.figure(figsize=(10,10))
    ax1  = fig.add_subplot(121)
    ax1.imshow(I)
    ax2  = fig.add_subplot(122)
    ax2.imshow(coord_im)


def feature_stats_test():
    X, Y = scatter_data_test(showFigs=False)
    I = plt.imread('../data/dataset_brains/1_1_t1.tif')
    c, coord_im = seg.extract_coordinate_feature(I)
    X_data = np.concatenate((X, c), axis=1)

    #------------------------------------------------------------------#
    # TODO: Write code to examine the mean and standard deviation of your dataset containing variety of features
    #!studentstart
    m = np.mean(X_data, axis=0)
    s = np.std(X_data, axis=0)
    print('X_data            : [T1 intensity, T1 gauss 2, T1 gauss 4, T1 gauss 6, Coordinate]')
    print('Mean              : {}'.format(m))
    print('Standard deviation: {}'.format(s))
    #!studentend
    #------------------------------------------------------------------#


def normalized_stats_test():
    X, Y = scatter_data_test(showFigs=False)
    I = plt.imread('../data/dataset_brains/1_1_t1.tif')
    c, coord_im = seg.extract_coordinate_feature(I)
    X_data = np.concatenate((X, c), axis=1)

    #------------------------------------------------------------------#
    # TODO: Write code to normalize your dataset containing variety of features,
    #  then examine the mean and std dev
    #!studentstart
    X_data_norm, test_data = seg.normalize_data(X_data)
    m = np.mean(X_data_norm, axis=0)
    s = np.std(X_data_norm, axis=0)
    print('X_data            : [T1 intensity, T1 gauss 2, T1 gauss 4, T1 gauss 6, Coordinate]')
    print('Mean              : {}'.format(m))
    print('Standard deviation: {}'.format(s))
    #!studentend
    #------------------------------------------------------------------#


def distance_test():
    #------------------------------------------------------------------#
    # TODO: Generate a Gaussian dataset, with 100 samples per class, and compute the distances.
    #  Use plt.imshow() to visualize the distance matrix as an image.
    #!studentstart
    X, Y = seg.generate_gaussian_data(100) # Generates 100 samples per Gaussian class
    ax = util.scatter_data(X,Y)
    D = scipy.spatial.distance.cdist(X, X, metric='euclidean')
    fig = plt.figure(figsize=(8,8))
    ax1  = fig.add_subplot(111)
    im1  = ax1.imshow(D)
    fig.colorbar(im1)
    #!studentend
    #------------------------------------------------------------------#


def small_samples_distance_test():
    #------------------------------------------------------------------#
    # TODO: Generate a small sample Gaussian dataset X,
    #  create dataset C as per the instructions,
    #  and calculate and plot the distances between the datasets.
    #!studentstart
    N = 3
    X, Y = seg.generate_gaussian_data(N) # Generates N samples per Gaussian class
    X = np.round(X*3) # Stretch and round the numbers
    C = np.array([[0,0],[1,1]])

    D = scipy.spatial.distance.cdist(X, C, metric='euclidean')
    D2 = scipy.spatial.distance.cdist(C, X, metric='euclidean')

    fig = plt.figure(figsize=(8,8))
    ax1  = fig.add_subplot(121)
    im1  = ax1.imshow(D)
    ax2  = fig.add_subplot(122)
    im2  = ax2.imshow(D2)
    fig.colorbar(im1)

    return X, Y, C, D
    #!studentend
    #------------------------------------------------------------------#


def minimum_distance_test(X, Y, C, D):
    #------------------------------------------------------------------#
    # TODO: plot the datasets on top of each other in different colors and visualize the data,
    #  calculate the distances between the datasets,
    #  order the distances (min to max) using the provided code,
    #  calculate how many samples are closest to each of the samples in `C`
    #!studentstart
    ax = util.scatter_data(X,Y)
    im2  = ax.scatter(C[:,0], C[:,1], color='g', label='C')
    plt.legend()
    # D = distance matrix of size: 2N x len(C)
    print('Distance matrix D:')
    print(D)
    # For each row/sample in D, which column has the minimum value...
    # (i.e. to which point in C is this sample the closest)
    min_index = np.argmin(D, axis=1)
    print('For each row in D, which column has the minimum value:')
    print(min_index)
    # Show these minimum distances in a vector
    min_dist = np.empty(*min_index.shape)
    for i in np.arange(len(D)):
        min_dist[i] = D[i, min_index[i]]
    print('Minimum distances:')
    print(min_dist)
    a = min_dist.dot(min_dist.T)
    # Calculate how many samples in X are closest to each of the samples in C
    b = np.empty(len(C))
    for j in np.arange(len(C)):
        b[j] = len(np.where(min_index==j)[0])
    print('Samples in X that are closest to each of the samples in C:')
    print(b)
    #!studentend
    #------------------------------------------------------------------#


def distance_classification_test():
    #------------------------------------------------------------------#
    # TODO: Use the provided code to generate training and testing data
    #  Classify the points in test_data, based on their distances d to the points in train_data
    #!studentstart
    train_data, train_labels = seg.generate_gaussian_data(2)
    test_data, test_labels = seg.generate_gaussian_data(1)
    d = scipy.spatial.distance.cdist(test_data, train_data, metric='euclidean')
    print('Train data:\n{}'.format(train_data))
    print('Test data:\n{}'.format(test_data))
    print('Distance matrix:\n{}'.format(d))
    #!studentend
    #------------------------------------------------------------------#


def funX(X):
    return lambda w: seg.cost_kmeans(X,w)


def kmeans_demo():
    ## Define some data and parameters
    n = 100
    X1 = np.random.randn(n, 2)
    X2 = np.random.randn(n, 2)+5
    X = np.concatenate((X1, X2), axis=0)
    Y = np.concatenate((np.zeros((n,1)), np.ones((n,1))), axis=0)
    N, M = X.shape

    #Define number of clusters we want
    clusters = 2

    # the learning rate
    mu = 1

    # iterations
    num_iter = 100

    # Cost function used by k-Means
    fun = funX(X)

    ## Algorithm
    # Initialize cluster centers
    idx = np.random.randint(N, size=clusters)
    initial_w = X[idx,:]
    w_draw = initial_w
    print(w_draw)

    # Reshape into vector (needed by ngradient)
    w_vector = initial_w.reshape(clusters*M, 1)

    # Vector to store cost
    xx = np.linspace(1, num_iter, num_iter)
    kmeans_cost = np.empty(*xx.shape)
    kmeans_cost[:] = np.nan

    # Create figure
    fig = plt.figure(figsize=(14,6))
    ax1  = fig.add_subplot(121)
    im1  = ax1.scatter(X[:n,0], X[:n,1], label='X-class0')
    im2  = ax1.scatter(X[n:,0], X[n:,1], label='X-class1')
    line1, = ax1.plot(w_draw[:,0], w_draw[:,1], "or", markersize=5, label='W-vector')
    ax1.grid()
    ax2  = fig.add_subplot(122, xlim=(0, num_iter), ylim=(0, 10))
    text_str = 'k={}, g={:.2f}\ncost={:.2f}'.format(0, 0, 0)
    txt2 = ax2.text(0.3, 0.95, text_str, bbox={'facecolor': 'green', 'alpha': 0.4, 'pad': 10},
             transform=ax2.transAxes)
    line2, = ax2.plot(xx, kmeans_cost, lw=2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Cost')
    ax2.grid()

    for k in np.arange(num_iter):
        # Gradient ascent
        g = util.ngradient(fun,w_vector)
        w_vector = w_vector - mu*g
        # Calculate cost for plotting
        kmeans_cost[k] = fun(w_vector)
        text_str = 'k={}, cost={:.2f}'.format(k, kmeans_cost[k])
        txt2.set_text(text_str)
        # Plot
        line2.set_ydata(kmeans_cost)
        w_draw_new = w_vector.reshape(clusters, M)
        line1.set_data(w_draw_new[:,0], w_draw_new[:,1])
        display(fig)
        clear_output(wait = True)
        plt.pause(.005)

    return kmeans_cost


def kmeans_clustering_test():
    #------------------------------------------------------------------#
    #TODO: Store errors for training data
    #!studentstart

    #Subject 3, slice 1 is the test data = normalize data(test data);
    test_data, test_labels, feature_labels = util.create_dataset(3,1,'brain')
    test_data_norm  = seg.normalize_data(test_data)[0]
    predicted_labels = seg.kmeans_clustering(test_data_norm,2)
    predicted_labels = predicted_labels.astype(bool)
    dice = util.dice_overlap(test_labels,predicted_labels)
    print('Dice coefficient:\n{}'.format(dice))
    I = plt.imread('../data/dataset_brains/3_1_t1.tif')

    predicted_mask = predicted_labels.reshape(I.shape)

    fig = plt.figure(figsize=(14,7))
    ax1 = fig.add_subplot(121)
    ax1.imshow(I)
    ax2 = fig.add_subplot(122)
    ax2.imshow(predicted_mask)
    #!studentend
    #------------------------------------------------------------------#


def nn_classifier_test_samples():
    train_data, train_labels = seg.generate_gaussian_data(2)
    test_data, test_labels = seg.generate_gaussian_data(1)
    predicted_labels = seg.nn_classifier(train_data, train_labels, test_data)

    err = util.classification_error(test_labels, predicted_labels)

    print('True labels:\n{}'.format(test_labels))
    print('Predicted labels:\n{}'.format(predicted_labels))
    print('Error:\n{}'.format(err))


def generate_train_test(N, task):
    # Generates a training and a test set with the same
    # data distribution from two possibilities: easy dataset with low class
    # overlap, or hard dataset with high class overlap
    # Input:
    # N - Number of samples per classs
    # task - String, either 'easy' or 'hard'

    if task == 'easy':
        #-------------------------------------------------------------------#
        #TODO: modify these values to create an easy train/test dataset pair
        #!studentstart
        mu1 = [0,0]
        mu2 = [0,0]
        sigma1 = [[1,0],[0,1]]
        sigma2 = [[1,0],[0,1]]
        #!studentend
        #-------------------------------------------------------------------#

    if task == 'hard':
        #-------------------------------------------------------------------#
        #TODO: modify these values to create an difficult train/test dataset pair
        #!studentstart
        mu1 = [0,0]
        mu2 = [0,0]
        sigma1 = [[1,0],[0,1]]
        sigma2 = [[1,0],[0,1]]
        #!studentend
        #-------------------------------------------------------------------#

    trainX, trainY = seg.generate_gaussian_data(N, mu1, mu2, sigma1, sigma2)
    testX, testY = seg.generate_gaussian_data(N, mu1, mu2, sigma1, sigma2)

    return trainX, trainY, testX, testY


def easy_hard_data_classifier_test():
    #-------------------------------------------------------------------#
    #TODO: generate and classify (using nn_classifier) 2 pairs of datasets (easy and hard)
    # calculate classification error in each case
    #!studentstart
    N = 10

    task = 'easy'
    trainX, trainY, testX, testY = generate_train_test(N, task)
    predicted_labels = seg.nn_classifier(trainX, trainY, testX)
    err = util.classification_error(testY, predicted_labels)
    print('Easy dataset:')
    print('Error:\n{}'.format(err))

    task = 'hard'
    trainX, trainY, testX, testY = generate_train_test(N, task)
    predicted_labels = seg.nn_classifier(trainX, trainY, testX)
    err = util.classification_error(testY, predicted_labels)
    print('Hard dataset:')
    print('Error:\n{}'.format(err))
    #!studentend
    #-------------------------------------------------------------------#


def nn_classifier_test_brains(testDice=False):
    # Subject 1, slice 1 is the train data
    X, Y, feature_labels_train = util.create_dataset(1,1,'brain')
    N = 1000
    ix = np.random.randint(len(X), size=N)
    train_data = X[ix,:]
    train_labels = Y[ix,:]

    # Subject 3, slice 1 is the test data
    test_data, test_labels, feature_labels_test  = util.create_dataset(3,1,'brain')

    predicted_labels = seg.nn_classifier(train_data, train_labels, test_data)
    predicted_labels = predicted_labels.astype(bool)
    test_labels = test_labels.astype(bool)
    err = util.classification_error(test_labels, predicted_labels)
    print('Error:\n{}'.format(err))

    if testDice:
        dice = util.dice_overlap(test_labels, predicted_labels)
        print('Dice coefficient:\n{}'.format(dice))
    else:
        I = plt.imread('../data/dataset_brains/3_1_t1.tif')
        GT = plt.imread('../data/dataset_brains/3_1_gt.tif')
        gt_mask = GT>0
        gt_labels = gt_mask.flatten() # labels
        predicted_mask = predicted_labels.reshape(I.shape)
        fig = plt.figure(figsize=(15,5))
        ax1 = fig.add_subplot(131)
        ax1.imshow(I)
        ax2 = fig.add_subplot(132)
        ax2.imshow(predicted_mask)
        ax3  = fig.add_subplot(133)
        ax3.imshow(gt_mask)


def knn_curve():
    # Load training and test data
    train_data, train_labels, train_feature_labels = util.create_dataset(1,1,'brain')
    test_data, test_labels, test_feature_labels = util.create_dataset(2,1,'brain')

    # Normalize data
    train_data, test_data = seg.normalize_data(train_data, test_data)

    # Define parameters
    num_iter = 3
    train_size = 100
    k = np.array([1, 3, 5, 9, 15, 25, 100])

    # Store errors
    test_error = np.empty([len(k),num_iter])
    test_error[:] = np.nan
    dice = np.empty([len(k),num_iter])
    dice[:] = np.nan

    ## Train and test with different values
    for i in np.arange(len(k)):
        for j in np.arange(num_iter):
            print('k = {}, iter = {}'.format(k[i], j))
            #Subsample training set
            ix = np.random.randint(len(train_data), size=train_size)
            subset_train_data = train_data[ix,:]
            subset_train_labels = train_labels[ix,:]

            # Train and evaluate classifier
            predicted_test_labels = seg.knn_classifier(subset_train_data, subset_train_labels, test_data, k[i])

            # #Train classifier
            # neigh = KNeighborsClassifier(n_neighbors=k[i])
            # neigh.fit(subset_train_data, subset_train_labels)
            # #Evaluate
            # predicted_test_labels = neigh.predict(test_data)

            test_error[i,j] = util.classification_error(test_labels, predicted_test_labels)
            dice[i,j] = util.dice_overlap(test_labels, predicted_test_labels)

    ## Display results
    fig = plt.figure(figsize=(8,8))
    ax1 = fig.add_subplot(111)
    p1 = ax1.plot(k, np.mean(test_error,1), 'r', label='error')
    p2 = ax1.plot(k, np.mean(dice,1), 'k', label='dice')
    ax1.set_xlabel('k')
    ax1.set_ylabel('error')
    ax1.grid()
    ax1.legend()


# SECTION 2. Generalization and overfitting


def learning_curve():
    # Load training and test data
    train_data, train_labels = seg.generate_gaussian_data(1000)
    test_data, test_labels = seg.generate_gaussian_data(1000)
    [train_data, test_data] = seg.normalize_data(train_data, test_data)

    # Define parameters
    train_sizes = np.array([1, 3, 10, 30, 100, 300])
    k = 1

    # How often to repeat the experiment
    num_iter = 3

    # Store errors
    test_error = np.empty([len(train_sizes),num_iter])
    test_error[:] = np.nan
    test_dice = np.empty([len(train_sizes),num_iter])
    test_dice[:] = np.nan

    #------------------------------------------------------------------#
    #TODO: Store errors for training data
    #!studentstart
    train_error = np.empty([len(train_sizes),num_iter])
    train_error[:] = np.nan
    train_dice = np.empty([len(train_sizes),num_iter])
    train_dice[:] = np.nan
    #!studentend
    #------------------------------------------------------------------#

    ## Train and test with different values
    for i in np.arange(len(train_sizes)):
        for j in np.arange(num_iter):
            print('train_size = {}, iter = {}'.format(train_sizes[i], j))
            #Subsample training set
            ix = np.random.randint(len(train_data), size=train_sizes[i])
            subset_train_data = train_data[ix,:]
            subset_train_labels = train_labels[ix,:]

            #Train classifier
            neigh = KNeighborsClassifier(n_neighbors=k)
            neigh.fit(subset_train_data, subset_train_labels.ravel())
            #Evaluate
            predicted_test_labels = neigh.predict(test_data)

            test_labels = test_labels.astype(bool)
            predicted_test_labels = predicted_test_labels.astype(bool)

            test_error[i,j] = util.classification_error(test_labels, predicted_test_labels)
            test_dice[i,j] = util.dice_overlap(test_labels, predicted_test_labels)

            #------------------------------------------------------------------#
            #TODO: Predict training labels and evaluate
            #!studentstart
            predicted_train_labels = neigh.predict(subset_train_data)
            subset_train_labels = subset_train_labels.astype(bool)
            predicted_train_labels = predicted_train_labels.astype(bool)
            train_error[i,j] = util.classification_error(subset_train_labels, predicted_train_labels)
            train_dice[i,j] = util.dice_overlap(subset_train_labels, predicted_train_labels)
            #!studentend
            #------------------------------------------------------------------#

    # Display results
    fig = plt.figure(figsize=(8,8))
    ax1 = fig.add_subplot(111)
    x = np.log(train_sizes)
    y_test = np.mean(test_error,1)
    yerr_test = np.std(test_error,1)
    p1 = ax1.errorbar(x, y_test, yerr=yerr_test, label='Test error')

    #------------------------------------------------------------------#
    #TODO: Plot training size
    #!studentstart
    y_train = np.mean(train_error,1)
    yerr_train = np.std(train_error,1)
    p2 = ax1.errorbar(x, y_train, yerr=yerr_train, label='Train error')
    #!studentend
    #------------------------------------------------------------------#

    ax1.set_xlabel('Number of training samples (k)')
    ax1.set_ylabel('error')
    ticks = list(x)
    ax1.set_xticks(ticks)
    tick_lbls = [str(i) for i in train_sizes]
    ax1.set_xticklabels(tick_lbls)
    ax1.grid()
    ax1.legend()


def feature_curve(use_random=False):
    # Load training and test data
    train_data, train_labels, train_feature_labels = util.create_dataset(1,1,'brain')
    test_data, test_labels, test_feature_labels = util.create_dataset(2,1,'brain')

    if use_random:
        #------------------------------------------------------------------#
        #TODO: Replace features by random numbers, of the same size as the data
        #!studentstart
        train_data = np.random.randn(*train_data.shape)
        test_data = np.random.randn(*test_data.shape)
        #!studentend
        #------------------------------------------------------------------#

    # Normalize data
    train_data, test_data = seg.normalize_data(train_data, test_data)

    # Define parameters
    feature_sizes = np.arange(train_data.shape[1])+1
    train_size = 10
    k = 3
    num_iter = 5

    # Store errors
    test_error = np.empty([len(feature_sizes),num_iter])
    test_error[:] = np.nan
    train_error = np.empty([len(feature_sizes),num_iter])
    train_error[:] = np.nan

    # Train and test with different sizes
    for i in np.arange(len(feature_sizes)):
        for j in np.arange(num_iter):
            print('feature size = {}, iter = {}'.format(feature_sizes[i], j))
            start_time = timeit.default_timer()

            #Subsample training set
            ix = np.random.randint(len(train_data), size=train_size)
            subset_train_data = train_data[ix,:]
            subset_train_labels = train_labels[ix,:]

            #Train classifier
            neigh = KNeighborsClassifier(n_neighbors=k)
            neigh.fit(subset_train_data[:,:feature_sizes[i]], subset_train_labels.ravel())

            #Evaluate
            predicted_test_labels = neigh.predict(test_data[:,:feature_sizes[i]])
            predicted_train_labels = neigh.predict(subset_train_data[:,:feature_sizes[i]])

            test_error[i,j] = util.classification_error(test_labels, predicted_test_labels)
            train_error[i,j] = util.classification_error(subset_train_labels, predicted_train_labels)

            # Timer log
            elapsed = timeit.default_timer() - start_time

    # Display results
    fig = plt.figure(figsize=(8,8))
    ax1 = fig.add_subplot(111)
    x = feature_sizes
    y_test = np.mean(test_error,1)
    yerr_test = np.std(test_error,1)
    p1 = ax1.errorbar(x, y_test, yerr=yerr_test, label='Test error')

    #------------------------------------------------------------------#
    #TODO: Plot training size
    #!studentstart
    y_train = np.mean(train_error,1)
    yerr_train = np.std(train_error,1)
    p2 = ax1.errorbar(x, y_train, yerr=yerr_train, label='Train error')
    #!studentend
    #------------------------------------------------------------------#

    ax1.set_xlabel('Number of features')
    ax1.set_ylabel('Error')
    ax1.grid()
    ax1.legend()


def high_dimensions_demo():
    # Base figure
    fig = plt.figure(figsize=(15,10))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)

    # Generate data from 2D Gaussian distribution
    X1 = np.random.randn(100,2)
    mn1, mx1, mns1 = high_dimensions_output(X1, ax1, 20)
    print('2D Gaussian distribution')
    print('Mean = {}, Max = {}, Mean nn = {}'.format(mn1, mx1, mns1))

    # Generate data from 1000D Gaussian distribution
    X2 = np.random.randn(100,1000)
    mn2, mx2, mns2 = high_dimensions_output(X2, ax2, 20)
    print('1000D Gaussian distribution')
    print('Mean = {}, Max = {}, Mean nn = {}'.format(mn2, mx2, mns2))

    # More surprising properties of high dimensions
    n = 10
    k = 0.01
    frac = np.empty(n)
    frac[:] = None
    for i in np.arange(n):
        frac[i] = k**(1/(i+1))

    ax3 = fig.add_subplot(133)
    ax3.plot(frac)
    ax3.set_xlabel('dimensions')
    ax3.set_ylabel('fraction to travel per dimension')


def high_dimensions_output(X, ax, n_bins=20):
    # Calculate distances
    D = scipy.spatial.distance.cdist(X, X, metric='euclidean')

    # Plot histogram
    ax.hist(D.flatten(), bins=n_bins)

    #------------------------------------------------------------------#
    # TODO: Mean, maximum distance, and average nearest neighbor distance
    #!studentstart
    mn = np.mean(D.flatten())
    mx = np.max(D.flatten())
    sorted_dist = np.argsort(D, axis=1)
    # The first column will contain the zeros - distances to the same point,
    # so we take the second column as the nearest neighbor
    mns = np.mean(sorted_dist[:,2].flatten())
    #!studentend
    #------------------------------------------------------------------#

    return mn, mx, mns


def covariance_matrix_test():
    N=100
    mu1=[0,0]
    mu2=[0,0]
    sigma1=[[3,1],[1,1]]
    sigma2=[[3,1],[1,1]]
    X, Y = seg.generate_gaussian_data(N, mu1, mu2, sigma1, sigma2)

    #------------------------------------------------------------------#
    # TODO: Calculate the mean and covariance matrix of the data,
    #  and compare them to the parameters you used as input.
    #!studentstart
    mn = np.mean(X, axis=0)
    print('Mean:\n{}'.format(mn))
    sigma = np.cov(X, rowvar=False)
    print('Covariance matrix:\n{}'.format(sigma))
    return X, Y, sigma
    #!studentend
    #------------------------------------------------------------------#


def eigen_vecval_test(sigma):
    #------------------------------------------------------------------#
    # TODO: Compute the eigenvectors and eigenvalues of the covariance matrix,
    #  what two properties can you name about the eigenvectors? How can you verify these properties?
    #  which eigenvalue is the largest and which is the smallest?
    #!studentstart
    w, v = np.linalg.eig(sigma)
    print('Eigenvalues w:\n{}'.format(w))
    print('Eigenvectors v:\n{}'.format(v))

    # Find ordering of eigenvalues
    ix = np.argsort(w)[::-1]
    # Reorder eigenvalues
    w = w[ix]
    # Reorder eigenvectors
    v = v[:, ix]
    print('Order v:\n{}'.format(ix))
    print('Reordered eigenvalues v:\n{}'.format(w))
    print('Reordered eigenvectors v:\n{}'.format(v))
    print('Checking results:')
    print('Dot procuct of first eigenvector with itself:\n{}'.format(v[:,0].dot(v[:,0])))
    print('Dot procuct of the two eigenvectors:\n{}'.format(v[:,0].dot(v[:,1])))
    return v, w
    #!studentend
    #------------------------------------------------------------------#


def rotate_using_eigenvectors_test(X, Y, v):
    #------------------------------------------------------------------#
    # TODO: Rotate X using the eigenvectors
    #!studentstart
    X_rotated = v.T.dot(X.T)
    X_rotated = X_rotated.T
    fig = plt.figure(figsize=(15,6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    util.scatter_data(X,Y,ax=ax1)
    util.scatter_data(X_rotated,Y,ax=ax2)
    return X_rotated
    #!studentend
    #------------------------------------------------------------------#


def test_mypca():
    #Generates some toy data in 2D, computes PCA, and plots both datasets
    N=100
    mu1=[0,0]
    mu2=[2,0]
    sigma1=[[2,1],[1,1]]
    sigma2=[[2,1],[1,1]]

    XG, YG = seg.generate_gaussian_data(N, mu1, mu2, sigma1, sigma2)

    fig = plt.figure(figsize=(15,6))

    ax1 = fig.add_subplot(121)
    util.scatter_data(XG,YG,ax=ax1)
    sigma = np.cov(XG, rowvar=False)
    w, v = np.linalg.eig(sigma)
    ax1.plot([0, v[0,0]], [0, v[1,0]], c='g', linewidth=3, label='Eigenvector1')
    ax1.plot([0, v[0,1]], [0, v[1,1]], c='k', linewidth=3, label='Eigenvector2')
    ax1.set_title('Original data')
    ax_settings(ax1)

    ax2 = fig.add_subplot(122)
    X_pca, v, w, fraction_variance = seg.mypca(XG)
    util.scatter_data(X_pca,YG,ax=ax2)
    sigma2 = np.cov(X_pca, rowvar=False)
    w2, v2 = np.linalg.eig(sigma2)
    ax2.plot([0, v2[0,0]], [0, v2[1,0]], c='g', linewidth=3, label='Eigenvector1')
    ax2.plot([0, v2[0,1]], [0, v2[1,1]], c='k', linewidth=3, label='Eigenvector2')
    ax2.set_title('My PCA')
    ax_settings(ax2)

    handles, labels = ax2.get_legend_handles_labels()
    plt.figlegend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), bbox_transform=plt.gcf().transFigure, ncol = 4)

    print(fraction_variance)


def ax_settings(ax):
    ax.set_xlim(-7,7)
    ax.set_ylim(-7,7)
    ax.set_aspect('equal', adjustable='box')
    ax.grid()


# SECTION 3. Atlases and active shapes


def segmentation_combined_atlas_test():
    task = 'brain'
    n = 5
    all_subjects = np.arange(n)
    train_slice = 1
    tmp_data, tmp_labels, tmp_feature_labels = util.create_dataset(1,train_slice,task)
    all_data_matrix = np.empty((tmp_data.shape[0], tmp_data.shape[1], n))
    all_labels_matrix = np.empty((tmp_labels.shape[0], n))

    #Load datasets once
    for i in all_subjects:
        train_data, train_labels, train_feature_labels = util.create_dataset(i+1,train_slice,task)
        all_data_matrix[:,:,i] = train_data
        all_labels_matrix[:,i] = train_labels.ravel()

    #------------------------------------------------------------------#
    # TODO: Use provided code to Combine labels of training images,
    #  Convert combined label into mask image,
    #  Convert true label into mask image, and
    #  View both masks on the same axis,
    #  Also calculate dice coefficient and error
    #!studentstart

    # Combine labels of training images:
    predicted_labels = stats.mode(all_labels_matrix[:,:4], axis=1)[0]
    # Convert combined label into mask image:
    predicted_mask = predicted_labels.reshape(240,240)
    # Convert true label into mask image:
    true_mask = all_labels_matrix[:,4].reshape(240,240)

    # View both masks:
    fig = plt.figure(figsize=(8,8))
    ax1 = fig.add_subplot(111)
    ax1.imshow(true_mask, 'gray')
    ax1.imshow(predicted_mask, 'viridis', alpha=0.5)

    test_labels = all_labels_matrix[:,4].astype(bool)
    predicted_labels = predicted_labels.astype(bool)

    err = util.classification_error(test_labels, predicted_labels)
    print('Error:\n{}'.format(err))

    dice = util.dice_overlap(test_labels, predicted_labels)
    print('Dice coefficient:\n{}'.format(dice))
    #!studentend
    #------------------------------------------------------------------#


def segmentation_combined_atlas_minmax_test():
    task = 'brain'
    n = 5
    all_subjects = np.arange(n)
    train_slice = 1
    tmp_data, tmp_labels, tmp_feature_labels = util.create_dataset(1,train_slice,task)
    all_data_matrix = np.empty((tmp_data.shape[0], tmp_data.shape[1], n))
    all_labels_matrix = np.empty((tmp_labels.shape[0], n))

    #Load datasets once
    for i in all_subjects:
        train_data, train_labels, train_feature_labels = util.create_dataset(i+1,train_slice,task)
        all_data_matrix[:,:,i] = train_data
        all_labels_matrix[:,i] = train_labels.ravel()

    predicted_labels_min = seg.segmentation_combined_atlas(all_labels_matrix, combining='min')
    predicted_labels_max = seg.segmentation_combined_atlas(all_labels_matrix, combining='max')

    test_labels = all_labels_matrix[:,4].astype(bool)

    print('Combining method = min:')
    err = util.classification_error(test_labels, predicted_labels_min)
    print('Error:\n{}'.format(err))
    dice = util.dice_overlap(test_labels, predicted_labels_min)
    print('Dice coefficient:\n{}'.format(dice))

    print('Combining method = max:')
    err = util.classification_error(test_labels, predicted_labels_max)
    print('Error:\n{}'.format(err))
    dice = util.dice_overlap(test_labels, predicted_labels_max)
    print('Dice coefficient:\n{}'.format(dice))


def plot_hand_shapes():
    #------------------------------------------------------------------#
    #!studentstart
    fn = '../data/dataset_hands/coordinates.txt'
    coordinates =  np.loadtxt(fn)

    fig = plt.figure(figsize=(16,4))
    n = 0
    lbl = 'hand_' + str(n+1)
    ax1  = fig.add_subplot(141)
    ax1.plot(coordinates[n,:56], coordinates[n,56:], label=lbl)
    ax1.set_title(lbl)
    n = 1
    lbl = 'hand_' + str(n+1)
    ax2  = fig.add_subplot(142)
    ax2.plot(coordinates[n,:56], coordinates[n,56:], label=lbl)
    ax2.set_title(lbl)
    n = 2
    lbl = 'hand_' + str(n+1)
    ax3  = fig.add_subplot(143)
    ax3.plot(coordinates[n,:56], coordinates[n,56:], label=lbl)
    ax3.set_title(lbl)
    n = 3
    lbl = 'hand_' + str(n+1)
    ax4  = fig.add_subplot(144)
    ax4.plot(coordinates[n,:56], coordinates[n,56:], label=lbl)
    ax4.set_title(lbl)

    mn = np.mean(coordinates, axis=0)
    fig2 = plt.figure(figsize=(4,4))
    ax1  = fig2.add_subplot(111)
    ax1.plot(mn[:56], mn[56:])
    ax1.set_title('Mean hand shape')
    #!studentend
    #------------------------------------------------------------------#


def test_mypca_hands():
    #------------------------------------------------------------------#
    #!studentstart
    fn = '../data/dataset_hands/coordinates.txt'
    coordinates =  np.loadtxt(fn)

    X_pca, v, w, fraction_variance = seg.mypca(coordinates)
    num_dims = np.nonzero(fraction_variance >= 0.98)[0][0]+1
    v_new = v[:,:num_dims]
    #!studentend
    #------------------------------------------------------------------#

    return num_dims, v_new


def test_remaining_variance():
    #------------------------------------------------------------------#
    #!studentstart
    fn = '../data/dataset_hands/coordinates.txt'
    coordinates =  np.loadtxt(fn)
    mn = np.mean(coordinates, axis=0)

    X_pca, v, w, fraction_variance = seg.mypca(coordinates)
    num_dims = np.nonzero(fraction_variance >= 0.98)[0][0]+1
    v_new = v[:,:num_dims]

    fig = plt.figure(figsize=(15,10))

    for i in np.arange(num_dims):
        variation = mn +  (v_new[:,i] * w[i]*5).T
        ax  = fig.add_subplot(2,3,i+1)
        ax.plot(mn[:56], mn[56:])
        ax.plot(mn[:56], mn[56:], 'ro')
        ax.plot(variation[:56], variation[56:], 'bo')
        ax.grid()
    #!studentend
    #------------------------------------------------------------------#


def plot_hand_grayscale():
    #------------------------------------------------------------------#
    #!studentstart
    fn = '../data/dataset_hands/test001.jpg'
    img_hand =  plt.imread(fn)
    fn = '../data/dataset_hands/coordinates.txt'
    coordinates =  np.loadtxt(fn)

    fig = plt.figure(figsize=(16,8))
    ax1 = fig.add_subplot(121)
    ax1.imshow(img_hand)

    # L = R * 299/1000 + G * 587/1000 + B * 114/1000
    img2 = img_hand[:,:,0] * 299/1000 + img_hand[:,:,1] * 587/1000 + img_hand[:,:,2] * 114/1000
    ax2 = fig.add_subplot(122)
    ax2.imshow(img2, cmap='gray')

    n = 0
    lbl = 'hand_' + str(n+1)
    ax1.plot(coordinates[n,:56], coordinates[n,56:], 'r', lw=3, label=lbl)
    ax2.plot(coordinates[n,:56]*100, coordinates[n,56:]*100, 'r', lw=3, label=lbl)
    #!studentend
    #------------------------------------------------------------------#


def test_transformed_hand():
    #------------------------------------------------------------------#
    #!studentstart
    fn = '../data/dataset_hands/test001.jpg'
    img_hand =  plt.imread(fn)
    fn = '../data/dataset_hands/coordinates.txt'
    coordinates =  np.loadtxt(fn)
    mn = np.mean(coordinates, axis=0)

    # Initialize position
    # Convert shape to 2D format first (easier to work with for the next steps)
    initialpos = np.concatenate((mn[:56].reshape(1,-1), mn[56:].reshape(1,-1)), axis=0)

    # Define a scaling/rotation/alignment matrix and transform the shape
    # as close as possible to the image
    alignmatrix = np.array([[500, 0, 150], [0, 500, -80]])
    initialpos = alignmatrix.dot(reg_util.c2h(initialpos))

    # Plot image and transformed shape
    fig = plt.figure(figsize=(16,8))
    # L = R * 299/1000 + G * 587/1000 + B * 114/1000
    img2 = img_hand[:,:,0] * 299/1000 + img_hand[:,:,1] * 587/1000 + img_hand[:,:,2] * 114/1000
    ax2 = fig.add_subplot(122)
    ax2.imshow(img2, cmap='gray')
    ax2.plot(initialpos[0,:], initialpos[1,:], 'r')
    #!studentend
    #------------------------------------------------------------------#