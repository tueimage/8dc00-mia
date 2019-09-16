"""
Segmentation module code for 8DC00 course
"""

# Imports

import numpy as np
import scipy
from sklearn.neighbors import KNeighborsClassifier


# SECTION 1. Segmentation in feature space

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


def extract_coordinate_feature(im):
    # Creates a coordinate feature, which encodes how far a pixel is from the center of the image.
    #
    # Input:
    # im -   An NxM image
    # Output:
    # c  -   A (N*M)x1 vector which encodes how far each pixel is from the center of the image

    # Get the image size
    n_rows, n_cols = im.shape   
    
    # Find image center
    x_center = np.floor(n_rows/2);
    y_center = np.floor(n_cols/2);
    
    # Generate coordinate images
    ar = np.arange(n_cols).reshape(1,-1)
    x_coord = np.tile(ar, (n_rows, 1))
    ar = ar.T
    y_coord = np.tile(ar, (1, n_cols))
    
    #------------------------------------------------------------------#
    # TODO: Use the above variables to create an image coord_im
    # that combines the information from x_coord and y_coord 
    #------------------------------------------------------------------#
    
    # Create a feature from the coordinate image
    c = coord_im.flatten().T
    c = c.reshape(-1, 1)

    return c, coord_im


def normalize_data(train_data, test_data=None):
    # Normalizes data train_data (and optionally, test_data), by
    # subtracting the mean of train_data, and dividing by the standard deviation of
    # train_data.
    #
    # Input:
    # train_data            num_train x k dataset with Ntrain samples and k features
    # test_data            (Optional input) num_test x k dataset with Ntest samples and k features 
    # Output:
    # train_data            num_train x k dataset with Ntrain samples and k features, that has been normalized by trainX
    # test_data            (Optional output) num_test x k dataset with Ntest samples and k features, that has been normalized by trainX 
    
    #Find mean and standard deviation of trainX
    mean_train = np.mean(train_data,axis=0)
    std_train = np.std(train_data,axis=0)

    # Subtract mean
    train_data = train_data - mean_train

    # Divide by standard deviation
    train_data = train_data / std_train

    # (Optional) If testX needs to be normalized also - note it is normalized by
    #the mean and variance of trainX, not testX! 
    if test_data is not None:
        test_data = test_data - mean_train
        test_data = test_data / std_train

    return train_data, test_data


def cost_kmeans(X, w_vector):
    # Computes the cost of assigning data in X to clusters in w_vector 
    
    # Get the data dimensions
    n, m = X.shape

    # Number of clusters
    K = int(len(w_vector)/m)

    # Reshape cluster centers into dataset format
    W = w_vector.reshape(K, m)

    #------------------------------------------------------------------#
    # TODO: Find distance of each point to each cluster center
    # Then find the minimum distances min_dist and indices min_index
    # Then calculate the cost
    #------------------------------------------------------------------#
    return J


def kmeans_clustering(test_data, K=2):
    # Returns the labels for test_data, predicted by the kMeans
    # classifier which assumes that clusters are ordered by intensity
    #
    # Input:
    # test_data          num_test x p matrix with features for the test data
    # k                  Number of clusters to take into account (2 by default)
    # Output:
    # predicted_labels    num_test x 1 predicted vector with labels for the test data

    # Link to the cost function of kMeans
    fun = lambda w: cost_kmeans(test_data, w)


    # the learning rate
    mu = 0.01

    # iterations
    num_iter = 100

    #------------------------------------------------------------------#
    # TODO: Initialize cluster centers and store them in w_initial
    #------------------------------------------------------------------#

    #Reshape centers to a vector (needed by ngradient)
    w_vector = w_initial.reshape(K*M, 1)

    for i in np.arange(num_iter):
        # gradient ascent
        w_vector = w_vector - mu*reg.ngradient(fun,w_vector)

    #Reshape back to dataset
    w_final = w_vector.reshape(K, M)

    #------------------------------------------------------------------#
    # TODO: Find distance of each point to each cluster center
    # Then find the minimum distances min_dist and indices min_index
    #------------------------------------------------------------------#

    # Sort by intensity of cluster center
    sorted_order = np.argsort(w_final[:,0], axis=0)

    # Update the cluster indices based on the sorted order and return results in
    # predicted_labels
    predicted_labels = np.empty(*min_index.shape)
    predicted_labels[:] = np.nan

    for i in np.arange(len(sorted_order)):
        predicted_labels[min_index==sorted_order[i]] = i

    return predicted_labels


def nn_classifier(train_data, train_labels, test_data):
    # Returns the labels for test_data, predicted by the 1-NN
    # classifier trained on train_data and train_labels
    #
    # Input:
    # train_data        num_train x p matrix with features for the training data
    # train_labels      num_train x 1 vector with labels for the training data
    # test_labels       num_test x p matrix with features for the test data
    #
    # Output:
    # predicted_labels   num_test x 1 predicted vector with labels for the test data

    #------------------------------------------------------------------#
    # TODO: Implement missing functionality
    #------------------------------------------------------------------#

    #Return fraction of variance
    fraction_variance = np.zeros((X_pca.shape[1],1))
    for i in np.arange(X_pca.shape[1]):
        fraction_variance[i] = np.sum(w[:i+1])/np.sum(w)

    return X_pca, v, w, fraction_variance


# SECTION 3. Atlases and active shapes

def segmentation_combined_atlas(train_labels_matrix, combining='mode'):
    # Segments the image defined based only on the labels/atlases of the other subjects
    #
    # Input:
    # train_labels   num_train x num_atlases training labels vector
    # combining      String corresponding to combining type: 'mode', 'min'
    # (only binary labels), 'max' (only binary labels)
    #
    # Output:
    # predicted_labels    Predicted labels for the test slice

    r, c = train_labels_matrix.shape

    # Segment the test subject by each individual atlas
    predicted_labels = np.empty([r,c])
    predicted_labels[:] = np.nan

    for i in np.arange(c):
        predicted_labels[:,i] = segmentation_atlas(None, train_labels_matrix[:,i], None)

    # Combine labels
    # Option 1: Most frequent label
    if combining == 'mode':
        predicted_labels = scipy.stats.mode(predicted_labels, axis=1)[0]
    
    #------------------------------------------------------------------#
    # TODO: Add options for combining with min and max
    #------------------------------------------------------------------#
    else:
        raise ValueError("No such combining type exists")

    return predicted_labels.astype(bool)


def segmentation_atlas(train_data, train_labels, test_data):

    # Segments the image defined by test_subject and test_slice,
    # based only on the labels/atlases of the other subjects
    #
    # Input:
    # train_labels   num_train x 1 training labels vector
    # Output:
    # predicted_labels    Predicted labels for the test slice

    # Note that train_data and test_data are not used here because we assume the
    # images are registered. But in practice, we would want to first do
    # registration on the image intensity

    #Assume predicted labels are the atlas labels
    predicted_labels = train_labels

    return predicted_labels


def segmentation_combined_knn(train_data_matrix, train_labels_matrix, test_data, k=1):

    # Segments the image defined by test_data based on
    # kNN classifiers trained on data in train_data_matrix and
    # train_labels_matrix
    #
    # Input:
    # train_data_matrix   num_pixels x num_features x num_subjects matrix of
    # features
    # train_labels_matrix num_pixels x num_subjects matrix of labels
    # test_data           num_pixels x num_features test data
    # k                   Number of neighbors
    #
    # Output:
    # predicted_labels    Predicted labels for the test slice

    r, c = train_labels_matrix.shape

    predicted_labels = np.empty([r,c])
    predicted_labels[:] = np.nan

    for i in np.arange(c):
        predicted_labels[:,i] = segmentation_knn(train_data_matrix[:,:,i], train_labels_matrix[:,i], test_data, k)

    #Combine labels
    predicted_labels = scipy.stats.mode(predicted_labels, axis=1)[0]

    return predicted_labels.astype(bool)


def segmentation_knn(train_data, train_labels, test_data, k=1):

    # Segments the image using a knn classsifier trained on
    # train_data and train_labels
    #
    # Input:
    # train_data     num_train x num_features training data matrix
    # train_labels   num_train x 1 training labels vector
    # test_data      num_test x num_features test data matrix
    # k              Number of neighbors
    #
    # Output:
    # predicted_labels    Predicted labels for the test slice

    
    # Subsample training data for efficiency
    num_samples=3000
    ix = np.random.randint(train_data.shape[0], size=num_samples)

    subset_train_data = train_data[ix,:]
    subset_train_labels = train_labels[ix]


    #Normalize
    [train_data_norm, test_data_norm] = normalize_data(subset_train_data, test_data);

    #Train and apply kNN classifier

    # Option 1: The implementation we made in this course (slower)
    # predicted_labels = knn_classifier(train_data_norm, subset_train_labels, test_data_norm, k)

    # Option 2: The implementation of sklearn (faster)
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(train_data_norm, subset_train_labels)
    predicted_labels = neigh.predict(test_data_norm)

    return predicted_labels
