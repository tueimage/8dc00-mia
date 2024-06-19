"""
Code for notebook OPTIONAL_active-shape-models
"""

import numpy as np
import cad
import registration_util as reg_util
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
plt.rcParams['image.cmap'] = 'gray'


def plot_hand_shapes():
    # Load the hand dataset (40 hand shapes)
    fn = '../data/dataset_hands/coordinates.txt'
    coordinates =  np.loadtxt(fn)

    # Plotting a few shapes to see the variations
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
    
    #------------------------------------------------------------------#
    # TODO: Calculate the mean hand shape and plot this in a new figure.
    #!studentstart
    mn = np.mean(coordinates, axis=0)    
    fig2 = plt.figure(figsize=(4,4))
    ax1  = fig2.add_subplot(111)
    ax1.plot(mn[:56], mn[56:])
    ax1.set_title('Mean hand shape')
    #!studentend
    #------------------------------------------------------------------#

def pca_hands():
    fn = '../data/dataset_hands/coordinates.txt'
    coordinates =  np.loadtxt(fn)
    #------------------------------------------------------------------#
    # TODO: Apply PCA to the coordinates data.
    #!studentstart
    X_pca, v, w, fraction_variance = cad.mypca(coordinates)
    num_dims = np.nonzero(fraction_variance >= 0.98)[0][0]+1
    v_new = v[:,:num_dims]
    #!studentend
    #------------------------------------------------------------------#
    # Note: this function also needs to return the eigenvectors v and the 
    # eigenvalues w (you will need these in the next exercise)
    return num_dims, v_new, v, w

def test_remaining_variance():
    # Load the data again and apply PCA
    fn = '../data/dataset_hands/coordinates.txt'
    coordinates =  np.loadtxt(fn)
    mn = np.mean(coordinates, axis=0)
    num_dims, v_new, v, w = pca_hands()

    fig = plt.figure(figsize=(15,10))
    
    #------------------------------------------------------------------#
    # TODO: Create a loop to go through the dimensions left in v and
    # compute a variation that this dimension produces.
    #!studentstart
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
    fn = '../data/dataset_hands/test001.jpg'
    img_hand =  plt.imread(fn)
    fn = '../data/dataset_hands/coordinates.txt'
    coordinates =  np.loadtxt(fn)

    fig = plt.figure(figsize=(16,8))
    ax1 = fig.add_subplot(121)
    ax1.imshow(img_hand)

    #L = R * 299/1000 + G * 587/1000 + B * 114/1000
    img2 = img_hand[:,:,0] * 299/1000 + img_hand[:,:,1] * 587/1000 + img_hand[:,:,2] * 114/1000
    ax2 = fig.add_subplot(122)
    ax2.imshow(img2, cmap='gray')
    #------------------------------------------------------------------#
    # TODO: plot the hand template on top of the greyscale hand image
    #!studentstart
    n = 0
    lbl = 'hand_' + str(n+1)
    ax1.plot(coordinates[n,:56], coordinates[n,56:], 'r', lw=3, label=lbl)
    ax2.plot(coordinates[n,:56]*100, coordinates[n,56:]*100, 'r', lw=3, label=lbl)
    #!studentend
    #------------------------------------------------------------------#


def test_transformed_hand():
    fn = '../data/dataset_hands/test001.jpg'
    img_hand =  plt.imread(fn)
    fn = '../data/dataset_hands/coordinates.txt'
    coordinates =  np.loadtxt(fn)
    mn = np.mean(coordinates, axis=0)

    # Initialize position
    # Convert mean shape to 2D format first (easier to work with for the next steps)
    initialpos = np.concatenate((mn[:56].reshape(1,-1), mn[56:].reshape(1,-1)), axis=0)
    
    #------------------------------------------------------------------#
    # TODO: Define a scaling/rotation/alignment matrix and transform the shape
    # as close as possible to the image (result: a variable called shape_t)
    #!studentstart
    alignmatrix = np.array([[500, 0, 150], [0, 500, -80]])
    shape_t = alignmatrix.dot(reg_util.c2h(initialpos))
    #!studentend
    #------------------------------------------------------------------#

    # Plot image and transformed shape
    fig = plt.figure(figsize=(16,8))
    # L = R * 299/1000 + G * 587/1000 + B * 114/1000
    img2 = img_hand[:,:,0] * 299/1000 + img_hand[:,:,1] * 587/1000 + img_hand[:,:,2] * 114/1000
    ax2 = fig.add_subplot(122)
    ax2.imshow(img2, cmap='gray')
    ax2.plot(shape_t[0,:], shape_t[1,:], 'r')
