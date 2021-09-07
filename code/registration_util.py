"""
Utility functions for registration.
"""

import numpy as np
import matplotlib.pyplot as plt


def test_object(centered=True):
    # Generate an F-like test object.
    # Input:
    # centered - set the object centroid to the origin
    # Output:
    # X - coordinates of the test object

    X = np.array([[4, 4, 4.5, 4.5, 6, 6, 4.5, 4.5, 7, 7, 4], [10, 4, 4, 7, 7, 7.5, 7.5, 9.5, 9.5, 10, 10]])

    if centered:
        X[0, :] = X[0, :] - np.mean(X[0, :])
        X[1, :] = X[1, :] - np.mean(X[1, :])

    return X


def c2h(X):
    # Convert cartesian to homogeneous coordinates.
    # Input:
    # X - cartesian coordinates
    # Output:
    # Xh - homogeneous coordinates

    n = np.ones([1,X.shape[1]])
    Xh = np.concatenate((X,n))

    return Xh


def t2h(T, t):
    # Convert a 2D transformation matrix to homogeneous form.
    # Input:
    # T - 2D transformation matrix
    # t - 2D translation vector
    # Output:
    # Th - homogeneous transformation matrix

    #------------------------------------------------------------------#
    # TODO: Implement conversion of a transformation matrix and a translation vector to homogeneous transformation matrix.
    #------------------------------------------------------------------#
    
    
    print("change output to return Th")

def plot_object(ax, X):
    # Plot 2D object.
    #
    # Input:
    # X - coordinates of the shape

    ax.plot(X[0,:], X[1,:], linewidth=2)


def cpselect(imagePath1, imagePath2):
	# Pops up a matplotlib window in which to select control points on the two images given as input.
	#
	# Input:
    # imagePath1 - fixed image path
    # imagePath2 - moving image path
    # Output:
    # X - control points in the fixed image
    # Xm - control points in the moving image
	
	#load the images
	image1 = plt.imread(imagePath1)
	image2 = plt.imread(imagePath2)
	
	#ensure that the plot opens in its own window
	get_ipython().run_line_magic('matplotlib', 'qt')
	
	#set up the overarching window
	fig, axes = plt.subplots(1,2)
	fig.figsize = [16,9]
	fig.suptitle("Left Mouse Button to create a point.\n Right Mouse Button/Delete/Backspace to remove the newest point.\n Middle Mouse Button/Enter to finish placing points.\n First select a point in Image 1 and then its corresponding point in Image 2.")
	
	#plot the images
	axes[0].imshow(image1)
	axes[0].set_title("Image 1")
	
	axes[1].imshow(image2)
	axes[1].set_title("Image 2")
	
	#accumulate points
	points = plt.ginput(n=-1, timeout=30)
	plt.close(fig)
	
	#restore to inline figure placement
	get_ipython().run_line_magic('matplotlib', 'inline')
	
	#if there is an uneven amount of points, raise an exception
	if not (len(points)%2 == 0):
		raise Exception("Uneven amount of control points: {0}. Even amount of control points required.".format(len(points)))
		
	#if there are no points, raise an exception
	if not (len(points)> 0):
		raise Exception("No control points selected.")
	
	#subdivide the points into two different arrays. If the current number is even belongs to the first first image, and uneven to the second image. (Assuming the points were entered in the correct order.)
	#X and Y values are on rows, with each column being a pair of values.
	k = len(points)//2
	X = np.empty((2,k))
	X[:] = np.nan
	Xm = np.empty((2,k))
	Xm[:] = np.nan

	for i in np.arange(len(points)):
		if i%2 == 0 :
			X[0,i//2] = points[i][0]
			X[1,i//2] = points[i][1]
		else:
			Xm[0,i//2] = points[i][0]
			Xm[1,i//2] = points[i][1]
	
	return X, Xm
