"""
Test code for computer-aided diagnosis.
"""

from builtins import print
import numpy as np
import cad_util as util
import segmentation_util as seg_util
import registration as reg
import segmentation as seg
import matplotlib.pyplot as plt
import cad
from IPython.display import display, clear_output, HTML


# SECTION 1: Linear regression


def linear_regression():
    
    # load the training, validation and testing datasets
    fn1 = '../data/linreg_ex_test.txt'
    fn2 = '../data/linreg_ex_train.txt'
    fn3 = '../data/linreg_ex_validation.txt'
    # shape (30,2) numpy array; x = column 0, y = column 1
    test_data = np.loadtxt(fn1)
    # shape (20,2) numpy array; x = column 0, y = column 1
    train_data = np.loadtxt(fn2)
    # shape (10,2) numpy array; x = column 0, y = column 1
    validation_data = np.loadtxt(fn3)

    # plot the training dataset
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    ax.plot(train_data[:,0], train_data[:,1], '*')
    ax.grid()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Training data')

    #---------------------------------------------------------------------#
    # TODO: Implement training of a linear regression model.
    # Here you should reuse ls_solve() from the registration mini-project.
    # The provided addones() function adds a column of all ones to a data
    # matrix X in a similar way to the c2h() function used in registration.

    trainX = train_data[:,0].reshape(-1,1)
    trainXones = util.addones(trainX)
    trainY = train_data[:,1].reshape(-1,1)
    
    #---------------------------------------------------------------------#

    fig1 = plt.figure(figsize=(10,10))
    ax1 = fig1.add_subplot(111)
    util.plot_regression(trainX, trainY, Theta, ax1)
    ax1.grid()
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.legend(('Original data', 'Regression curve', 'Predicted Data', 'Error'))
    ax1.set_title('Training set')

    testX = test_data[:,0].reshape(-1,1)
    testY = test_data[:,1].reshape(-1,1)

    fig2 = plt.figure(figsize=(10,10))
    ax2 = fig2.add_subplot(111)
    util.plot_regression(testX, testY, Theta, ax2)
    ax2.grid()
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.legend(('Original data', 'Regression curve', 'Predicted Data', 'Error'))
    ax2.set_title('Test set')

    #---------------------------------------------------------------------#
    # TODO: Compute the error for the trained model.
    #---------------------------------------------------------------------#

    return E_validation, E_test


def quadratic_regression():
    #---------------------------------------------------------------------#
    # TODO: Implement training of a quadratic regression model.
    #---------------------------------------------------------------------#

    return E_validation, E_test


# SECTION 2: Logistic regression
 

def logistic_regression():
    
    # dataset preparation
    num_training_samples = 300
    num_validation_samples = 100
    
    # here we reuse the function from the segmentation practicals
    m1=[2,3]
    m2=[-0,-4]
    s1=[[8,7],[7,8]]
    s2=[[8,6],[6,8]]

    [trainingX, trainingY] = seg.generate_gaussian_data(num_training_samples, m1, m2, s1, s2)
    r,c = trainingX.shape
    print('Training sample shape: {}'.format(trainingX.shape))

    # we need a validation set to monitor for overfitting
    [validationX, validationY] = seg.generate_gaussian_data(num_validation_samples, m1, m2, s1, s2)
    r_val,c_val = validationX.shape
    print('Validation sample shape: {}'.format(validationX.shape))
    
    validationXones = util.addones(validationX)

    # train a logistic regression model:
    # the learning rate for the gradient descent method
    # (the same as in intensity-based registration)
    mu = 0.001

    # we are actually using stochastic gradient descent
    batch_size = 30

    # initialize the parameters of the model with small random values,
    # we need one parameter for each feature and a bias
    Theta = 0.02*np.random.rand(c+1, 1)

    # number of gradient descent iterations
    num_iterations = 300

    # variables to keep the loss and gradient at every iteration
    # (needed for visualization)
    iters = np.arange(num_iterations)
    loss = np.full(iters.shape, np.nan)
    validation_loss = np.full(iters.shape, np.nan)

    # Create base figure
    fig = plt.figure(figsize=(15,8))
    ax1 = fig.add_subplot(121)
    im1, Xh_ones, num_range_points = util.plot_lr(trainingX, trainingY, Theta, ax1)
    seg_util.scatter_data(trainingX, trainingY, ax=ax1)
    ax1.grid()
    ax1.set_xlabel('x_1')
    ax1.set_ylabel('x_2')
    ax1.legend()
    ax1.set_title('Training set')
    text_str1 = '{:.4f};  {:.4f};  {:.4f}'.format(0, 0, 0)
    txt1 = ax1.text(0.3, 0.95, text_str1, bbox={'facecolor': 'white', 'alpha': 1, 'pad': 10}, transform=ax1.transAxes)

    ax2 = fig.add_subplot(122)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Loss (average per sample)')
    ax2.set_title('mu = '+str(mu))
    h1, = ax2.plot(iters, loss, linewidth=2, label='Training loss')
    h2, = ax2.plot(iters, validation_loss, linewidth=2, label='Validation loss')
    ax2.set_ylim(0, 0.7)
    ax2.set_xlim(0, num_iterations)
    ax2.grid()
    ax1.legend()

    text_str2 = 'iter.: {}, loss: {:.3f}, val. loss: {:.3f}'.format(0, 0, 0)
    txt2 = ax2.text(0.3, 0.95, text_str2, bbox={'facecolor': 'white', 'alpha': 1, 'pad': 10}, transform=ax2.transAxes)

    # iterate
    for k in np.arange(num_iterations):
        
        # pick a batch at random
        idx = np.random.randint(r, size=batch_size)

        # the loss function for this particular batch
        loss_fun = lambda Theta: cad.lr_nll(util.addones(trainingX[idx,:]), trainingY[idx], Theta)

        # gradient descent:
        # here we reuse the code for numerical computation of the gradient
        # of a function
        Theta = Theta - mu*reg.ngradient(loss_fun, Theta)

        # compute the loss for the current model parameters for the
        # training and validation sets
        # note that the loss is divided with the number of samples so
        # it is comparable for different number of samples
        loss[k] = loss_fun(Theta)/batch_size
        validation_loss[k] = cad.lr_nll(validationXones, validationY, Theta)/r_val

        # upldate the visualization
        ph = cad.sigmoid(Xh_ones.dot(Theta)) > 0.5
        decision_map = ph.reshape(num_range_points, num_range_points)
        decision_map_trns = np.flipud(decision_map)
        im1.set_data(decision_map_trns)
        text_str1 = '{:.4f};  {:.4f};  {:.4f}'.format(Theta[0,0], Theta[1,0], Theta[2,0])
        txt1.set_text(text_str1)
        h1.set_ydata(loss)
        h2.set_ydata(validation_loss)
        text_str2 = 'iter.={}, loss={:.3f}, val. loss={:.3f} '.format(k, loss[k], validation_loss[k])
        txt2.set_text(text_str2)


        display(fig)
        clear_output(wait = True)
