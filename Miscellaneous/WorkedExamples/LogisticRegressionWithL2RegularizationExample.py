# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 14:54:38 2014

@author: Curtis.Neiderer
"""

from __future__ import division
from scipy.optimize.optimize import fmin_cg, fmin_bfgs, fmin
import numpy as np
import matplotlib.pyplot as plt
from pylab import *

########## ----- Main Function ----- ##########

def main():
    # Create 20 dimensional data set with 25 points -- this will be
    # susceptible to overfitting.
    data = SyntheticClassifierData(25, 20)
    
    # Run for a variety of regularization strengths
    alphas = [0, .001, .01, .1]
    for j, a in enumerate(alphas):
        
        # Create a new learner, but use the same data for each run
        lr = LogisticRegression(x_train=data.X_train, y_train=data.Y_train,
        x_test=data.X_test, y_test=data.Y_test,
        alpha=a)
     
        print "Initial likelihood:"
        print lr.lik(lr.betas)
        
        # Train the model
        lr.train()
        
        # Display execution info
        print "Final betas:"
        print lr.betas
        print "Final lik:"
        print lr.lik(lr.betas)

        # Plot the results
        plt.subplot(len(alphas), 2, 2*j + 1)
        lr.plot_training_reconstruction()
        plt.ylabel("Alpha=%s" % a)
        if j == 0:
            plt.title("Training set reconstructions")

        plt.subplot(len(alphas), 2, 2*j + 2)
        lr.plot_test_predictions()
        if j == 0:
            plt.title("Test set predictions")

    show()       

########## ----- Main Function ----- ##########


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

class SyntheticClassifierData():

    def __init__(self, N, d):
        """ 
        Create N instances of d dimensional input vectors and a 1D
        class label (-1 or 1). 
        """
    
        means = .05 * np.random.randn(2, d)
        
        self.X_train = np.zeros((N, d))
        self.Y_train = np.zeros(N)
        for ii in range(N):
            if np.random.random() > .5:
                y = 1
            else:
                y = 0
            self.X_train[ii, :] = np.random.random(d) + means[y, :]
            self.Y_train[ii] = 2.0 * y - 1
        
        self.X_test = np.zeros((N, d))
        self.Y_test = np.zeros(N)
        for ii in range(N):
            if np.random.randn() > .5:
                y = 1
            else:
                y = 0
            self.X_test[ii, :] = np.random.random(d) + means[y, :]
            self.Y_test[ii] = 2.0 * y - 1
        
class LogisticRegression():
    """
    A simple logistic regression model with L2 regularization 
    (zero-mean Gaussian priors on parameters)
    """
    
    def __init__(self, x_train=None, y_train=None, x_test=None, y_test=None, 
                 alpha=0.1, sythetic=False):
        
        # Set of L2 regularization strength
        self.alpha = alpha
        
        # Set of data
        self.set_data(x_train, y_train, x_test, y_test)
        
        # Initialize parameters to zero, for lack of a better choice
        self.betas = np.zeros(self.x_train.shape[1])
        
    def negative_lik(self, betas):
        return -1 * self.lik(betas)
    
    def lik(self, betas):
        """
        Likelihood of the data under the current settings of parameters. 
        """
        # Data likelihood
        l = 0
        for ii in range(self.n):
            l += np.log(sigmoid(self.y_train[ii] * np.dot(betas, self.x_train[ii, :])))
        
        # Prior likelihood
        for kk in range(1, self.x_train.shape[1]):
            l -= (self.alpha / 2.0) * self.betas[kk] ** 2
            
        return l
        
    def train(self):
        """
        Define the gradient and hand it off to a scipy gradient-based optimizer. 
        """
        # Define the derivative of the likelihood with respect to beta_k.
        # Need to multiply by -1 because we will be minimizing.
        dB_k = lambda B, k : (k > 0) * self.alpha * B[k] - np.sum([ \
                self.y_train[i] * self.x_train[i, k] * \
                sigmoid(-self.y_train[i] *\
                np.dot(B, self.x_train[i,:])) \
                for i in range(self.n)])
        
        # The full gradient is just an array of componentwise derivatives
        dB = lambda B : np.array([dB_k(B, k) \
                for k in range(self.x_train.shape[1])])
        
        # Optimize
        self.betas = fmin_bfgs(self.negative_lik, self.betas, fprime=dB)
        
    def set_data(self, x_train, y_train, x_test, y_test):
        """ 
        Take data that's already been generated. 
        """
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.n = y_train.shape[0]

    def training_reconstruction(self):
        p_y1 = np.zeros(self.n)
        for i in range(self.n):
            p_y1[i] = sigmoid(np.dot(self.betas, self.x_train[i,:]))

        return p_y1

    def test_predictions(self):
        p_y1 = np.zeros(self.n)
        for i in range(self.n):
            p_y1[i] = sigmoid(np.dot(self.betas, self.x_test[i,:]))
        
        return p_y1

    def plot_training_reconstruction(self):
        plt.plot(np.arange(self.n), .5 + .5 * self.y_train, 'bo')
        plt.plot(np.arange(self.n), self.training_reconstruction(), 'rx')
        plt.ylim([-.1, 1.1])

    def plot_test_predictions(self):
        plt.plot(np.arange(self.n), .5 + .5 * self.y_test, 'yo')
        plt.plot(np.arange(self.n), self.test_predictions(), 'rx')
        plt.ylim([-.1, 1.1]) 
 
########### ----- Enables Command Line Call ----- ##########
if __name__ == '__main__':
    main()
########### ----- Enables Command Line Call ----- ########## 
        