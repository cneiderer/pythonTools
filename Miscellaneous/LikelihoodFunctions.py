# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 09:28:51 2014

@author: Curtis.Neiderer
"""

from __future__ import division
import numpy as np
import math
import matplotlib.pyplot as plt

########## ----- Main Function ----- ##########
def main():
    n = 11
    x = 7
    p = np.linspace(0.00, 1.00, 101)
    
    binL = binomialLikelihoodFunction(n, x, p)
    
    print 'x = ' + str(x), 'n = ' + str(n) + '\n'
    print 'p: \n', np.round(p, 3)
    print 'Binomial Likeliehood: X ~ Bin(n,p), L(p; x)\n', np.round(binL, 3)
    print 'Binomial Log-Likelihood: l(p;x)\n', np.round(np.log(binL), 3)
    
    fig = plt.figure(1)
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.grid(True)
    # plot fit line
    ax1.plot(p, binL, '-r')
    fig.show()
    
    fig = plt.figure(2)
    ax2 = fig.add_subplot(1, 1, 1)
    ax2.grid(True)
    # plot fit line
    ax2.plot(p, np.log(binL), '-r')
    fig.show()
    
    print np.concatenate((np.array([p]).T, np.array([binL]).T, np.array([np.log(binL)]).T), axis=1)
    
########## ----- Main Function ----- ##########

def binomialLikelihoodFunction(n, x, p):
    '''
    binomialLikelihoodFunction
        Calculates the binomial likelihood function
        
        L(p|n,x) = (n choose x)*p^y*(1-p)^(n-y)
        
    Inputs:
        n   -   sample size
        x   -   number of successes in sample
        p   -   probability (0 <= p <= 1)
        
    Outputs:
        L   -   binomial likelihood function value
    
    Example:
    
    Author:
        Curtis Neiderer
    '''
    
    L = (math.factorial(n) / (math.factorial(x) * math.factorial(n - x))) * (p ** x) * ((1 - p) ** (n - x))
    
    return L
    
    
########### ----- Enables Command Line Call ----- ##########
if __name__ == '__main__':
    main()
########### ----- Enables Command Line Call ----- ##########