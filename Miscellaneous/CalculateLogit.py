# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 16:49:12 2014

@author: Curtis.Neiderer
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import statsmodels as sm
import pandas as pd

########## ----- Main Function ----- ##########
def main():

    x = np.linspace(0, 40)
    alpha = np.array([-4, -8, -12, -20])
    beta = np.array([0.4, 0.4, 0.6, 1.0])
    
    fig = plt.figure(1)
    ax = fig.add_subplot(1, 1, 1)
    ax.grid(True)
    
    color = 'rgbk'
    for ii in range(len(alpha)):
        calcLogit = logit(x, alpha[ii], beta[ii])
#        print 'Alpha = ' + str(alpha[ii]) + ', beta = ' + str(beta[ii]) + ":\n", calcLogit
        ax.plot(x, calcLogit, '-' + color[ii])
        
    fig.show()
    

########## ----- Main Function ----- ##########

def logit(x, alpha, beta):
    p = np.exp(alpha + (beta * x)) / (1 + np.exp(alpha + (beta * x)))
    
    return np.round(p, 3)
    
########### ----- Enables Command Line Call ----- ##########
if __name__ == '__main__':
    main()
########### ----- Enables Command Line Call ----- ##########