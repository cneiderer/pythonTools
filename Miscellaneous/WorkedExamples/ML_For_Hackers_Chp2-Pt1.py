# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 09:25:14 2014

@author: Curtis.Neiderer
"""

import os
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pandas import *
# formulas.api lets us use patsy formulas.
from statsmodels.formula.api import ols

########## ----- Main Function ----- ##########
def main():
    # Height and weight data
    heights_weights = read_csv('C:\AnalysisTools\python_toolbox\DataSets\\01_heights_weights_genders.csv')

    # Inspecting the data with head    
    print heights_weights.head(10), '\n'
    
    print heights_weights.groupby('Gender')['Gender'].count(), '\n'

    # Numeric summaries, especially quantiles
    heights = heights_weights['Height']
    print heights.describe(), '\n'
    
    # Note that the default argument gives quartiles. We can get deciles by calling
    deciles = my_quantiles(heights, prob = arange(0, 1.1, 0.1))    
    print deciles
    
    # Histograms
    # First, 1-inch bins:
    fig = plt.figure()
    bins1 = np.arange(heights.min(), heights.max(), 1.0)
    heights.hist(bins = bins1, fc = 'steelblue')
    
    # Next, 5-inch bins:
    fig = plt.figure()
    bins5 = np.arange(heights.min(), heights.max(), 5.)
    heights.hist(bins = bins5, fc = 'steelblue')

    # And finally, 0.001-inch bins:
    fig = plt.figure()
    bins001 = np.arange(heights.min(), heights.max(), .001)
    heights.hist(bins = bins001, fc = 'steelblue')
    plt.savefig('height_hist_bins001.png')

########## ----- Main Function ----- ##########

def my_range(s):
    '''
    Difference between the max and min of an array or Series
    '''
    return s.max() - s.min()

def my_quantiles(s, prob=(0.0, 0.25, 0.5, 0.75, 1.0)):
    '''
    Calculate quantiles of a series.
    
    Parameters:
    -----------
    s : a pandas Series
    prob : a tuple (or other iterable) of probabilities at
    which to compute quantiles. Must be an iterable,
    even for a single probability (e.g. prob = (0.50,)
    not prob = 0.50).
    
    Returns:
    --------
    A pandas series with the probabilities as an index.
    '''
    q = [s.quantile(p) for p in prob]
    return Series(q, index = prob)
    
    
########### ----- Enables Command Line Call ----- ##########
if __name__ == '__main__':
    main()
########### ----- Enables Command Line Call ----- ##########