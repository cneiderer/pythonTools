# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 14:28:28 2014

@author: Curtis.Neiderer
"""

from __future__ import division
import numpy as np

########## ----- Main Function ----- ##########
def main():
    
    populationLength = 1000
    sampleLength = 100
#    indList = np.array([np.arange(0, populationLength, 1)]).T
    indList = list(np.arange(0, populationLength, 1))
    randIndList = randomlySampleInds(indList, sampleLength)
    print 'Index List: \n', indList, '\n'
    print 'Randomly Sampled Index List: \n', randIndList, '\n'
    print 'Unique Samples Expected: \n', sampleLength
    print 'Unique Samples Actual: \n', len(np.unique(randIndList)), '\n'

########## ----- Main Function ----- ##########
def randomlySampleInds(indList, numSamples):
    '''
    ----
    
    randomlySampleInds(Array of indices, Integer defining number of samples) -> Array of sampled indices
    
    Description:
        Returns the array of strings with everything in lowercase
    
    Inputs:
        indList
        numSamples
    
    Outputs:
        sampledList
        
    Example:
        >>> indList = np.arange(0, 10, 1)
        >>> indList
        array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        >>> numSamples = 5
        >>> numSamples
        5
        >>> randomlySampleInds(indList, numSamples)
        array([ 4.,  7.,  5.,  1.,  2.])
    ----
    
    Author:
        Curtis Neiderer, 2/17/2014
    '''
        
    # Find number of row indices
    arrayLength = len(indList)
    arrayInds = np.array([np.arange(0, arrayLength, 1)]).T

    # Set random index maximum
    randIndMax = arrayLength

    # Pre-allocate array order with default values    
    sampledList = np.ones((numSamples, 1)) * np.nan

    for ii in range(numSamples):
        # Get random index
        randInd = arrayInds[np.random.randint(low=0, high=randIndMax)]
        # Add to arrayOrder array
        sampledList[ii] = randInd
        # Remove used indices from index array
        arrayInds = arrayInds[arrayInds != randInd]
        # Decrement max index
        randIndMax -= 1
        
    return sampledList[:, 0]
    

########### ----- Enables Command Line Call ----- ##########
if __name__ == '__main__':
    main()
########### ----- Enables Command Line Call ----- ##########