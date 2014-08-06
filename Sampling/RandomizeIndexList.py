# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 08:21:21 2014

@author: Curtis.Neiderer
"""

from __future__ import division
import numpy as np

########## ----- Main Function ----- ##########
def main():
    
    indList = np.array([np.arange(0, 100, 1)]).T
    randIndList = randomizeIndexList(indList)
#    print indList, '\n', randIndList, '\n'
    print np.concatenate((indList, randIndList), axis=1)

########## ----- Main Function ----- ##########
def randomizeIndexList(origArray):
        
    # Find number of row indices
    arrayLength = len(origArray)
    arrayInds = np.array([np.arange(0, arrayLength, 1)]).T

    # Set random index maximum
    randIndMax = arrayLength

    # Pre-allocate array order with default values    
    arrayOrder = np.ones((arrayLength, 1)) * -99

    for ii, index in enumerate(arrayInds):
        # Get random index
        randInd = arrayInds[np.random.randint(low=0, high=randIndMax)]
        # Add to arrayOrder array
        arrayOrder[ii] = randInd
        # Remove used indices from index array
        arrayInds = arrayInds[arrayInds != randInd]
        # Decrement max index
        randIndMax -= 1
    
    randomArray = origArray[arrayOrder[:, 0].tolist()]
        
    return randomArray
    


########### ----- Enables Command Line Call ----- ##########
if __name__ == '__main__':
    main()
########### ----- Enables Command Line Call ----- ##########