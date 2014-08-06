# -*- coding: utf-8 -*-
"""
Created on Thu May 29 13:19:09 2014

@author: Curtis.Neiderer
"""

import numpy as np

def main():
    
    col1 = np.concatenate([np.ones((5,1), dtype=bool), 
                           np.zeros((5,1), dtype=bool)], axis=0)
    col2 = np.zeros((10,1), dtype=bool)
    col3 = np.concatenate([np.ones((2,1), dtype=bool), 
                           np.zeros((2,1), dtype=bool), 
                           np.ones((2,1), dtype=bool),
                           np.zeros((2,1), dtype=bool),
                           np.ones((2,1), dtype=bool)])
                               
    combo = np.concatenate([col1, col2, col3], axis=1)

    orResult = np.zeros((np.shape(combo)[0],1), dtype=bool)
    for row in range(np.shape(combo)[0]):
        orResult[row] = listOr(combo[row, :])

    print orResult

def listOr(vectorList):
    if np.shape(vectorList)[0] == 1:
        return vectorList[0]
    else:
        return vectorList[0] | listOr(vectorList[1:])

########### ----- Enables Command Line Call ----- ##########
if __name__ == '__main__':
    main()
########### ----- Enables Command Line Call ----- ##########