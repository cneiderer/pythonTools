# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 12:20:32 2014

@author: Curtis.Neiderer
"""

##### Import Necessary Packages #####
from __future__ import division
#
import pandas as pd
import statsmodels.api as sm
import pylab as pl
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

########## ----- Main Function ----- ##########
def main():

    df = pd.read_csv("C:\AnalysisTools\python_toolbox\Output\cleanDF_binary.csv")
    df = df[["admit", "gre", "gpa", "rank"]]    
    
    df.pivot_table()
    
#    inputVal = 35
#    print "Fibonacci(" + str(inputVal) + "): ", fib(inputVal)    
    
#    num_list = [1, 2, [11, 13], 8]
#    print "Recursive Sum: ", r_sum(num_list)    
    
#    inputVal = 5
#    print "Factorial(" + str(inputVal) + "): ", factorial(inputVal)
    
#    print numCombinations(4, 2)    
    
    test = 1
    
def numCombinations(N, k):
    return np.math.factorial(N) / (np.math.factorial(k) * (np.math.factorial(N - k)))
    
def fib(n):
    if n <= 1:
        return n
    t = fib(n - 1) + fib(n - 2)
    return t

def r_sum(nested_num_list):
    tot = 0
    for element in nested_num_list:
        if type(element) == type([]):
            tot += r_sum(element)
        else:
            tot += element
    return tot

def factorial(N):
    if type(N) is int:
        if N == 1:
            return 1
        else:
            return N * factorial(N - 1)
    else:
        print "Error: factorial() only accepts integer values"
        


########### ----- Enables Command Line Call ----- ##########
if __name__ == '__main__':
    main()
########### ----- Enables Command Line Call ----- ##########