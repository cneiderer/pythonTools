# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 09:03:57 2014

@author: Curtis.Neiderer
"""

from __future__ import division
import pandas as pd
import statsmodels.api as sm
import pylab as pl
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt

########## ----- Main Function ----- ##########
def main():
    
    # Describe dataset
    print sm.datasets.scotland.DESCRLONG
    
    # Load the data and add a constant to the exogenous variables:
    data2 = sm.datasets.scotland.load()
    data2.exog = sm.add_constant(data2.exog, prepend=False)
    print data2.exog[:5, :]
    print data2.endog[:5]
    
    # Fit and summary
    glm_gamma = sm.GLM(data2.endog, data2.exog, family=sm.families.Gamma())
    glm_results = glm_gamma.fit()
    print glm_results.summary()
    
    #
    
########## ----- Main Function ----- ##########


########### ----- Enables Command Line Call ----- ##########
if __name__ == '__main__':
    main()
########### ----- Enables Command Line Call ----- ##########