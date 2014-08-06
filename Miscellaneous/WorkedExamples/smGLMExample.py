# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 15:43:29 2014

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
    data = sm.datasets.star98.load()
    
    print sm.datasets.star98.NOTE
    
    # Load the data and add a constant to the exogenous (independent) variables
    data.exog = sm.add_constant(data.exog, prepend=False)
    print data.endog[:5, :], '\n'
    
    # The dependent variable is N by 2 (Success: NABOVE, Failure: NBELOW):
    print data.exog[:2, :], '\n'

    # Fit and summary
    glm_binom = sm.GLM(data.endog, data.exog, family=sm.families.Binomial())

    res = glm_binom.fit()

    print res.summary(), '\n'        
        
    # We extract information that will be used to draw some interesting plots
    nobs = res.nobs
    y = data.endog[:, 0] / data.endog.sum(1)
    yhat = res.mu
        
    #Plot yhat vs y:
    plt.figure()
    plt.scatter(yhat, y)            
    
    line_fit = sm.OLS(y, sm.add_constant(yhat, prepend=False)).fit().params

    fit = lambda x: line_fit[1] + line_fit[0] * x  # better way in scipy?
    plt.plot(np.linspace(0, 1, nobs), fit(np.linspace(0, 1, nobs)))
    
    plt.title('Model Fit Plot')
    plt.ylabel('Observed Values')
    plt.xlabel('Fitted Values')
    
    # Plot yhat vs. Pearson residuals:    
    plt.figure();
    plt.scatter(yhat, res.resid_pearson);
    plt.plot([0.0, 1.0], [0.0, 0.0], 'k-');
    plt.title('Residual Dependence Plot');
    plt.ylabel('Pearson Residuals');
    plt.xlabel('Fitted values');
    
    # Histogram of standardized deviance residuals
    plt.figure();
    resid = res.resid_deviance.copy()
    resid_std = (resid - resid.mean()) / resid.std()
    plt.hist(resid_std, bins=25);
    plt.title('Histogram of standardized deviance residuals');

    # QQ Plot of Deviance Residuals
    from statsmodels import graphics
    graphics.gofplots.qqplot(resid, line='r');    

    test = 1
    
########## ----- Main Function ----- ##########


########### ----- Enables Command Line Call ----- ##########
if __name__ == '__main__':
    main()
########### ----- Enables Command Line Call ----- ##########