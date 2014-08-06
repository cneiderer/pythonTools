# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 12:19:18 2014

@author: Curtis.Neiderer
"""

from __future__ import division
import pandas as pd
import statsmodels.api as sm
import pylab as pl
import numpy as np

########## ----- Main Function ----- ##########
def main():
    # read the data in
    df = pd.read_csv("http://www.ats.ucla.edu/stat/data/binary.csv")

    # Dataset summary
    print df.head(), '\n'
    
    # Rename 'rank' column to prestige since there's a DataFrame method called 'rank'
    df.columns = ['admit', 'gre', 'gpa', 'prestige']
    print df.columns, '\n'

    # Summary/descriptive statistics
    print df.describe(), '\n'
    
    # Standard deviation of each column
    print df.std(), '\n'
    
    # Frequency Table cutting prestige and whether or not someone was admitted (i.e., Pivot Table)
    print pd.crosstab(df['admit'], df['prestige']), '\n'
        
#    # Plot histogram of each column
#    df.hist()    
#    pl.show()
    
    # Dummify rank
    dummy_ranks = pd.get_dummies(df['prestige'], prefix='prestige')
    print dummy_ranks.head(), '\n'
    
    # Create a clean data frame for regression
    cols_to_keep = ['admit', 'gre', 'gpa']
    data = df[cols_to_keep].join(dummy_ranks.ix[:, 'prestige_2':])
    print data.head(), '\n'
    
    # Manually add the intercept
    data['intercept'] = 1.0
    print data.head(), '\n'
    
    # Select data columns
    train_cols = data.columns[1:]
    
    # Calculate logic of data columns
    logit = sm.Logit(data['admit'], data[train_cols])
    
    # Fit the model
    result = logit.fit()
    
    # Print the results
    print result.summary(), '\n'
    
    # Calculate Hosmer-Lemeshow Test Statistic
    print 'HL-Statistic: \n', calculateHLStat(data.admit, result.predict())    
    
    # 95% CI only
    print result.conf_int(), '\n'
    
    # Odds ratios only
    print np.exp(result.params), '\n'
    
    # Odds ratios and 95% CI
    params = result.params  
    conf = result.conf_int()
    conf['OR'] = params
    conf.columns = ['2.5%', '97.5%', 'OR']
    print np.exp(conf)
    
    # instead of generating all possible values of GRE and GPA, we're going
    # to use an evenly spaced range of 10 values from the min to the max 
    gres = np.linspace(data['gre'].min(), data['gre'].max(), 10)
    print gres, '\n'
    gpas = np.linspace(data['gpa'].min(), data['gpa'].max(), 10)
    print gpas, '\n'
    
    # enumerate all possibilities
    combos = pd.DataFrame(cartesian([gres, gpas, [1, 2, 3, 4], [1.]]))
    # recreate the dummy variables
    combos.columns = ['gre', 'gpa', 'prestige', 'intercept']
    dummy_ranks = pd.get_dummies(combos['prestige'], prefix='prestige')
    dummy_ranks.columns = ['prestige_1', 'prestige_2', 'prestige_3', 'prestige_4']
     
    # keep only what we need for making predictions
    cols_to_keep = ['gre', 'gpa', 'prestige', 'intercept']
    combos = combos[cols_to_keep].join(dummy_ranks.ix[:, 'prestige_2':])
     
    # make predictions on the enumerated dataset
    combos['admit_pred'] = result.predict(combos[train_cols])
     
    print combos.head(), '\n'    

    def isolate_and_plot(variable):
        # isolate gre and class rank
        grouped = pd.pivot_table(combos, values=['admit_pred'], rows=[variable, 'prestige'],
                                 aggfunc=np.mean)
        
        # in case you're curious as to what this looks like
        # print grouped.head()
        #                      admit_pred
        # gre        prestige            
        # 220.000000 1           0.282462
        #            2           0.169987
        #            3           0.096544
        #            4           0.079859
        # 284.444444 1           0.311718
        
        # make a plot
        colors = 'rbgyrbgy'
        pl.figure()
        for col in combos.prestige.unique():
            plt_data = grouped.ix[grouped.index.get_level_values(1)==col]
            pl.plot(plt_data.index.get_level_values(0), plt_data['admit_pred'],
                    color=colors[int(col)])
    
        pl.xlabel(variable)
        pl.ylabel("P(admit=1)")
        pl.legend(['1', '2', '3', '4'], loc='upper left', title='Prestige')
        pl.title("Prob(admit=1) isolating " + variable + " and presitge")
        pl.show()
    
    isolate_and_plot('gre')
    isolate_and_plot('gpa')

    test = 1


########## ----- Main Function ----- ##########

def calculateHLStat(obsOutcome, predOutcomeProb):
    
    # Break predicted outcome probabilities into deciles
    predDeciles = pd.qcut(predOutcomeProb, np.arange(0, 1.1, 0.1))
    
    # Pre-allocate
    onesArray = np.nan * np.ones((10,3))
    zerosArray = np.nan * np.zeros((10,3))
    
    # Loop through deciles
    for group in range(10):
        # Observation Counts
        onesCnt = np.sum(obsOutcome[predDeciles.labels == group])
        onesArray[group, 0] = onesCnt
        zerosCnt = np.sum(predDeciles.labels == group) - onesCnt
        zerosArray[group, 0] = zerosCnt
        # Predicted Probabilities
        onesProb = np.sum(predOutcomeProb[predDeciles.labels == group])
        onesArray[group, 1] = onesProb
        zerosProb = np.sum(predDeciles.labels == group) - onesProb
        zerosArray[group, 1] = zerosProb
        # Chi-Squared 
        onesChiSquare = (onesCnt - onesProb) ** 2 / onesProb
        onesArray[group, 2] = onesChiSquare
        zerosChiSquare = (zerosCnt - zerosProb) ** 2 / zerosProb
        zerosArray[group, 2] = zerosChiSquare

    # Chi-Squared Sum and probability
    chiSquareSum = np.sum(onesArray[:, 2]) + np.sum(zerosArray[:, 2])
    chiSquaredof = 8 # dof = g - 2
    chiSquareProb = sm.stats.stattools.stats.chisqprob(chiSquareSum, chiSquaredof)
    
    return chiSquareSum, chiSquareProb  

def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.
 
    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.
 
    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.
 
    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])
 
    """
 
    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype
 
    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)
 
    m = n / arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out

#def isolate_and_plot(variable):
#    # isolate gre and class rank
#    grouped = pd.pivot_table(combos, values=['admit_pred'], rows=[variable, 'prestige'],
#                             aggfunc=np.mean)
#    
#    # in case you're curious as to what this looks like
#    # print grouped.head()
#    #                      admit_pred
#    # gre        prestige            
#    # 220.000000 1           0.282462
#    #            2           0.169987
#    #            3           0.096544
#    #            4           0.079859
#    # 284.444444 1           0.311718
#    
#    # make a plot
#    colors = 'rbgyrbgy'
#    for col in combos.prestige.unique():
#        plt_data = grouped.ix[grouped.index.get_level_values(1)==col]
#        pl.plot(plt_data.index.get_level_values(0), plt_data['admit_pred'],
#                color=colors[int(col)])
#
#    pl.xlabel(variable)
#    pl.ylabel("P(admit=1)")
#    pl.legend(['1', '2', '3', '4'], loc='upper left', title='Prestige')
#    pl.title("Prob(admit=1) isolating " + variable + " and presitge")
#    pl.show()
#
#isolate_and_plot('gre')
#isolate_and_plot('gpa')

########### ----- Enables Command Line Call ----- ##########
if __name__ == '__main__':
    main()
########### ----- Enables Command Line Call ----- ##########