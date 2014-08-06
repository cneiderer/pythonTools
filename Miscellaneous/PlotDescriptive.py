# -*- coding: utf-8 -*-
"""
Created on Wed Jun 04 11:43:30 2014

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
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import cPickle as pickle
import csv
import gc
#
import LogisticRegression.GoodnessOfFitAnalysis as gof
import Sampling.RandomlySampleInds as rs
import LogisticRegression.ResidualOutlierAnalysis as roa

########## ----- Main Function ----- ##########
def main():
    
    scatterGramExample(withPandasDF=0)
    scatterGramExample(withPandasDF=1)
    
#    listDict = createDescriptiveStat_pTableList()
    
    test = 1
    
#    concatenateAndPlotDescriptiveStat_pTables(listDict)
#    
#    test = 3
    
    
########## ----- Main Function ----- ##########

def scatterGramExample(withPandasDF=0):
    
    # the random data
    x = np.random.randn(1000)
    y = np.random.randn(1000)
    
    # Move data into DataFrame
    data = pd.DataFrame([x, y])
    
    fig, axScatter = plt.subplots(figsize=(5.5,5.5))
    
    # the scatter plot:
    axScatter.scatter(x, y)
    axScatter.set_aspect(1.)
    
    # create new axes on the right and on the top of the current axes
    # The first argument of the new_vertical(new_horizontal) method is
    # the height (width) of the axes to be created in inches.
    divider = make_axes_locatable(axScatter)
    axHistx = divider.append_axes("top", 1.2, pad=0.1, sharex=axScatter)
    axHisty = divider.append_axes("right", 1.2, pad=0.1, sharey=axScatter)
    
    # make some labels invisible
    plt.setp(axHistx.get_xticklabels() + axHisty.get_yticklabels(),
             visible=False)
    
    # now determine nice limits by hand:
    binwidth = 0.25
    xymax = np.max( [np.max(np.fabs(x)), np.max(np.fabs(y))] )
    lim = ( int(xymax/binwidth) + 1) * binwidth
    
    bins = np.arange(-lim, lim + binwidth, binwidth)  

    if not withPandasDF:    
        axHistx.hist(x, bins=bins)
        axHisty.hist(y, bins=bins, orientation='horizontal')
    else:
        axHistx.hist(data[data.columns[0]], bins)
        axHisty.hist(data[data.columns[1]], bins=bins, orientation='horizontal')
    
    # the xaxis of axHistx and yaxis of axHisty are shared with axScatter,
    # thus there is no need to manually adjust the xlim and ylim of these
    # axis.
    
    #axHistx.axis["bottom"].major_ticklabels.set_visible(False)
    for tl in axHistx.get_xticklabels():
        tl.set_visible(False)
    axHistx.set_yticks([0, 50, 100])
    
    #axHisty.axis["left"].major_ticklabels.set_visible(False)
    for tl in axHisty.get_yticklabels():
        tl.set_visible(False)
    axHisty.set_xticks([0, 50, 100])
    
    plt.draw()
    plt.show()

def concatenateAndPlotDescriptiveStat_pTables():

    outputDir = 'C:\\AnalysisTools\\Output\\'    
    
    listDict = pickle.load(open(r'C:\AnalysisTools\Output\descriptiveStat_pTables.pkl'))
    
    concatDescriptiveStat_pTableListDict = {'algorithm_1': [], 
                                            'algorithm_2': [], 
                                            'algorithm_3': []}
    concat_pTableList = []                                            
    for alg in listDict:
        
        pTable = pd.concat(listDict[alg], axis=0)
        
        concat_pTableList.append(pTable)
        
    

def createDescriptiveStat_pTableList():    
    
    outputDir = 'C:\\AnalysisTools\\Output\\'    
    
    pTableList = pickle.load(open(r'C:\AnalysisTools\Output\pTables.pkl'))
    
    descriptiveStat_pTableListDict = {'algorithm_0': [],
                                      'algorithm_1': [],    
                                      'algorithm_2': []}
    descriptiveStatConcat_pTableListDict = {'algorithm_0': [],
                                            'algorithm_1': [],    
                                            'algorithm_2': []}                              
    for alg, pTable in enumerate(pTableList):
        
        alg_name = 'algorithm_' + str(alg)
        
        descriptiveStat_pTableList = []
        for level in pTable.index.levels[0]:
            # Calculate Descriptive Stats
            descriptiveStats = pTable.ix[level].describe()
            # Create MultiIndex
            idx = pd.MultiIndex.from_tuples(zip([alg_name] * len(descriptiveStats), 
                                                [level] * len(descriptiveStats), 
                                                descriptiveStats.index), 
                                            names=['alg', 'level', 'stat'])
            # Replace descriptiveStat index with newly created MultiIndex
            descriptiveStats.index = idx
            # Add to descriptiveStat_pTableList
            descriptiveStat_pTableList.append(descriptiveStats)
        
        # Concatenate pTableList
        concat_pTable = pd.concat(descriptiveStat_pTableList, axis=0)
        # Reorder row index levels
        concat_pTable = concat_pTable.reorder_levels(['stat', 'alg', 'level'], axis=0)
        
        # Compile to dictionaries for saving
        descriptiveStatConcat_pTableListDict[alg_name] = concat_pTable
        descriptiveStat_pTableListDict[alg_name] = descriptiveStat_pTableList
    
    test = 2

    # Save into pickle file
    with open(outputDir + 'descriptiveStatConcat_pTables.pkl', 'w') as f:
        pickle.dump(descriptiveStatConcat_pTableListDict, f)
        
    # Save into pickle file
    with open(outputDir + 'descriptiveStat_pTables.pkl', 'w') as f:
        pickle.dump(descriptiveStat_pTableListDict, f)
    
    return descriptiveStat_pTableListDict   
    
########### ----- Enables Command Line Call ----- ##########
if __name__ == '__main__':
    main()
########### ----- Enables Command Line Call ----- ##########