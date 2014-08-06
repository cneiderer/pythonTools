# -*- coding: utf-8 -*-
"""
Created on Mon May 05 09:22:21 2014

@author: Curtis.Neiderer
"""

# Load packages
from __future__ import division
import numpy as np
import scipy
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import itertools

########## ----- Main Function ----- ##########
def main():
    
    # Load data into a DataFrame
#    dataFilePath = 'http://www.ats.ucla.edu/stat/data/binary.csv'
#    dataFilePath = 'C:/AnalysisTools/python_toolbox/DataSets/AdultDataSet.csv'
    
    dataFilePath = 'C:/AnalysisTools/python_toolbox/DataSets/glow500.csv'
    df = pd.read_csv(dataFilePath) 
        
    # Make all parameter names lowercase
    df.columns = [col.lower() for col in df.columns]
        
    #  Define parameters of interest (i.e., create a clean DataFrame)
    # If set to True, keep parameters in paramsOfInterest; if set to False, remove
    paramsOfInterestFlag = False
    paramsOfInterest = ['phy_id']
    # Modify DataFrame to contain parameters of interest only, if they're defined
    if len(paramsOfInterest) > 0:
        if paramsOfInterestFlag == True:
            # Keep parameters of interest    
            df = df[paramsOfInterest]
        elif paramsOfInterestFlag == False:
            # Remove parameters not of interest
            for ii, param in enumerate(paramsOfInterest):
                df = df[df.columns[df.columns != param]]
        else:
            # Throw exception, ValueError: paramsOfInterestFlag must be True or False
            raise ValueError('paramsOfInterestFlag must be set to True or False')
        
    # Bin continuous variables    
    # Define number of bins for each variable
    params_to_bin = [['weight', 5, False],
                     ['height', 5, False], 
                     ['bmi', 5, False],
                     ['age', 5]]
    for p2Bin in params_to_bin:
        df[p2Bin[0]] = pd.cut(df[p2Bin[0]], p2Bin[1])
        
    # Create combos of interset parameter source list 
    # (i.e., remove observation and outcome parameters)
    ignore_list = ['sub_id', 'fracture', 'momfrac']
    combos_src_list = df.columns
    for item in ignore_list:
        combos_src_list = combos_src_list[combos_src_list != item]
        
    # Find all combos of specified length from list
    combos_list = all_combos_of_size(combos_src_list, 1)
      
    # Build a frequency count pivot table with each unique combination as the rows, 
    # saving each pivot table into a list of pivot tables
    pTableList = []
    pTableKeyList = []
    for combo in combos_list:
        # Create key and append to pTableKeyList
        pTableKeyList.append('-'.join(combo))
        # Create pTable and append to pTableList
        pTableList.append(df.pivot_table(rows=combo, cols=['momfrac', 'fracture'], \
                          values=['sub_id'], aggfunc=count, fill_value=0, \
                          margins=False))
    
    # Concatenate pivot tables
    completeTable = pd.concat(pTableList, axis=0, keys=pTableKeyList)
        
    # Add Total Count column
    # [Note: We do note set margins to get total counts because we end up with 
    # total rows scattered throughout the concatenated completeTable]
    completeTable['Total'] = completeTable.sum(axis=1)
    
    # Add Prediction Accuracy column
    completeTable['Pred Acc'] = \
        (completeTable['sub_id', 1, 1] + completeTable['sub_id', 0, 0]) / completeTable['Total'] 
        
    # Add "By-Chance" Accuracy column
    completeTable['By-Chance Acc'] = \
        ((completeTable['sub_id', 1, 1] + completeTable['sub_id', 1, 0]) / completeTable['Total']) ** 2 + \
        ((completeTable['sub_id', 0, 0] + completeTable['sub_id', 0, 1]) / completeTable['Total']) ** 2
    
    # Add "By-Chance" Criteria column (i.e., "By-Chance" Accuracy * 1.25, 
    # meaning performance should be at least 25% greater than "By-Chance")
    completeTable['By-Chance Crit'] = completeTable['By-Chance Acc'] * 1.25
    # Limit "By-Chance" criteria to max of 1.0 since you can't have more than 100% accuracy
    completeTable['By-Chance Crit'][completeTable['By-Chance Crit'] >= 1.0] = 1.0
        
    # Add Total Count row
    # [Note: We do note set margins to get total counts because we end up with 
    # total rows scattered throughout the concatenated completeTable]
    total_row = completeTable.sum(axis=0).to_frame().T
    total_row.index = ['Total']
    completeTable = pd.concat([completeTable, total_row])
        
    # Add percentage count columns
    completeTable['0-0_pct'] = \
        completeTable['sub_id', 0, 0] / (completeTable['sub_id', 0, 0] + completeTable['sub_id', 0, 1])
    completeTable['0-1_pct'] = \
        completeTable['sub_id', 0, 1] / (completeTable['sub_id', 0, 0] + completeTable['sub_id', 0, 1])
    completeTable['1-0_pct'] = \
        completeTable['sub_id', 1, 0] / (completeTable['sub_id', 1, 0] + completeTable['sub_id', 1, 1])
    completeTable['1-1_pct'] = \
        completeTable['sub_id', 1, 1] / (completeTable['sub_id', 1, 0] + completeTable['sub_id', 1, 1])
        
    # Replace all inf values in completeTable DataFrame 
    completeTable = completeTable.replace(np.inf, np.nan) # Replace Inf with NaN
#    completeTable = completeTable.replace(np.inf, 0) # Replace Inf with 0
    
    # Calculate descriptive/summary statistics minus the total row
    summary_stats = completeTable.ix[:-1].describe()
    
    test = 1
    
    # View the head of the completeTable
#    print 'Head: '
#    print completeTable.head()
    # View the tail of the completeTable
#    print 'Tail: '
#    print completeTable.tail()
    # View the Descriptive/Summary statistics
#    print 'Summary Stats: '
#    print summary_stats
    
    test = 1
    
    # Plot TPR and TNR as well as their respective 25th percentiles on single plot
    # (Remember: 25th percentile => 75% of slices meet this level of performance)
    
    # Create figure
    fig_h = plt.figure()
    
    # Add plotting axis
    ax_h = fig_h.add_subplot(1, 1, 1)
    ax_h.grid(True)
        
    # Plot the data
        
    # TPR and TPR 25th percentile
    ax_h.scatter(range(1, len(completeTable.index)), completeTable['1-1_pct'][:-1], s=20, c='g', label='TP')
    ax_h.axhline(summary_stats['1-1_pct'].ix['25%'], lw=2, c='g', \
        label='TP 25th-Pct: '+'{0:.3f}'.format(summary_stats['1-1_pct'].ix['25%']))
    
    # FPR [Note: Creates lots of clutter/business on plot]
    # ax_h.scatter(range(1, len(completeTable.index)), completeTable['0-1_pct'][:-1], s=20, c='r', label='FPR')
        
    # TNR and TNR 25th percentile
    ax_h.scatter(range(1, len(completeTable.index)), completeTable['0-0_pct'][:-1], s=20, c='b', label='TN')
    ax_h.axhline(summary_stats['0-0_pct'].ix['25%'], lw=2, c='b', \
        label='TN 25th-Pct: '+'{0:.3f}'.format(summary_stats['0-0_pct'].ix['25%']))    
        
    # FNR [Note: Creates lots of clutter/business on plot]
    # ax_h.scatter(range(1, len(completeTable.index)), completeTable['1-0_pct'][:-1], s=20, c='r', label='FNR')
    
    # Label axes
    ax_h.axes.set_xlabel('Slice')
    ax_h.axes.set_ylabel('Percent')    
    
    # Turn off x-axis tick properties since they have no meaning
    # (i.e., It's just an index that maps to a slice)
    ax_h.tick_params(\
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off') # labels along the bottom edge are off
    
    # Set y-axis properties
    ax_h.axes.set_yticks(np.arange(0, 1.2, 0.2))
    ax_h.axes.set_ylim([-0.1, 1.1])
       
    # Add plot legend
    ax_h.legend(loc='best')

    # Add plot title        
    ax_h.axes.set_title('TPR and TNR Performance Across Parameter Slices')
        
    # Set figure title
    fig_h.canvas.set_window_title('TPR and TNR Performance Across Parameter Slices')
    fig_h.show()
    
    # Save figure
    
    # Copy figure to PowerPoint
    
    test = 1
    
    # Plot TPR and TNR as well as their respective 25th percentiles as two subplots
    
    # Create figure
    fig_h2 = plt.figure()

    # Set space around subplots 
    # [Note: May want to consider using tight layout]
    fig_h2.subplots_adjust(hspace=0.5, wspace=0.5)
#    fig_h2.tight_layout()
        
    # Add plotting axes for subplots
    ax_h_tpr = fig_h2.add_subplot(2, 1, 1)
    ax_h_tpr.grid(True)
    ax_h_tnr = fig_h2.add_subplot(2, 1, 2)
    ax_h_tnr.grid(True)
        
    # Plot the data
        
    # TPR and TPR 25th percentile
    ax_h_tpr.scatter(range(1, len(completeTable.index)), completeTable['1-1_pct'][:-1], s=20, c='g', label='TP')
    ax_h_tpr.axhline(summary_stats['1-1_pct'].ix['25%'], lw=2, c='g', \
        label='TP 25th-Pct: '+'{0:.3f}'.format(summary_stats['1-1_pct'].ix['25%']))
    
    # FPR
#    ax_h_tpr.scatter(range(1, len(completeTable.index)), completeTable['0-1_pct'][:-1], s=20, c='r', label='FP')
    
    # TNR and TNR 25th percentile
    ax_h_tnr.scatter(range(1, len(completeTable.index)), completeTable['0-0_pct'][:-1], s=20, c='b', label='TN')
    ax_h_tnr.axhline(summary_stats['0-0_pct'].ix['25%'], lw=2, c='b', \
        label='TN 25th-Pct: '+'{0:.3f}'.format(summary_stats['0-0_pct'].ix['25%']))
    
    # FNR
#    ax_h_tnr.scatter(range(1, len(completeTable.index)), completeTable['1-0_pct'][:-1], s=20, c='r', label='FN')
    
    # Add axes labels to each subplot
    ax_h_tpr.axes.set_xlabel('Slice')
    ax_h_tpr.axes.set_ylabel('Percent')
    
    ax_h_tnr.axes.set_xlabel('Slice')
    ax_h_tnr.axes.set_ylabel('Percent')    
    
    # Turn off x-axes tick properties since they have not valuable meaning
    ax_h_tpr.tick_params(\
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off') # labels along the bottom edge are off
    ax_h_tnr.tick_params(\
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off') # labels along the bottom edge are off
    
    # Set y-axes properties for each subplot
    ax_h_tpr.axes.set_yticks(np.arange(0, 1.2, 0.2))
    ax_h_tpr.axes.set_ylim([-0.1, 1.1])
    
    ax_h_tnr.axes.set_yticks(np.arange(0, 1.2, 0.2))
    ax_h_tnr.axes.set_ylim([-0.1, 1.1])
       
    # Add legend to each subplot
    ax_h_tpr.legend(loc='best')
    ax_h_tnr.legend(loc='best')
        
    # Add plot title
    ax_h_tpr.axes.set_title('Performance Across Parameter Slices')
    #ax_h_tnr.axes.set_title('TNR Performance Across Parameter Slices')
        
    # Set figure title
    fig_h2.canvas.set_window_title('TPR and TNR Performance Across Parameter Slices')
    fig_h2.show()
        
    # Save figure
    
    # Copy figure to PowerPoint
    
    
########## ----- Main Function ----- ##########

# Define function to filter dataframe
def filter_DataFrame(df, col_name, col_levels, remove=True):
    '''
    ----
    filter_DataFrame(DataFrame, String, List of Items or Single Item) -> DataFrame

    Description:
        Returns the DataFrame removing all the specified items or keeping only 
        the single specified item
        
    Inputs:
        df          -   DataFrame
        col_name    -   String
        col_levels  -   List or Single Item
        remove      -   Boolean (Optional, default is set to True)
        
    Outputs:
        df          -   DataFrame
        
    Example 1:
        >>> df.head(10)
                   sub_id  site_id  priorfrac           age           weight      height  \
        0       1        1          0  (54.965, 62]   (57.32, 74.74]  (147, 160]   
        1       2        4          0      (62, 69]   (74.74, 92.16]  (147, 160]   
        2       3        6          1      (83, 90]  (39.813, 57.32]  (147, 160]   
        3       4        6          0      (76, 83]   (57.32, 74.74]  (147, 160]   
        4       5        1          0  (54.965, 62]   (57.32, 74.74]  (147, 160]   
        5       6        5          1      (62, 69]   (57.32, 74.74]  (160, 173]   
        6       7        5          0      (83, 90]  (39.813, 57.32]  (147, 160]   
        7       8        1          1      (76, 83]  (39.813, 57.32]  (147, 160]   
        8       9        1          1      (83, 90]   (57.32, 74.74]  (147, 160]   
        9      10        4          0  (54.965, 62]   (57.32, 74.74]  (160, 173]   
        
                        bmi  premeno  momfrac  armassist  smoke  raterisk  fracscore  \
        0  (21.718, 28.559]        0        0          0      0         2          1   
        1    (28.559, 35.4]        0        0          0      0         2          2   
        2  (14.842, 21.718]        0        1          1      0         1         11   
        3  (21.718, 28.559]        0        0          0      0         1          5   
        4    (28.559, 35.4]        0        0          0      0         2          1   
        5  (21.718, 28.559]        0        0          0      1         2          4   
        6  (21.718, 28.559]        0        0          0      0         1          6   
        7  (14.842, 21.718]        0        0          0      0         2          7   
        8  (21.718, 28.559]        0        0          0      0         2          7   
        9  (21.718, 28.559]        0        0          0      0         1          0   
        
           fracture  
        0         0  
        1         0  
        2         0  
        3         0  
        4         0  
        5         0  
        6         0  
        7         0  
        8         0  
        9         0  
        >>> filter_DataFrame(df, 'fracscore', [7, 11], remove=True)
           sub_id  site_id  priorfrac           age           weight      height  \
        0       1        1          0  (54.965, 62]   (57.32, 74.74]  (147, 160]   
        1       2        4          0      (62, 69]   (74.74, 92.16]  (147, 160]   
        2       4        6          0      (76, 83]   (57.32, 74.74]  (147, 160]   
        3       5        1          0  (54.965, 62]   (57.32, 74.74]  (147, 160]   
        4       6        5          1      (62, 69]   (57.32, 74.74]  (160, 173]   
        5       7        5          0      (83, 90]  (39.813, 57.32]  (147, 160]   
        6      10        4          0  (54.965, 62]   (57.32, 74.74]  (160, 173]   
        7      11        6          0      (62, 69]   (57.32, 74.74]  (147, 160]   
        8      12        1          0  (54.965, 62]    (109.58, 127]  (160, 173]   
        9      13        6          0  (54.965, 62]   (57.32, 74.74]  (160, 173]   
        
                         bmi  premeno  momfrac  armassist  smoke  raterisk  fracscore  \
        0   (21.718, 28.559]        0        0          0      0         2          1   
        1     (28.559, 35.4]        0        0          0      0         2          2   
        2   (21.718, 28.559]        0        0          0      0         1          5   
        3     (28.559, 35.4]        0        0          0      0         2          1   
        4   (21.718, 28.559]        0        0          0      1         2          4   
        5   (21.718, 28.559]        0        0          0      0         1          6   
        6   (21.718, 28.559]        0        0          0      0         1          0   
        7     (28.559, 35.4]        0        1          0      1         1          4   
        8  (42.241, 49.0824]        0        0          1      1         2          3   
        9   (21.718, 28.559]        0        0          0      1         1          1   
        
           fracture  
        0         0  
        1         0  
        2         0  
        3         0  
        4         0  
        5         0  
        6         0  
        7         0  
        8         0  
        9         0  
    Example 2:
        >>> filter_DataTrame(df, 'fracscore', 11, remove=False)
        sub_id  site_id  priorfrac       age           weight      height  \
        0       3        6          1  (83, 90]  (39.813, 57.32]  (147, 160]   
        1     169        1          1  (83, 90]  (39.813, 57.32]  (160, 173]   
        2     179        3          1  (83, 90]  (39.813, 57.32]  (147, 160]   
        
                        bmi  premeno  momfrac  armassist  smoke  raterisk  fracscore  \
        0  (14.842, 21.718]        0        1          1      0         1         11   
        1  (14.842, 21.718]        0        0          1      1         1         11   
        2  (14.842, 21.718]        0        1          1      0         3         11   
        
           fracture  
        0         0  
        1         0  
        2         0 
        
    ----
    
    Author:
        Curtis Neiderer, 4/2014
    '''
    if remove:
        for level in col_levels:
            df = df[df[col_name] != level]
    else:
        df = df[df[col_name] == col_levels]
    df.index = range(len(df))
    return df
    
# Define custom combination functions
def all_combos(full_list):
    '''
    ----
    
    '''
    combos = []
    for i in xrange(1, len(full_list)+1):
        els = [list(x) for x in itertools.combinations(full_list, i)]
        combos.extend(els)
    return combos    
    
def all_combos_size_limited(full_list, size_limit):
    '''
    ----
    
    '''
    combos = []
    for i in xrange(1, size_limit + 1):
        els = [list(x) for x in itertools.combinations(full_list, i)]
        combos.extend(els)
    return combos
    
def all_combos_of_size(full_list, size_limit):
    '''
    ----
    
    '''
    return [list(x) for x in itertools.combinations(full_list, size_limit)]
    
def all_combos_of_interest(full_list, of_interest_list):
    '''
    ----
    
    '''
    combos = []
    for i in xrange(1, len(full_list)+1):
        els = [list(x) for x in itertools.combinations(full_list, i)]
        combos.extend(els)    
    return prune_combos(combos, of_interest_list)
    
def all_combos_of_interest_size_limited(full_list, size_limit, of_interest_list):
    '''
    ----
    
    '''
    combos = [list(x) for x in itertools.combinations(full_list, size_limit)]
    return prune_combos(combos, of_interest_list)
    
def all_combos_of_interest_of_size(full_list, of_size, of_interest_list):
    '''
    ----
    
    '''
    combos = [list(x) for x in itertools.combinations(full_list, of_size)]
    return prune_combos(combos, of_interest_list)

# Check if combination contains parameter of interest
def any_in(test_list, full_list):
    '''
    ----
    any_in(List, List) -> Boolean
    
    Description:
        Checks to see if any items from test list are within the full list
        
    Inputs:
        test_list   -   List of parameters
        full_list   -   List of parameters
    
    Outputs:
        Boolean
        
    Example:
        >>>
        
    ----
    
    Author:
        Curtis Neiderer, 4/2014
    '''
    for element in test_list:
        if element in full_list:
            return True 
    return False
    
# Remove combinations that don't contain parameters of interest 
def prune_combos(combos, of_interest_list):
    '''
    ----
    prune_combos(List, List) -> List
    
    Description:
        Removes combinations from combos list that do not contain parameters of interest
        
    Inputs:
        combos              -   Full list of combos   
        of_interest_list    -   List of parameters of interest
    
    Outputs:
        pruned_combos       -   List of combos containing parameters of interest
        
    Example:
        >>>
    ----
    
    Author:
        Curtis Neiderer, 4/2014
    '''
    pruned_combos = []
    for combo_list in combos:
        if any_in(of_interest_list, combo_list):
            pruned_combos.append(combo_list)
    return pruned_combos

# Define custom aggregation functions
def freq_count(x):
    '''
    ----
    freq_count(Item) -> Integer
    
    Description:
        Custom aggregation function to count the frequency count for each bin based on the data slicing
        
    Inputs:
        x   -
        
    Outputs:
        ?
        
    Example:
        >>>
        
    ----
    
    Author:
        Curtis Neiderer, 4/2014
    '''
    return x.value_counts().count()
#    return len(x)

    
########## ----- Enables Command Line Call ----- ##########
if __name__ == '__main__':
    main()
########## ----- Enables Command Line Call ----- ##########