# -*- coding: utf-8 -*-
"""
Created on Fri Mar 07 10:06:34 2014

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
    '''
    ----
    
    main() -> None
    
    Description:
        Tests the functionality/results of various residual outlier analysis tests
    
    Dependencies:
        __future__,
        pandas,
        statsmodels,
        numpy,
        matplotlib,
        sklearn
        
    ----
    
    Author:
        Curtis Neiderer, 3/2014
    '''
    
    # read the data in
    df = pd.read_csv("http://www.ats.ucla.edu/stat/data/binary.csv")

    # Dataset summary
#    print df.head(), '\n'
    
    # Rename 'rank' column to prestige since there's a DataFrame method called 'rank'
    df.columns = ['admit', 'gre', 'gpa', 'prestige']
    
    # Dummify rank
    dummy_ranks = pd.get_dummies(df['prestige'], prefix='prestige')
#    print dummy_ranks.head(), '\n'
    
    # Create a clean data frame for regression
    cols_to_keep = ['admit', 'gre', 'gpa']
    data = df[cols_to_keep].join(dummy_ranks.ix[:, 'prestige_2':])
#    print data.head(), '\n'
    
    # Manually add the intercept
    data['intercept'] = 1.0
#    print data.head(), '\n'
#    return data
    
    # Select data columns
    train_cols = data.columns[1:]
    
    # Calculate logic of data columns
    logit = sm.Logit(data['admit'], data[train_cols])
    
    # Fit the model
    result = logit.fit()
#    return result
    
    # Print the results
    print result.summary(), '\n'
    
    # Calculate standardized residuals
    standardRes = calculateStandardizedRes(result.resid_dev, result.predict())
#    print 'Residuals: ', result.resid_dev
#    print 'Standardized Residuals: ', standardRes
    
    # TODO: Create function to plot residuals (Should be able to use with any type of residuals)
#    plotStandardizedResiduals(result.resid)

    # Calculate the MLE residual variance
    mleResVar = calculateMLEResidualVar(result.resid_dev, result.df_resid)
#    print 'MLE Residual Variance: ', mleResVar

    # Calculate hat-matrix (leverage statistic)
    leverage, hatMatrix = calculateHatMatrix(data[data.columns[data.columns != 'admit']], result.cov_params())
#    print 'Leverage: ', leverage
#    print 'Hat Matrix: ', hatMatrix

    # Calculate studentized residuals
    studentRes = calculateStudentizedResiduals(result.resid_dev, mleResVar, leverage)
#    print 'Studentized Residuals: ', studentRes
    
#    plotStudentizedResiduals(result.resid)

    # Calculate Cook distances
    cooksD = calculateCooksDistance(result.df_model, studentRes, leverage)
#    print 'Cook''s Distance: ', cooksD
    
    # Calculate Delta-Beta Statistics (DBetas)
    dBetaStat = calculateDBetaStat(result.resid_pearson, leverage)
#    print 'Delta-Beta Statistic: ', dBetaStat
    
    # Calculate Standardized Pearson Residuals
    standardPearsonRes = calculateStandardPearsonRes(result.resid_pearson, leverage)
#    print 'Pearson Residuals: ', result.resid_pearson
#    print 'Standardized Pearson Residuals: ', standardPearsonRes
    
    # Calculate Standardized Delta-Beta Statistics (DBetas)
    standardDBetaStat = calculateDBetaStat(standardPearsonRes, leverage, fromSM=0)    
#    print 'Standardized Delta-Beta Statistic: ', standardDBetaStat
    
    
########## ----- Main Function ----- ##########
    
class ResidualOutlierAnalysisResult:
    '''
    ----
    
    ResidualOutlierAnalysisResult
    
    Description:
        A container class for all the residual outlier analysis results
    
    Inputs:
    
    Outputs:
    
    Reference / Notes:
    
    Author:
        Curtis Neiderer, 3/2014
    
    '''
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    
    def add_data(self, **kwargs):
        self.__dict__.update(kwargs)
        
        
def calculateStandardPearsonRes(pearsonRes, leverage, fromSM=True):
    '''
    ----
    
    calculateStandardPearsonRes(pearsonRes, leverage, fromSM) -> standardPearsonRes

    Description:
        Calculates standardized Pearson Residuals
        
    Inputs:
        pearsonRes  -   Array of Pearson Residuals
        leverage    -   Array of leverage values (i.e., the hii diagonal from the hat matrix)
        fromSM      -   Boolean flag to designate if the data is coming from a statsmodels DataFrame
        
    Outputs:
        standardPearsonRes  -   Array of standardized Pearson Residuals
        
    Example:
        >>> leverage, hatMatrix = calculateHatMatrix(data[data.columns[data.columns != 'admit']], result.cov_params(), fromSM=True)
        >>> leverage
        0     0.112874
        1     0.052771
        2     0.121151
        ...
        397    0.092921
        398    0.041673
        399    0.063458
        Length: 400, dtype: float64
        >>> result.resid_pearson
        0    -0.456776
        1     1.556473
        2     0.595201
        ...
        397   -0.470206
        398   -0.929793
        399   -0.655793
        Length: 400, dtype: float64
        >>> standardPearsonRes = calculateStandardPearsonRes(result.resid_pearson, leverage, fromSM=True)
        >>> standardPearsonRes
        0    -0.484965
        1     1.599241
        2     0.634902
        ...
        397   -0.493703
        398   -0.949794
        399   -0.677646
        Length: 400, dtype: float64
    
    ----
    
    Reference / Notes:
        
        
    ----
    
    Author:
        Curtis Neiderer, 3/2014
    '''   
#    if fromSM==True:
#        pearsonRes = np.array([pearsonRes]).T
        
    standardPearsonRes = pearsonRes / np.sqrt(1 - leverage)

    return standardPearsonRes    
    
def calculateDBetaStat(pearsonRes, leverage, fromSM=True):
    '''
    ----
    
    calculateDBetaStat(pearsonRes, leverage, fromSM) -> dBeta
    
    Description:
        Calculates delta beta (deletion displacement) statistic

    Inputs:
        pearsonRes  -   Array of Pearson Residuals
        leverage    -   Array of leverage values (i.e., the hii diagonal from the hat matrix)
        fromSM      -   Boolean flag to designate if the data is coming from a statsmodels DataFrame  

    Outputs:
        dBeta       -   Array of delta-beta statistic values

    Example:
        >>> leverage, hatMatrix = calculateHatMatrix(data[data.columns[data.columns != 'admit']], result.cov_params(), fromSM=True)
        >>> leverage
        0     0.112874
        1     0.052771
        2     0.121151
        ...
        397    0.092921
        398    0.041673
        399    0.063458
        Length: 400, dtype: float64
        >>> result.resid_pearson
        0    -0.456776
        1     1.556473
        2     0.595201
        ...
        397   -0.470206
        398   -0.929793
        399   -0.655793
        Length: 400, dtype: float64
        >>> dBetaStat = calculateDBetaStat(result.resid_pearson, leverage, fromSM=True)
        >>> dBetaStat
        0     0.026547
        1     0.134966
        2     0.048836
        ...
        397    0.022649
        398    0.037593
        399    0.029140
        Length: 400, dtype: float64
                
    ----
    
    Reference / Notes:
        

    ----
    
    Author:
        Curtis Neiderer, 3/2014
    '''    
#    if fromSM==True:
#        pearsonRes = np.array([pearsonRes]).T
        
    dBetaStat = ((pearsonRes ** 2) * leverage) / (1 - leverage)
    
    return dBetaStat
    
def calculateCooksDistance(dofRegress, studentRes, leverage):
    '''
    ----    
    
    calculateCooksDistance(dofRegress, studentRes, leverage) -> cooksD
    
    Description:
        Calculates Cook's distance for the residual errors

    Inputs:
        dofRegress  -   Degrees of freedom due to regression model
        studentRes  -   Array of studentized residual errors
        leverage    -   Array of leverage values (i.e., the hii diagonal from the hat matrix)

    Outputs:
        cooksD      -   Array of Cook's Distances for each residual

    Example:
        >>> leverage, hatMatrix = calculateHatMatrix(data[data.columns[data.columns != 'admit']], result.cov_params())
        >>> leverage
        0     0.112874
        1     0.052771
        2     0.121151
        ...
        397    0.092921
        398    0.041673
        399    0.063458
        Length: 400, dtype: float64
        >>> studentRes = calculateStudentizedResiduals(result.resid_dev, mleResVar, leverage)
        >>> studentRes
        0    -0.605893
        1     1.494106
        2     0.770077
        ...
        397   -0.615181
        398   -1.056994
        399   -0.810201
        Length: 400, dtype: float64
        >>> cooksD = calculateCooksDistance(result.df_model, studentRes, leverage)
        >>> cooksD
        0     0.007785
        1     0.020728
        2     0.013625
        ...
        397    0.006461
        398    0.008097
        399    0.007413
        Length: 400, dtype: float64

    ----
    
    Reference / Notes:
        
        
    ----
    
    Author:
        Curtis Neiderer, 2/2014
    '''
    cooksD = (studentRes ** 2) * leverage / ((dofRegress + 1) * (1 - leverage))

    return cooksD
    
def calculateStudentizedResiduals(residuals, mleResVar, leverage, fromSM=True):
    '''
    ----
    
    calculateStudentRes(residuals, mleResVar, leverage, fromSM) -> studentizedRes
    
    Description:
        Calculates the studentized residual errors

    Inputs:
        residuals   -   Array of residual errors of dependent variable
        mleVar      -   Variance of dependent variable
        leverage    -   Array of leverage values (i.e., the hii diagonal from the hat matrix)
        fromSM      -   Boolean flag to designate if the data is coming from a statsmodels DataFrame  

    Outputs:
        studentizedRes  -   Array of studentized residuals

    Example:
        >>> mleResVar = calculateMLEResidualVar(result.resid_dev, result.df_resid)
        >>> mleResVar
        1.163749980903297
        >>> leverage, hatMatrix = calculateHatMatrix(data[data.columns[data.columns != 'admit']], result.cov_params(), fromSM=True)
        >>> leverage
        0     0.112874
        1     0.052771
        2     0.121151
        ...
        397    0.092921
        398    0.041673
        399    0.063458
        Length: 400, dtype: float64
        >>> studentRes = calculateStudentizedResiduals(result.resid_dev, mleResVar, leverage, fromSM=True)
        >>> studentRes
        0    -0.605893
        1     1.494106
        2     0.770077
        ...
        397   -0.615181
        398   -1.056994
        399   -0.810201
        Length: 400, dtype: float64
        
    ----
    
    Reference / Notes:
        
        
    ----
    
    Author:
        Curtis Neiderer, 2/2014
    '''
#    if fromSM == True:
#        residuals = np.array([residuals]).T
        
    studentizedRes = residuals / np.sqrt(mleResVar * (1 - leverage))

    return studentizedRes
    
def  calculateHatMatrix(X, covMatrix, fromSM=True):
    '''
    ----
    
    calculateHatMatrix(X, covMatrix, fromSM) -> leverage, hatMatrix
    
    Description
        Calculates the hat matrix for leverage comparison

    Inputs:
        X           -   Array of independent variables
        covMatrix   -   DataFrame representing the covariance matrix for independent variables
        fromSM      -   Boolean flag to designate if the data is coming from a statsmodels DataFrame 

    Outputs:
        hatMatrix   -   Array representing the Hat Matrix
        leverage    -   Array representing the diagonal elements of the Hat Matrix

    Example:
        >>> leverage, hatMatrix = calculateHatMatrix(data[data.columns[data.columns != 'admit']], result.cov_params(), fromSM=True)
        >>> leverage
        0     0.112874
        1     0.052771
        2     0.121151
        ...
        397    0.092921
        398    0.041673
        399    0.063458
        Length: 400, dtype: float64
        >>> hatMatrix
        matrix([[ 0.11287417,  0.0377949 , -0.03207805, ...,  0.01165793, -0.01670422,  0.06253614],
        [ 0.0377949 ,  0.05277119,  0.01483564, ..., -0.01445022, 0.00709978,  0.05165675],
        [-0.03207805,  0.01483564,  0.12115068, ..., -0.05208924, 0.02436066,  0.01349483],
        ..., 
        [ 0.01165793, -0.01445022, -0.05208924, ...,  0.09292062, 0.00595047, -0.02438431],
        [-0.01670422,  0.00709978,  0.02436066, ...,  0.00595047, 0.04167272,  0.00568177],
        [ 0.06253614,  0.05165675,  0.01349483, ..., -0.02438431, 0.00568177,  0.06345822]])        
        
    ----
    
    Reference / Notes:
        

    ----
    
    Author:
        Curtis Neiderer, 2/2014
    '''
    
#    if fromSM == True:
#        covMatrix = np.matrix(covMatrix)

#    hatMatrix = np.dot(np.dot(X, covMatrix), X.T)
#    leverage = np.array([np.diag(hatMatrix)]).T
    
    try:
        hatMatrix = np.dot(np.dot(X, covMatrix), X.T)
        leverage = np.array([np.diag(hatMatrix)]).T
    except:
        # hatMatrix too big to fit into memory, 
        # break X into pieces to calculate leverage
        hatMatrix = np.nan
        
        # Set chunk size and initialize chunk start and stop
        chunkSize = 500.0
        chunkStart = 0
        chunkStop = int(chunkSize)
    
        # Initialize leverage array
        leverage = np.array([np.nan * np.ones((int(len(X))))]).T
        for ii in range(int(np.ceil(len(X) / chunkSize))):
            xChunk = X[chunkStart:chunkStop]
            hatMatrixChunk = np.dot(np.dot(xChunk, covMatrix), xChunk.T)
            leverageChunk = np.array([np.diag(hatMatrixChunk)]).T
            
            # Record leverage chunk to overall leverage array
            chunkCnt = 0
            for jj in range(chunkStart, chunkStop):
                leverage[jj] = leverageChunk[chunkCnt]
                chunkCnt += 1
            
            # Update chunk start and stop indices
            if chunkStop < len(X):
                chunkStart = chunkStop + 1
                if chunkStop + chunkSize <= len(X):
                    chunkStop = chunkStop + int(chunkSize)
                else:
                    chunkStop = len(X)

    # Convert leverage from Array to Series
    leverage = pd.Series(leverage[:, 0])
    
    return leverage, hatMatrix

def calculateMLEResidualVar(residuals, dofResiduals):
    '''
    ----
    
    calculateMLEVar(residuals, dofResiduals) -> mleResVar
    
    Description:
        Calculates maximum likelihood estimate variance of the error residuals

    Inputs:
        residuals       -   Array of calculated y residual errors
        dofResiduals    -   Integer designating the number of degrees of freedom from residual error

    Outputs:
        mleResVar   -   Float designating the maximum likelihood estimate of the variance of the residual errors

    Example:
        >>> mleResVar = calculateMLEResidualVar(result.resid_dev, result.df_resid)
        1.163749980903297
    
    ----
    
    Reference / Notes:
    
    ----

    Author:
        Curtis Neiderer, 3/2014
    '''
    residuals = np.array(residuals)
    mle = np.sum(residuals ** 2) / dofResiduals

    return mle
    
def calculateStandardizedRes(residuals, predOutcomeProb):
    '''
    ----
    
    calculateStandardizedResiduals(residuals, predOutcomeProb) -> standardizedRes
    
    Description:
        Calculates the standardized error residuals

    Inputs:
        residuals       -   Array of residual errors
        predOutcomeProb -   Array of the predicted outcome probabilities 

    Outputs:
        standardizedRes -   Array of standardized residuals

    Example:
        >>> standardRes = calculateStandardizedRes(result.resid_dev, result.predict())
        >>> standardRes
        0    -1.628973
        1     3.449484
        2     1.771989
        ...
        397   -1.641406
        398   -2.238406
        399   -1.844485
        Length: 400, dtype: float64

    ----
    
    Reference / Notes:
    
    ----
    
    Author:
        Curtis Neiderer, 3/2014
    '''
    standardizedRes = residuals / np.sqrt(predOutcomeProb * (1 - predOutcomeProb))

    return standardizedRes

########### ----- Enables Command Line Call ----- ##########
if __name__ == '__main__':
    main()
########### ----- Enables Command Line Call ----- ##########