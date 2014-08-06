# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 13:19:35 2014

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
from decimal import *
#
import csv
import pickle

########## ----- Main Function ----- ##########
def main():
    '''
    ----
    
    main() -> None
    
    Description:
        Test the functionality/results of the various goodness of fit analysis tests
    
    Dependencies:
        __future__,
        pandas,
        statsmodels,
        numpy,
        matplotlib,
        sklearn
        
    ----
    
    Author:
        Curtis Neiderer, 2/2014
    '''
    # read the data in
    dataFilePath = 'http://www.ats.ucla.edu/stat/data/binary.csv'
#    dataFilePath = 'C:/AnalysisTools/python_toolbox/DataSets/AdultDataSet.csv'
    df = pd.read_csv(dataFilePath)

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
    
    # Select data columns
    train_cols = data.columns[1:]
    
    # Calculate logic of data columns
    logit = sm.Logit(data['admit'], data[train_cols])
    
    # Fit the model
    result = logit.fit()
#    return result
    
    # Print the results
    print result.summary(), '\n'
    
    # Intialize goodness-of-fit container object
    gofEval = GoodnessOfFitEvaluation()    
    
    # Calculate Pseudo R-Squared
    R2m = calculatePseudoRSquared(result.llf, result.llnull, int(result.nobs), 'McFadden')
    R2cs = calculatePseudoRSquared(result.llf, result.llnull, int(result.nobs), 'Cox-Snell')
    R2n = calculatePseudoRSquared(result.llf, result.llnull, int(result.nobs), 'Nagelkerke')
#    print 'McFadden R-Squared: \n', round(R2m, 5)
#    print 'Cox-Snell R-Squared: \n', round(R2cs, 5)
#    print 'Nagelkerke R-Squared: \n', round(R2n, 5), '\n'
    
    # Calculate Hosmer-Lemeshow Test Statistic
    hlStat, hlProb = calculateHLStat(data.admit, result.predict())
#    print 'HL Statistic: \n', round(hlStat, 3)
#    print 'HL Probability: \n', round(hlProb, 3), '\n'
    
    # Calculate By-Chance Accurracy
    classCorrectPct, byChanceCorrectPct, byChanceCriteria, classAccuracySatisfactory = calculateByChanceAccuracy(result.pred_table())
#    print 'Classification Correct Rate: \n' + str(classCorrectPct) + '%'
#    print 'By Chance Classification Correct Criteria: \n' + str(byChanceCorrectPct) + '%'
#    print 'Classification Accuracy Criteria: \n' + classAccuracyCriteria + '\n'
    
    # Create Classification Table
    classTable, sensitivity, specificity, tableComp = createClassTable(result.pred_table()) 
    gofEval.add_data(model_ClassTable=classTable, model_Sensitivity=sensitivity, \
        model_Specificity=specificity, tpCnt=tableComp['tpCnt'], tnCnt=tableComp['tnCnt'], \
        fpCnt=tableComp['fpCnt'], fnCnt=tableComp['fnCnt'])    
#    print 'Classification Table: '
#    print classTable, '\n'
#    print 'Sensitivity: \n' + str(sensitivity)
#    print 'Specificity: \n' + str(specificity), '\n'
    
    # Predict Outcomes at cutpoint
    predOutcome = predOutcomeBasedOnCutpoint(result.predict(), 0.5)
#    print 'Predicted Outcome: '    
#    print predOutcome
    
    # Create Prediction Table
    pred_table = createPredTable(data.admit, result.predict())   
#    print 'Prediction Table: '
#    print pred_table, '\n'
    
    # Calculate the beta coefficient variance inflation factors
    VIF, multiCriteria = calculateVIF(result.cov_params())        
#    print 'VIFs: \n', VIF
#    print multiCriteria, '\n'
    
    regressionEqn = createFormattedRegressionEquation(result.params)
    print regressionEqn
    
    fig_handles = []    
    # Create Histogram of Estimated Outcome Probabilities
    fig_handles.extend(plotHistOfEstProbByOutcomeWithCutoffLine(data.admit, result.predict(), 0.5))
    # Create Jittered Outcome vs Estimated Outcome Probabilities
    fig_handles.append(plotJitteredOutcomeVsEstOutcomeProb(data.admit, result.predict()))
    # Create ROC Curve:
    tmp_h, roc_auc = plotROCCurve(data.admit, result.predict())
    fig_handles.append(tmp_h)
    gofEval.add_data(roc_auc=roc_auc)
#    print 'ROC Area Under Curve: \n', roc_auc
    
#    # Cutpoint Evaluation
#    plotCutpointEvalPlot(data.admit, result.predict())
    
########## ----- Main Function ----- ##########

class GoodnessOfFitEvaluation:
    '''
    ----
    
    GoodnessOfFitEvaluation
    
    Description:
        A container class for all the goodness-of-fit evaluation results
    
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

def createFormattedRegressionEquation(betaCoeff):
    '''
    createFormattedRegressionEquation(betaCoeff) -> eqnString
    
    Description:
        Creates a human readable LSR equation

    Inputs:
        betaCoeff   -   Array of beta coefficients

    Outputs:
        eqnString   -   Formated LSR equation string

    Example:
        >>> regressionEqn = createFormattedRegressionEquation(result)
        >>> regressionEqn 
        

    ----
    
    Reference / Notes:
    
    ----

    Author:
        Curtis Neiderer, 2/2014
    '''

    eqnString = 'y = ' + str('%.6f' % betaCoeff['intercept'])    
#    eqnString = 'y = ' + str(round(betaCoeff[0, 0], 5))
    betaCoeff = betaCoeff[betaCoeff.index != 'intercept']
    for ii, coeff in enumerate(betaCoeff[betaCoeff.index != 'intercept']):
        coeff = round(coeff, 6)
        if coeff == 0.0:
            continue
        elif (coeff < 0):
            sign = ' - '
        else:
            sign = ' + '
            
        eqnString += sign + str(abs(coeff)) + '*' + betaCoeff.index[ii]

    return eqnString
    
def calculateVIF(covMatrix, fromSM=True, vifCritical=5):
    '''
    ----
    
    calculateVIF(covMatrx, fromSM) -> VIF, multicollinearityCriteria
    
    Description:
        Calculates the variance inflation factors for the beta coefficients

    Inputs:
        covMatrix   -   Array representing the covariance matrix of the independent variables
        fromSM      -   Boolean flag to designate if the data is coming from a statsmodels DataFrame 

    Outputs:
        VIF                         -   Series of variance inflation factors for the beta coefficients
        multicollinearityCriteria   -   String statement of whether or not multicollinearity is a problem

    Example:
        >>> VIF, multiCriteria = calculateVIF(result.cov_params())
        >>> VIF
        array([['gre', -0.0],
        ['gpa', -0.00107],
        ['prestige_2', -0.03097],
        ['prestige_3', -0.0509],
        ['prestige_4', -0.11645],
        ['intercept', -0.01302]], dtype=object)
        >>> multiCriteria
        'No VIFs > 5, no multicollinearity problems'
    
    ----
    
    Reference / Notes:
    
    ----

    Author:
        Curtis Neiderer, 2/2014
    '''
    # Calculate the correlation matrix based on input source
    if fromSM:
        corrMatrix = np.linalg.inv(np.matrix(covMatrix))
    else:
        corrMatrix = np.linalg.inv(covMatrix)

    # Calculate beta coefficient VIFs
    paramNames = np.array([covMatrix.columns]).T
    VIF = np.ones((np.shape(corrMatrix)[0], 1))
    for ii in range(np.shape(corrMatrix)[0]):
        VIF[ii, 0] = 1 / (1 - corrMatrix[ii, ii])
        
    # Concatenate names and values into one array
#    VIF = np.concatenate((paramNames, VIF), axis=1)
    # Convert names and values from two arrays into one Series
    VIF = pd.Series(VIF[:, 0], index=paramNames[:, 0])
#    VIF.index = paramNames[:, 0]
    
    # Check for multicollinearity
#    vifCritical = 5
    if np.any(VIF > vifCritical):
#    if np.any(VIF[:, 1] > vifCritical):
        # Find VIF problem parameters
        vifProblem = VIF.index[VIF > vifCritical]
        # Build problem parameter string
        hiVIFStr = ''
        for jj, vif in enumerate(vifProblem):
            if jj == len(vifProblem) - 1:
                hiVIFStr += 'and ' + vif + ' each have a VIF > ' + str(vifCritical)
            else:
                hiVIFStr += vif + ', '       
        multicollinearityCriteria = hiVIFStr + '; therefore, multicollinearity is a problem'
    else:
        multicollinearityCriteria = 'No VIFs > ' + str(vifCritical) + \
            '; therefore, no multicollinearity problems'

    return VIF, multicollinearityCriteria
    
def calculatePseudoRSquared(llFull, llNull, numObs, r2Name='McFadden'):
    '''
    ----
    
    calculatePseudoRSquared(llFull, llNull, numObs) -> R2_McFadden, R2_CoxSnell, R2_Nagelkerke

    Description:
        Calculates the pseudo R-squared values
        
    Inputs:
        llFull  -   Float representing the log-likelihood of full regression model
        llNull  -   Float representing the log-likelihood of the null regression model (i.e., intercept only, no independent parameters)
        numObs  -   Integer representing the number of observations used to build the model
        
    Outputs:
        R2_McFadden     -   Float
        R2_CoxSnell     -   Float
        R2_Nagelkerke   -   Float
        
    Example:
        >>> R2m = calculatePseudoRSquared(result.llf, result.llnull, int(result.nobs), 'McFadden')
        >>> R2m
        0.0829219445781
        >>> R2cs = calculatePseudoRSquared(result.llf, result.llnull, int(result.nobs), 'Cox-Snell')
        >>> R2cs
        0.098457021188
        >>> R2n = calculatePseudoRSquared(result.llf, result.llnull, int(result.nobs), 'Nagelkerke')
        >>> R2n
        0.13799580131 
        
    ----
    
    Reference / Notes:
        
        
    ----
    
    Author:
        Curtis Neiderer, 3/2014
    ''' 

    # Calculate McFadden R-Squared
    R2_McFadden = 1 - (llFull / llNull)
    # Calculate Cox-Snell R-Squared
    R2_CoxSnell = 1 - np.exp(-(2 / numObs) * (llFull - llNull))
    # Calculate Nagelkerke R-Squared
    R2_Max = 1 - np.exp((2 / numObs) * llNull)
    R2_Nagelkerke = R2_CoxSnell / R2_Max
    
    # Return requested R-squared
    if r2Name == 'McFadden':
        return R2_McFadden
    elif r2Name == 'Cox-Snell':
        return R2_CoxSnell
    elif r2Name == 'Nagelkerke':
        return R2_Nagelkerke 
    else:
        # Throw exception, ValueError: paramsOfInterestFlag must be True or False
        raise ValueError(r2Name + ' does not match any of the pseudo R-squared values that can be calculated')    
    
def createPredTable(obsOutcome, predOutcomeProb):
    '''
    ----

    createPredTable(obsOutcome, predOutcomeProb) -> predTable

    Description:
        Creates a prediction table using a cutpoint of 0.5 for the classification decision

    Inputs:
        obsOutcome      -   Array of observed outcomes (i.e., array of binary outcomes)
        predOutcomeProb -   Array of predicted outcome probabilities 

    Outputs:
        predTable   -   Array representing the prediction table based on a classification cutpoint of 0.5

    Example:
        >>> pred_table = createPredTable(data.admit, result.predict())
        >>> pred_table
        array([[ 30,  97],
        [ 19, 254]])
        
    ----
    
    Reference / Notes:

    ----

    Author:
        Curtis Neiderer, 2/2014
        
    '''        
    cutpt = 0.5        
    predOutcome = np.ones(np.shape(predOutcomeProb)) * np.nan
    inds = np.arange(0, len(predOutcomeProb), 1)
    predOutcome[inds[predOutcomeProb >= cutpt]] = np.ones(len(inds[predOutcomeProb >= cutpt]))
    predOutcome[inds[predOutcomeProb < cutpt]] = np.zeros(len(inds[predOutcomeProb < cutpt]))
    
    truePosCnt = 0
    falsePosCnt = 0
    trueNegCnt = 0
    falseNegCnt = 0
    for ii in range(len(obsOutcome)):
        if obsOutcome[ii] == 1 and predOutcome[ii] == 1:
            truePosCnt += 1
        elif obsOutcome[ii] == 1 and predOutcome[ii] == 0:
            falsePosCnt += 1
        elif obsOutcome[ii] == 0 and predOutcome[ii] == 0:
            trueNegCnt += 1
        elif obsOutcome[ii] == 0 and predOutcome[ii] == 1:
            falseNegCnt += 1
            
    pred_table = np.array([[truePosCnt, falsePosCnt],
                               [falseNegCnt, trueNegCnt]])
                               
    return pred_table
        
def predOutcomeBasedOnCutpoint(predOutcomeProb, cutpt):
    '''
    ----

    predOutcomeBasedOnCutpoint(predOutcomeProb, cutpt) -> predOutcome

    Description:
        Calculates the predicted outcome from the predicted outcome probabilities

    Inputs:
        predOutcomeProb -   Array of predicted outcome probabilities 
        cutpt           -   Float designating the cutpoint for classification decisions (0 <= cutpt <= 1)

    Outputs:
        predOutcome -   Array of predicted outcomes (i.e., binary outcome array)

    Example:
        >>> predOutcome = predOutcomeBasedOnCutpoint(result.predict(), 0.5)
        >>> predOutcome
        array([ 0.,  0.,  1.,  0.,  0., ...,  0.,  0.,  0.,  0.,  0.])

    ----
    
    Reference / Notes:

    ----

    Author:
        Curtis Neiderer, 2/2014
        
    '''  
    predOutcome = np.ones(np.shape(predOutcomeProb)) * np.nan
    inds = np.arange(0, len(predOutcomeProb), 1)
    predOutcome[inds[predOutcomeProb >= cutpt]] = np.ones(len(inds[predOutcomeProb >= cutpt]))
    predOutcome[inds[predOutcomeProb < cutpt]] = np.zeros(len(inds[predOutcomeProb < cutpt]))
    
    return predOutcome
        
def calculateHLStat(obsOutcome, predOutcomeProb):
    '''
    ----

    calculateHLStat(obsOutcome, predOutcomeProb) -> chiSquareStat, chiSquareProb

    Description:
        Calculates the Hosmer-Lemeshow statistic and probability

    Inputs:
        obsOutcome      -   Array of observed outcomes (i.e., array of binary outcomes)
        predOutcomeProb -   Array of predicted outcome probabilities 

    Outputs:
        chiSquareSum    -  Float designating the chi-squared sum from the HL decile contigency table
        chiSquareProb   -  Float designating the corresponding chi-squared probability  

    Example:
        >>> hlStat, hlProb = calculateHLStat(data.admit, result.predict())
        >>> hlStat
        11.085471996692494
        >>> hlProb
        0.19690311592785789 

    ----
    
    Reference / Notes:

    ----

    Author:
        Curtis Neiderer, 2/2014
        
    '''  
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
    chiSquareDOF = 8 # dof = g - 2
    chiSquareProb = sm.stats.stattools.stats.chisqprob(chiSquareSum, chiSquareDOF)
    
    return chiSquareSum, chiSquareProb
        
def calculateByChanceAccuracy(predTable, fromSM=True):
    '''
    ----

    calculateByChanceAccuracy(predTable, fromSM=True) -> overallCorrectPct, byChancePct, classAccuracyCriteria

    Description:
        Calculates the "by-chance" accuracy and compares to the model classification accuracy

    Inputs:
        predTable   -   Array representing a prediction table
        fromSM      -   Boolean flag designating if the given prediction table is from a statsmodels DataFrame object

    Outputs:
        modelAccuracy           -   Float representing the regression model classification correct percentage
        byChanceAccuracy        -   Float representing the "by-chance" classification correct percentage
        classAccuracyCriteria   -   String designating if the model accuracy criteria is met
        
    Example:
        >>> classCorrectPct, byChanceCriteria, classAccuracyCriteria = calculateByChanceAccuracy(result.pred_table())
        >>> classCorrectPct
        71.0
        >>> byChanceCriteria
        78.501249999999985
        >>> classAccuracyCriteria
        'Not Satisfied, 71.00% < 1.25 * 78.50% = 98.13%'
        
    ----
    
    Reference / Notes:

    ----

    Author:
        Curtis Neiderer, 2/2014
        
    '''  
    if fromSM:
        predTable = np.array([[predTable[1, 1], predTable[1, 0]],
                              [predTable[0, 1], predTable[0, 0]]])
                              
    actualTrueCnt = np.sum(predTable[:, 0])
    actualFalseCnt = np.sum(predTable[:, 1])
    totalCnt = np.sum(predTable)

    byChancePct = ((actualTrueCnt / totalCnt) ** 2  + (actualFalseCnt / totalCnt) ** 2) * 100
    byChanceCriteria = 1.25 * byChancePct
    
    overallCorrectPct = (predTable[0, 0] + predTable[1, 1]) / totalCnt * 100
    
    # Build classAccuracySatisfactoryStr
    if overallCorrectPct > byChanceCriteria:
        classAccuracySatisfactoryStr = 'Satisfied, ' 
        compOperator = ' > '
    else:
        classAccuracySatisfactoryStr = 'Not Satisfied, '
        compOperator = ' < '        
    classAccuracySatisfactoryStr = classAccuracySatisfactoryStr + str('%.2f' % overallCorrectPct) + '%' + \
    compOperator + '1.25 * ' + str('%.2f' % byChancePct) + '% = ' + str('%.2f' % byChanceCriteria) + '%'
    
    return overallCorrectPct, byChancePct, byChanceCriteria, classAccuracySatisfactoryStr
    
def createClassTable(predTable, fromSM=True):
    '''
    ----

    createClassTable(predTable, fromSM) -> classTableDF, sensitivity, specificity

    Description:
        Creates a detailed classification table and calculates sensitivity and specificity of the model

    Inputs:
        predTable   -   Array representing a prediction table
        fromSM      -   Boolean flag designating if the given prediction table is from a statsmodels DataFrame object

    Outputs:
        classTableDF    -   DataFrame representing the model classification table
        sensitivity     -   Float designating the sensitivity of the model
        specificity     -   Float designating the specificity of the model
        tableComp       -   Dictionary of integer prediction counts for TP, FP, TN, and FN

    Example:
        >>>classTable, sensitivity, specificity, tableComp = createClassTable(result.pred_table()) 
        (Pdb) classTable
                Obs_1  Obs_0  Total  Pct_Correct
        Pred_1     30     97    127        23.62
        Pred_0     19    254    273        93.04
        Total      49    351    400        71.00
        >>> sensitivity
        61.22
        >>> specificity
        72.36
        >>> tableComp
        {'tpCnt': 30.0, 'fnCnt': 19.0, 'tnCnt': 254.0, 'fpCnt': 97.0}

    ----
    
    Reference / Notes:

    ----

    Author:
        Curtis Neiderer, 2/2014
        
    '''
    if fromSM:
        predTable = np.array([[predTable[1, 1], predTable[1, 0]],
                              [predTable[0, 1], predTable[0, 0]]])   
    
    tableComponents = { \
        'tpCnt': predTable[0, 0], \
        'tnCnt': predTable[1, 1], \
        'fpCnt': predTable[0, 1], \
        'fnCnt': predTable[1, 0], \
        }  
    
    colTotal = np.array([[np.sum(predTable[:, 0]), np.sum(predTable[:, 1])]])
    rowTotal = np.array([[np.sum(predTable[0, :]), np.sum(predTable[1, :]), np.sum(predTable)]]).T

    classTable = np.concatenate((np.concatenate((predTable, colTotal), axis=0), rowTotal), axis=1)

    zerosCorrect = round(classTable[0, 0] / classTable[0, 2] * 100, 2)
    onesCorrect = round(classTable[1, 1] / classTable[1, 2] * 100, 2)
    totalCorrect = round((classTable[0, 0] + classTable[1, 1]) / classTable[2, 2] * 100, 2)
    sensitivity = round(classTable[0, 0] / np.sum(classTable[:2, 0]) * 100, 2)
    specificity = round(classTable[1, 1] / np.sum(classTable[:2, 1]) * 100, 2)
    
    classTable = np.concatenate((classTable, np.array([[zerosCorrect, onesCorrect, totalCorrect]]).T), axis=1)
    
    rowLabels = ['Pred_1', 'Pred_0', 'Total']
    colLabels = ['Obs_1', 'Obs_0', 'Total', 'Pct_Correct']
    classTableDF = pd.DataFrame(classTable, rowLabels, colLabels)
    
    return classTableDF, sensitivity, specificity, tableComponents

def plotJitteredOutcomeVsEstOutcomeProb(obsOutcome, predOutcomeProb, cutpt=0.5):
    '''
    ----

    plotJitteredOutcomeVsEstOutcomeProb(obsOutcome, predOutcomeProb) -> figure

    Description:
        Creates a scatterplot of the jittered observed outcome colored by the predicted outcome

    Inputs:
        obsOutcome      -   Array of observed outcomes (i.e., array of binary outcomes)
        predOutcomeProb -   Array of predicted outcome probabilities  
        cutpt           -   Float representing the prediction cutoff
        
    Outputs:
        fig_h   -   Figure handle
        figure  -
        
    Example:
        >>> plotJitteredOutcomeVsEstOutcomeProb(data.admit, result.predict())
        
    ----
    
    Reference / Notes:

    ----

    Author:
        Curtis Neiderer, 2/2014
        
    '''
    jitterArray = np.random.uniform(-0.1, 0.1, size=len(obsOutcome))
    obsOutJitter = obsOutcome + jitterArray
    
    fig_h = plt.figure()
    ax_h = fig_h.add_subplot(1, 1, 1)
    ax_h.grid(True)
    # Create scatterplot of predicticted outcome probabilities vs. observed outcome, 
    # colored by predicted outcome
    ax_h.scatter(predOutcomeProb[predOutcomeProb >= cutpt], obsOutJitter[predOutcomeProb >= cutpt], s=20, c='b', label='pred1')
    ax_h.scatter(predOutcomeProb[predOutcomeProb < cutpt], obsOutJitter[predOutcomeProb < cutpt], s=20, c='r', label='pred0')
    # Set Axes Labels
    ax_h.axes.set_xlabel('Predicted Outcome Probabilities')
    ax_h.axes.set_ylabel('Observed Outcome')
    # Set Y-Axis Ticks
    ax_h.axes.set_yticks([0, 1])
    # Set X-Axis Limits
    ax_h.axes.set_xlim([0.0, 1.0])
    # Set Plot Title
    ax_h.axes.set_title('Jittered Outcome vs. Estimated Outcome Probability')
    # Add Legend
    ax_h.legend(loc="right")
    # Label the figure window
    fig_h.canvas.set_window_title('Jittered Outcome vs. Estimated Outcome Probability, Colored by Predicted Outcome')
    # Show plot
    fig_h.show()
    
    return fig_h
    
def plotHistOfEstProbByOutcomeWithCutoffLine(obsOutcome, predOutcomeProb, cutpt):
    '''
    ----

    plotHistOfEstProbByOutcomeWithCutoffLine(obsOutcome, predOutcomeProb) -> figure

    Description:
        Creates a histogram of the estimated outcome probabilities with the outcome prediction cutoff line

    Inputs:
        obsOutcome      -   Array of observed outcomes (i.e., array of binary outcomes)
        predOutcomeProb -   Array of predicted outcome probabilities 
        cutpt           -   Float designating the cutpoint for classification decisions (0 <= cutpt <= 1)

    Outputs:
        figure  -

    Example:
        >>> plotHistOfEstProbByOutcomeWithCutoffLine(data.admit, result.predict(), 0.5)

    ----
    
    Reference / Notes:

    ----

    Author:
        Curtis Neiderer, 2/2014
    
    ----
    
    TODO: Combine the plots into one figure as subplots
        
    '''
    # Observed Outcome = 0
    fig1_h = plt.figure()
    ax1_h = fig1_h.add_subplot(1, 1, 1)
    ax1_h.grid(True)
    ax1_h.axvline(x=cutpt, lw=2, color='black')
    ax1_h.hist(predOutcomeProb[obsOutcome == 0], color='blue')
    # Set Axes Labels
    ax1_h.axes.set_xlabel('Predicted Outcome Probabilities')
    ax1_h.axes.set_ylabel('Density')
    # Set X-Axis Limits
    ax1_h.axes.set_xlim([0.0, 1.0])
    # Set Plot Title
    ax1_h.axes.set_title('Histogram of Estimated Probabilities - Obs0')
    # Add Legend
    ax1_h.legend(loc="upper right")
    # Label the figure window
    fig1_h.canvas.set_window_title('Histogram of Estimated Outcome Probabilities - Obs0')
    # Show plot
    fig1_h.show()
    
    # Observed Outcome = 1
    fig2_h = plt.figure()
    ax2_h = fig2_h.add_subplot(1, 1, 1)
    ax2_h.grid(True)
    ax2_h.axvline(x=cutpt, lw=2, color='black')
    ax2_h.hist(predOutcomeProb[obsOutcome == 1], color='red')
    # Set Axes Labels
    ax2_h.axes.set_xlabel('Predicted Outcome Probabilities')
    ax2_h.axes.set_ylabel('Density')
    # Set X-Axis Limits
    ax2_h.axes.set_xlim([0.0, 1.0])
    # Set Plot Title
    ax2_h.axes.set_title('Histogram of Estimated Probabilities - Obs1')
    # Add Legend
    ax2_h.legend(loc="upper right")
    # Label the figure window
    fig2_h.canvas.set_window_title('Histogram of Estimated Outcome Probabilities - Obs1')
    # Show plot
    fig2_h.show()
    
    return [fig1_h, fig2_h]
    
def plotROCCurve(obsOutcome, predOutcomeProb):
    '''
    ----

    plotROCCurve(obsOutcome, predOutcomeProb) -> figure

    Description:
        Creates a plot of the receiver operating characteristic curve

    Inputs:
        obsOutcome      -   Array of observed outcomes (i.e., array of binary outcomes)
        predOutcomeProb -   Array of predicted outcome probabilities

    Outputs:
        figure  -

    Example:
        >>> roc_auc = plotROCCurve(data.admit, result.predict())
        >>> roc_auc
        0.692841279455

    ----
    
    Reference / Notes:

    ----

    Author:
        Curtis Neiderer, 2/2014
    '''
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(obsOutcome, predOutcomeProb)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    fig_h = plt.figure()
    ax_h = fig_h.add_subplot(1, 1, 1)
    # Add model Performance Line
    ax_h.plot(fpr, tpr, 'b-', label='ROC Curve (Area = %0.3f)' % roc_auc)
    # Add Random Line
    ax_h.plot([0, 1], [0, 1], 'k--', label='random')
    # Set Axes Limits
    ax_h.axes.set_xlim([0.0, 1.0])
    ax_h.axes.set_ylim([0.0, 1.0])
    # Set Axes Labels
    ax_h.axes.set_xlabel('False Positive Rate (1 - Specificity)')
    ax_h.axes.set_ylabel('True Positive Rate (Sensitivity)')
    # Set Plot Title
    ax_h.axes.set_title('ROC Curve: (1 - Specificity) vs. Sensitivity')
    # Add Legend
    ax_h.legend(loc="lower right")
    # Label the figure window
    fig_h.canvas.set_window_title('Histogram of Estimated Outcome Probabilities - Obs1')
    # Show plot
    fig_h.show()
    
    return fig_h, roc_auc
        
def plotCutpointEvalPlot(obsOutcome, predOutcomeProb):
    '''
    ----

    plotCutpointEvalPlot(obsOutcome, predOutcomeProb) -> figure

    Description:
        Creates a plot of cutpoint versus sensitivity and specificity

    Inputs:
        obsOutcome      -   Array of observed outcomes (i.e., array of binary outcomes)
        predOutcomeProb -   Array of predicted outcome probabilities

    Outputs:
        figure  -

    Example:
        >>>
        
    ----
    
    Reference / Notes:

    ----

    Author:
        Curtis Neiderer, 2/2014
        
    ----
    
    TODO: Needs work, doesn't currently work correctly
    '''
#    cutptArray = np.arange(0, 1.05, 0.05)
    cutptArray = np.arange(0, 1.01, 0.01)
    sensitivityArray = np.ones(np.shape(cutptArray))
    specificityArray = np.zeros(np.shape(cutptArray))
    for jj, cutpt in enumerate(cutptArray):
        predOutcome = np.ones(np.shape(predOutcomeProb)) * np.nan
        inds = np.arange(0, len(predOutcomeProb), 1)
        predOutcome[inds[predOutcomeProb >= cutpt]] = np.ones(len(inds[predOutcomeProb >= cutpt]))
        predOutcome[inds[predOutcomeProb < cutpt]] = np.zeros(len(inds[predOutcomeProb < cutpt]))
        
        truePosCnt = 0
        falsePosCnt = 0
        trueNegCnt = 0
        falseNegCnt = 0
        for ii in range(len(obsOutcome)):
            if obsOutcome[ii] == 1 and predOutcome[ii] == 1:
                truePosCnt += 1
            elif obsOutcome[ii] == 1 and predOutcome[ii] == 0:
                falsePosCnt += 1
            elif obsOutcome[ii] == 0 and predOutcome[ii] == 0:
                trueNegCnt += 1
            elif obsOutcome[ii] == 0 and predOutcome[ii] == 1:
                falseNegCnt += 1
                               
        pred_table = np.array([[truePosCnt, falsePosCnt],
                               [falseNegCnt, trueNegCnt]])                               
                               
        if (truePosCnt + falseNegCnt) > 0:
            sensitivityArray[jj] = truePosCnt / (truePosCnt + falseNegCnt)
        else:
            if jj > 0:
                sensitivityArray[jj] = sensitivityArray[jj - 1]
            else:
                sensitivityArray[jj] = 0
                
        if (trueNegCnt + falsePosCnt):
            specificityArray[jj] = trueNegCnt / (trueNegCnt + falsePosCnt)
        else:
            if jj > 0:
                specificityArray[jj] = specificityArray[jj - 1]                
            else:
                specificityArray[jj] = 1
            
#        print 'Cutpoint = ' + str(cutpt), '\n', pred_table, '\n', 
#        print 'Sensitivity = ' + str(sensitivity), '\n', 'Specificity = ' + str(specificity), '\n'
        
    print 'Sensitivity: \n', sensitivityArray
    print 'Specificity: \n', specificityArray
    
    fig = plt.figure(1)
    ax = fig.add_subplot(1, 1, 1)
    ax.grid(True)
    ax.plot(cutptArray, sensitivityArray, '-k', cutptArray, specificityArray, '-b')
    fig.show()

    fig = plt.figure(2)
    ax = fig.add_subplot(1, 1, 1)
    ax.grid(True)
    ax.plot(sensitivityArray, 1 - specificityArray, '-k')
    fig.show()
    
    test = 1   
    
########### ----- Enables Command Line Call ----- ##########
if __name__ == '__main__':
    main()
########### ----- Enables Command Line Call ----- ##########