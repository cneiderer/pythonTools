# -*- coding: utf-8 -*-
"""
Created on Wed Mar 05 06:23:39 2014

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
import os
import pickle
import csv
import gc
#
import LogisticRegression.GoodnessOfFitAnalysis as gof
import Sampling.RandomlySampleInds as rs
import LogisticRegression.ResidualOutlierAnalysis as roa

########## ----- Main Function ----- ##########
def main():
    '''
    ----
    
    main() -> Various figures, tables, and files
    
    Description:
        Builds a logistic regression model given a set of input conditions
    
    Dependencies:
        __future__,
        pandas,
        statsmodels,
        numpy,
        matplotlib,
        os,
        pickle,
        csv,
        RandomlySampleInds,
        GoodnessOfFitAnalysis,
        ResidualOutlierAnalysis
        
    ----
    
    Author:
        Curtis Neiderer, 2/2014
    '''
    
##### Configure the inputs #####
    # Select Data File
#    dataFilePath = 'http://www.ats.ucla.edu/stat/data/binary.csv'
    dataFilePath = 'C:/AnalysisTools/python_toolbox/DataSets/AdultDataSet.csv'
    
    # Define output path
    outputPath = 'C:/AnalysisTools/python_toolbox/Output/'
    # Find full data file name from data file path
    fullDataFileName = dataFileName = os.path.split(dataFilePath)[-1]
    # Separate data file name from extension
    dataFileName, dataFileExt = os.path.splitext(fullDataFileName)
    
    # Define Sample Size
    sampleSize = 0

    # Define parameters of interest
    # If set to True, keep parameters in paramsOfInterest; if set to False, remove
#    paramsOfInterestFlag = True 
#    paramsOfInterest = []
    
    paramsOfInterestFlag = False
    paramsOfInterest = ['education']
    
#    paramsOfInterestFlag = True
#    paramsOfInterest = ['age', 'workclass']

    # Define outcome parameter
#    outcomeParam = ''
#    outcomeParam = 'admit'
    outcomeParam = 'income-level'

    # Define known metric parameters to be dummified
    metricParamsToDum = np.array([])
#    metricParamsToDum = np.array(['rank'])
    
    # Define p-value significance level
    pValSig = 0.05    
    
    # Set flag to remove all parts of a dummified variable if any part is insignificant
    # If set to True, removes all dummy parts; if set to False, removes on insignificant part
    removeAllPartsOfDummy = True

    
##### Organize / Clean the data #####    
    # Read data into Pandas DataFrame
    df = pd.read_csv(dataFilePath)

#    df = pd.read_csv("C:\AnalysisTools\python_toolbox\Output\cleanDF_binary.csv")
#    df = df[["admit", "gre", "gpa", "rank"]]
    
#    pd.load("C:\AnalysisTools\python_toolbox\Output\cleanDF_binary.pkl")
    
    # Randomly sample the data 
    if sampleSize > 0:
        # Sample the data
        randomInds = rs.randomlySampleInds(np.arange(0, len(df), 1), sampleSize)
        # Sort randomInds
        # Create DataFrame with only the sampled data
        df = df.ix[randomInds]
    
    # Modify DataFrame to contain only parameters of interest
    # Make all parameter names lowercase
    df = convertParametersToLowerCase(df)
    
    # Make sure defined outcome parameter exists in data
    if not np.any(df.columns == outcomeParam):
        # Throw exception, ValueError: outcomeParam is not in the DataFrame
        raise KeyError(outcomeParam + ', the defined outcome parameter, is not in the DataFrame') 
    
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
    
    # Separate categorical and metric parameters
    # Create list of all categorical variable names
    catParams = np.array(df.dtypes[df.dtypes == 'object'].index)
    # Create list of all metric variable names
    metricParams = np.array(df.dtypes[df.dtypes != 'object'].index)
    # Remove outcomeParameter from categorical and/or metric parameter lists
    catParams = catParams[catParams != outcomeParam]
    metricParams = metricParams[metricParams != outcomeParam]
    
    # Convert outcomeParameter from string outcome into integer outcome, if necessary
    # (i.e., [A, B, A, A, B] to [0, 1, 0, 0, 1])
    if type(df[outcomeParam][0]) is str:
        df = convertOutcomeParameterToBinary(df, outcomeParam)      
    
    # Dummify all categorical variables
    # Remove metricParamsToDum from metricParams, if there are specified metric parameters to dummify
    if len(metricParamsToDum) > 0:
        for param in metricParamsToDum:
            metricParams = metricParams[metricParams != param]

    # Create combined list of categorical and metric parameters to dummify
    dummifyParams = np.concatenate((catParams, metricParamsToDum), axis=0)
    
    # Create clean DataFrame
    # Insert outcomeParam as first column of clean DataFrame followed by metric parameters
    data = df[np.insert(metricParams, 0, outcomeParam)]
    
    # Add dummified parameters to clean DataFrame after metric parameters
    # Initialize the dummyPrefixArray as empty array 
    # (Note: to be used to find all parts of a dummified parameter for removal)
    dummyPrefixArray = np.array([])
    dummyPrefix = ''
    for param in dummifyParams:
        # Get dummy parameter prefix
        dummyPrefix = getDummyPrefix(param)
        # Add dummyPrefix to dummyPrefixArray
        dummyPrefixArray = np.append(dummyPrefixArray, dummyPrefix)
        # Dummify the parameter
        paramDummy = pd.get_dummies(df[param], prefix=dummyPrefix)
        # Add dummified parameter to data 
        # (Note: remember to leave out first dummy parameter column)
        data = data.join(paramDummy.ix[:, paramDummy.columns[1]:]) 
    
    # Manually add intercept to clean DataFrame
    data['intercept'] = 1.0
    
    # Save organized data from clean Dataframe to a CSV and a PKL file
    # Define file names
    cleanDataFrameFileCSV = outputPath + 'cleanDF_' + dataFileName + '.csv'
    cleanDataFrameFilePKL = outputPath + 'cleanDF_' + dataFileName + '.pkl'
    # To CSV
    data.to_csv(cleanDataFrameFileCSV)
    # To PKL
    with open(cleanDataFrameFilePKL, 'w') as f:
        pickle.dump(data, f)

    # Free memory, delete variables that are no longer needed 
    # TODO: Create function to use for variable cleanup 
    #(Should check for variable existence first, then delete, to avoid choking)       
    del \
        catParams, \
        dummifyParams, \
        dummyPrefix, \
        metricParams, \
        metricParamsToDum, \
        paramsOfInterest, \
        paramsOfInterestFlag, \
        sampleSize, \
        fullDataFileName, \
        cleanDataFrameFileCSV, \
        cleanDataFrameFilePKL, \
        dataFilePath, \
        dataFileExt
        
    # Manually call garbage collector
    gc.collect()

##### ========== End while loop here ====================================================


##### Parameter / model significance check #####
    # Initialize stopAnalysisFlag to False
    stopAnalysisFlag = False
    # Initialize removeFromModel list to empty list
    removeFromModel = []
    whileCnt = 0    
    # Check stopAnalysisFlag and make sure you have at least one independent parameter 
    # (i.e., you need at least 3 columns in your DataFrame: dependent parameter, intercept, independent parameters)
    while not stopAnalysisFlag and (len(data.columns) > 2): 
        # Inrement whileCnt (Used in file names so we don't overwrite previous iterations.)
        whileCnt += 1
        print 'Iteration' + str(whileCnt)
        # Remove specified parameters from the DataFrame if necessary
        if len(removeFromModel) > 0:
            # Loop through removeFromModel 
            # (Note: Will only contain more than one parameter when removing all parts of a dummified parameter.)
            for removeParam in removeFromModel:
                data = data[data.columns[data.columns != removeParam]]


##### Build and fit logistic regression model #####
        # (9) Designate independent parameters to train data
        indepParams = data.columns[data.columns != outcomeParam]  
    
        # (10) Transform the data with the logit function
        logit = sm.Logit(data[outcomeParam], data[indepParams])
        
        # (11) Fit the transformed data
#        result = logit.fit()
        result = logit.fit_regularized()
        
        # (12) Save the model summary to a CSV, TXT, and PKL file 
        # Define file names
        modelFitSummaryFilePKL = outputPath + 'modelFitSummary_' + str(whileCnt) + '_' + dataFileName + '.pkl'
#        modelFitSummaryFilePKL = "%smodelFitSummary_%d_%s.pkl" % (outputPath, whileCnt, dataFileName)
        modelFitSummaryFileTXT = outputPath + 'modelFitSummary_' + str(whileCnt) + '_' + dataFileName + '.txt'
        
        # To PKL
#        result.save(modelFitSummaryFilePKL)
        
        # To TXT
        txtModelSummaryStr = result.summary().as_text()
        with open(modelFitSummaryFileTXT, 'w') as f:
            f.write(txtModelSummaryStr)
            
        # Free memory, delete variables that are no longer needed
        # TODO: Create function to use for variable cleanup
        del \
            logit, \
            modelFitSummaryFilePKL, \
            modelFitSummaryFileTXT, \
            txtModelSummaryStr
            
        # Manually call garbage collector
        gc.collect()    

    
##### Goodness-of-Fit model evaluation #####
        # (13) Check beta coefficient p-values for significance
        # Get beta coefficient p-values
        betaPVals = result.pvalues
        
        # Are any betaPVals non-significant to the specified sigLevel
        if not np.any(betaPVals > pValSig):
            # If no, set stopAnalysisFlag to True
            stopAnalysisFlag = True
        else:
            # If yes, do not set stopAnalysisFlag to False
            stopAnalysisFlag = False
            
            # Find largest non-significant pVal and corresponding parameter name
            # Initialize largestPVal to 0
            largestPVal = 0
            # Initialize paramToRemove to empty string
            paramToRemove = ''
            # Loop through betaPVals to find largest non-significant p-value
            for ii, pVal in enumerate(betaPVals):
                if pVal > largestPVal:
                    largestPVal = pVal 
                    paramToRemove = data.icol(ii).name
            # Remove loop variables to free memory
            del ii, pVal
            
            # If paramToRemove is 'intercept', set stopAnalysisFlag because we don't 
            # care about the significance of the intercept
            if paramToRemove == 'intercept':
                stopAnalysisFlag = True
            
            # If removeAllPartsOfDummy is set to True, all parts of a dummified 
            # parameter will be removed if any part is found to be insignificant;
            # If set to False, only the insignificant part will be removed from the model
            if removeAllPartsOfDummy:
                # Create an array of the independent parameter prefixes
                indepParamsPrefixArray = np.array([x[:5] for x in indepParams])
                # Create list of parameters to remove from model
                paramToRemovePrefix = paramToRemove[:5]
                # Remove all parts
                removeFromModel = indepParams[indepParamsPrefixArray == paramToRemovePrefix]     
            else:
                # Remove only insignificant part
                removeFromModel = indepParams[indepParams == paramToRemove]
    
        # Create human readable regression equation
        regressionEqn = gof.createFormattedRegressionEquation(result.params)
        
        # Convert beta pvalue from Series to DataFrame for TXT file
        betaPValDF = pd.DataFrame(result.pvalues)
        betaPValDF.columns = ['p-value']
    
        # Check likelihood ratio (Omnibus test of model coefficients) p-value for significance
        llrStat, llrStatPVal = result.llr, result.llr_pvalue
        
        # Calculate the Hosmer-Lemeshow statistic and check the chi-squared probability for significance
        hlStat, hlProb = gof.calculateHLStat(data[outcomeParam], result.predict()) 
        
        # Create Classification Table
        classTable, sensitivity, specificity, tableCnts = gof.createClassTable(result.pred_table()) 
            
        # Calculate the "by-chance" classification and compare with model performance
        classCorrectPct, byChancePct, byChanceCriteria, classAccuracyReq = gof.calculateByChanceAccuracy(result.pred_table())
        
        # Calculate pseudo-R-squared
        R2m = gof.calculatePseudoRSquared(result.llf, result.llnull, int(result.nobs), 'McFadden')
        R2cs = gof.calculatePseudoRSquared(result.llf, result.llnull, int(result.nobs), 'Cox-Snell')
        R2n = gof.calculatePseudoRSquared(result.llf, result.llnull, int(result.nobs), 'Nagelkerke')
    
        # Calculate beta-coefficient variance inflation factors (VIF)
        VIF, multicolPresent = gof.calculateVIF(result.cov_params())
        # Convert VIF from Series to DataFrame for TXT file
        betaVIF = pd.DataFrame(VIF)
        betaVIF.columns = ['Beta VIF']

        # Performance Visualization
        # TODO: Add input arguments to plot routines
        # Need to be able to pass dataset name for embedding into window/plot title and saving
        # Need a saveFigType to define how you save the figure
        # Need a saveFigFlag to define if figure is to be saved within the plot routine
        # Need outputPath string to define save location
        fig_handles = []
        
        # Plot ROC curve
        roc_figH, roc_auc = gof.plotROCCurve(data[outcomeParam], result.predict())
        # Save ROC curve
        
        # Add ROC curve handle to fig_handles
        fig_handles.append(roc_figH)
    
        # Plot histograms of estimated probabilities by outcome status with cutpoint line
    
        # Save histograms
    
        # Add histogram handles to fig_handles
        fig_handles.extend(gof.plotHistOfEstProbByOutcomeWithCutoffLine(data[outcomeParam], result.predict(), 0.5))
    
        # Plot estimated probabilities versus jittered outcome, colored by predicted outcome
        
        # Save jitter
        
        # Add jitter handle to fig_handles
        fig_handles.append(gof.plotJitteredOutcomeVsEstOutcomeProb(data[outcomeParam], result.predict()))
        
        # TODO: Save performance visualizations into PPT
        # classification table, ROC curve, histograms, jittered outcome
        
        # Close all open figures
        # TODO: Close each figure after it is saved
        plt.close('all')
        
        # Record all goodness-of-fit evaluation results into an object
        gofEvalResult = gof.GoodnessOfFitEvaluation( \
            regressionEqn=regressionEqn, \
            betaPValue=result.pvalues, \
            omnibusStat=llrStat, \
            omnibusPValue=llrStatPVal, \
            hosmerLemeshowStat=hlStat, \
            hosmerLemeshowProb=hlProb, \
            classAccuracy_classTable=classTable, \
            classAccuracy_sensitivity=sensitivity, \
            classAccuracy_specificity=specificity, \
            classAccuracy_classTableCnts=tableCnts, \
            classAccuracy_modelCorrectPct=classCorrectPct, \
            classAccuracy_byChanceCorrectPct=byChancePct, \
            classAccuracy_byChanceCriteria=byChanceCriteria, \
            classAccuracy_requirement=classAccuracyReq, \
            prsquared_McFadden=R2m, \
            prsquared_CoxSnell=R2cs, \
            prsquared_Nagelkerke=R2n, \
            betaVIF=VIF, \
            betaVIF_multicollinearity=multicolPresent, \
            roc_auc=roc_auc, \
            )
            
        # Manually call garbage collector
        gc.collect()
    
        # Build goodness-of-fit evaluation results summary string
        gofEvalResultStr = '=======================\n' + \
            'Goodness of Fit Summary\n' + \
            '=======================\n' + \
            '\n' + \
            'Model Equation\n' + \
            '--------------\n' + \
            regressionEqn + '\n' + \
            '\n' + \
            'Model Significance\n' + \
            '------------------\n' + \
            'Omnibus Test of Coefficients:\t' + str('%.5g' % llrStat) + '\t\tp-value:\t' + str('%.5g' % llrStatPVal) + '\n' + \
            'Hosmer-Lemeshow Statistic:\t' + str('%.5g' % hlStat) + '\t\tprobability:\t' + str('%.5g' % hlProb) + '\n' + \
            '\n' + \
            'Pseudo R-Squared:\n' + \
            '-----------------\n' + \
            'McFadden R-Squared:\t\t' + str('%.5g' % R2m) + '\n' + \
            'Cox-Snell R-Squared:\t\t' + str('%.5g' % R2cs) + '\n' + \
            'Nagelkerke R-Squared:\t\t' + str('%.5g' % R2n) + '\n' + \
            '\n' + \
            'Classification Table\n' + \
            '--------------------\n' + \
            str(classTable) + '\n' + \
            'Sensitivity:\t' + str('%.2f' % sensitivity) + '%\n' + \
            'Specificity:\t' + str('%.2f' % specificity) + '%\n' + \
            '\n' + \
            'Classification Accuracy\n' + \
            '-----------------------\n' + \
            'Model Classification Accuracy:\t\t' + str('%.2f' % classCorrectPct) + '%\n' + \
            'By-Chance Classification Accuracy:\t' + str('%.2f' % byChancePct) + '%\n' + \
            'Model Accurracy Requirement:\t\t' + classAccuracyReq + '\n' + \
            '\n' + \
            'Coefficient Significance\n' + \
            '------------------------\n' + \
            str(betaPValDF) + '\n' + \
            '\n' + \
            'Pseudo Coefficient VIFs\n' + \
            '-----------------------\n' + \
            str(betaVIF) + '\n' + \
            'Multi-Collinearity Check:\t' + multicolPresent   
        
        # Add goodness-of-fit evaluation summary string to gofResultObject
        gofEvalResult.add_data(summaryStr=gofEvalResultStr)
    
        # Define file names
        gofSummaryFilePKL = outputPath + 'gofSummary_' + str(whileCnt) + '_' + dataFileName + '.pkl'
        gofSummaryFileTXT = outputPath + 'gofSummary_' + str(whileCnt) + '_' + dataFileName + '.txt'
    
        # Save to PKL
        with open(gofSummaryFilePKL, 'w') as f:
            pickle.dump(gofEvalResult, f)    
    
        # Save to TXT
        with open(gofSummaryFileTXT, 'w') as f:
            f.write(gofEvalResultStr)

        # Free memory, delete variables that are no longer needed
        # TODO: Create function to use for variable cleanup
        del \
            llrStat, \
            llrStatPVal, \
            hlStat, \
            hlProb, \
            classTable, \
            sensitivity, \
            specificity, \
            tableCnts, \
            classCorrectPct, \
            byChancePct, \
            byChanceCriteria, \
            classAccuracyReq, \
            R2cs, \
            R2m, \
            R2n, \
            VIF, \
            betaPValDF, \
            multicolPresent, \
            roc_auc, \
            gofEvalResult, \
            gofEvalResultStr, \
            gofSummaryFilePKL, \
            gofSummaryFileTXT
            
##### Residual / Outlier Analysis #####
        # Calculate Standardized Residuals
        standardRes = roa.calculateStandardizedRes(result.resid_dev, result.predict())
    
        # Calculate the MLE residual variance
        mleResVar = roa.calculateMLEResidualVar(result.resid_dev, result.df_resid)
    
        # Calculate hat-matrix (leverage statistic)
        leverage, hatMatrix = roa.calculateHatMatrix(data[data.columns[data.columns != outcomeParam]], result.cov_params())
    
        # Calculate studentized residuals
        studentRes = roa.calculateStudentizedResiduals(result.resid_dev, mleResVar, leverage)
    
        # Calculate Cook distances
        cooksD = roa.calculateCooksDistance(result.df_model, studentRes, leverage)
        
        # Calculate Delta-Beta Statistics (DBetas)
        dBetaStat = roa.calculateDBetaStat(result.resid_pearson, leverage)
        
        # Calculate Standardized Pearson Residuals
        standardPearsonRes = roa.calculateStandardPearsonRes(result.resid_pearson, leverage)
        
        # Calculate Standardized Delta-Beta Statistics (DBetas)
        standardDBetaStat = roa.calculateDBetaStat(standardPearsonRes, leverage, fromSM=False) 
    
        # Record all residual outlier analysis results into an object
        roaResult = roa.ResidualOutlierAnalysisResult( \
            resid=result.resid_dev, \
            resid_standard=standardRes, \
            resid_mleVariance=mleResVar, \
            leverage=leverage, \
            resid_student=studentRes, \
            cookDistance=cooksD, \
            deltaBetaStat=dBetaStat, \
            resid_standardPearson=standardPearsonRes, \
            deltaBetaStat_standard=standardDBetaStat, \
            )
    
        # Combine various residual measures into a DataFrame
        roaResultSummaryDF = pd.concat({ \
            'resid': result.resid, \
            'resid_standardized': standardRes, \
            'resid_studentized': studentRes, \
            'resid_pearson': result.resid_pearson, \
            'resid_pearsonStandardized': standardPearsonRes, \
            'leverage': leverage, \
            'cookDistance': cooksD, \
            'deltaBeta': dBetaStat, \
            'deltaBeta_standardized': standardDBetaStat, \
            }, axis=1)

        # Add residual outlier analysis summary DataFrame to roaResult object
        roaResult.add_data(summaryDF=roaResultSummaryDF)
            
        # Save residual analysis summary 
        # Define file names
        roaSummaryFilePKL = outputPath + 'roaSummary_' + str(whileCnt) + '_' + dataFileName + '.pkl'
        roaSummaryFileCSV = outputPath + 'roaSummary_' + str(whileCnt) + '_' + dataFileName + '.csv'
        
        # Save to PKL
        with open(roaSummaryFilePKL, 'w') as f:
            pickle.dump(roaResult, f)    
        
        # Save to CSV
        roaResultSummaryDF.to_csv(roaSummaryFileCSV)
        
        # Free memory, delete variables that are no longer needed
        del \
            hatMatrix, \
            mleResVar, \
            roaResultSummaryDF, \
            roaSummaryFileCSV, \
            roaSummaryFilePKL
            
            
            
##### ========== End while loop here ====================================================    

    
    test = 1
    

########## ----- Main Function ----- ##########
    
def convertOutcomeParameterToBinary(df, outcomeParameter):
    '''
    ----
    
    convertOutcomeParameterToBinary(DataFrame, String) -> DataFrame

    Description:
        Returns the DataFrame with the specified outcome parameter converted to 
        binary array from string array
        
    Inputs:
        df
        outcomeParameter
        
    Outputs:
        df
        
    Example:
        >>> df
        <class 'pandas.core.frame.DataFrame'>
        Int64Index: 32561 entries, 0 to 32560
        Data columns (total 15 columns):
        age               32561  non-null values
        workclass         32561  non-null values
        fnlwgt            32561  non-null values
        ...
        hours-per-week    32561  non-null values
        native-country    32561  non-null values
        income-level      32561  non-null values
        dtypes: int64(6), object(9)
        >>> outcomeParam = 'income-level'
        >>> outcomeParam
        'income-level'
        >>> df[outcomeParam]
        0      <=50K
        1      <=50K
        2      <=50K
        ...
        32558     <=50K
        32559     <=50K
        32560      >50K
        Name: income-level, Length: 32561, dtype: object
        >>> type(df[outcomeParam][0]) 
        <type 'str'>
        >>> df = convertOutcomeParameterToBinary(df, outcomeParam)
        >>> df
        <class 'pandas.core.frame.DataFrame'>
        Int64Index: 32561 entries, 0 to 32560
        Data columns (total 15 columns):
        age               32561  non-null values
        workclass         32561  non-null values
        fnlwgt            32561  non-null values
        ...
        hours-per-week    32561  non-null values
        native-country    32561  non-null values
        income-level      32561  non-null values
        dtypes: int64(7), object(8)
        >>> df[outcomeParam]
        df[outcomeParam]
        0     0
        1     0
        2     0
        ...
        32558    0
        32559    0
        32560    1
        Name: income-level, Length: 32561, dtype: int64
        >>> type(df[outcomeParam][0])
        <type 'numpy.int64'> 
        
    ----
    
    Author:
        Curtis Neiderer, 3/2014
    '''    
    # Pull out outcome column from DataFrame
    outcome = np.array(df[outcomeParameter])
    # Find unique outcome levels/categories
    uniqueOutcomeLevels = np.unique(outcome)
    # Loop through unique outcome levels and assign integer levels
    for ii, level in enumerate(uniqueOutcomeLevels):
        df[outcomeParameter][df[outcomeParameter] == level] = ii
    # Convert outcomeParameter dtype from object to int64  
    df[outcomeParameter] = df[outcomeParameter].astype(int64)
     
    return df
        
def convertParametersToLowerCase(df):
    '''
    ----
    
    converParametersToLowerCase(Array of Strings) -> Array of Strings
    
    Description:
        Returns the array of strings with everything in lowercase
    
    Inputs:
        parametersList
    
    Outputs:
        parametersList
        
    Example:
        >>> parametersList = np.array(['A', 'B', 'C'])
        >>> parametersList
        array(['A', 'B', 'C'], dtype='|S1')
        >>> parametersList = convertParametersToLowerCase(parametersList)
        >>> parametersList
        array(['a', 'b', 'c'], dtype='|S1')
    
    ----
    
    Author:
        Curtis Neiderer, 3/2014
    '''
    # Pull parameter names out of DataFrame
    parametersList = np.array(df.columns)
    # Loop through the parametersList and convert all parameter names to lowercase
    for ii, param in enumerate(parametersList):
        parametersList[ii] = param.lower()
    # Replace parameter names in DataFrame with lowercase versions
    df.columns = parametersList
    
    return df
    
def getDummyPrefix(paramFullName):
    '''    
    ----
    
    getDummyPrefix(String) -> String
    
    Description:
        Returns the dummyPrefix with max length 5 for the given parameter name
    
    Inputs:
        paramFullName
    
    Outputs:
        dummyPrefix
    
    Example:
        >>> dummyPrefix = getDummyPrefix('parameterNameExample')
        >>> dummyPrefix
        'param'
    
    ----
    
    Author:
        Curtis Neiderer, 3/2014
    '''
    
    # Truncate parameter name to first 5 characters
    if len(paramFullName) > 5:
        dummyPrefix = paramFullName[:5]
    else: # If parameter name is less than length 5, pad to length 5 with underscores
        dummyPrefix = paramFullName + ('_' * (5 - len(paramFullName)))
    
    return dummyPrefix


    
########### ----- Enables Command Line Call ----- ##########
if __name__ == '__main__':
    main()
########### ----- Enables Command Line Call ----- ##########