# -*- coding: utf-8 -*-
"""
Created on Mon Feb 03 09:13:53 2014

@author: Curtis.Neiderer
"""

from __future__ import division
import pylab as py
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

########## ----- Main Function ----- ##########
def main():

    # Setup Input Data
    # Define independent variable matrix X
    X = np.matrix([[2, 8, 11, 10, 8, 4, 2, 2, 9, 8, 4, 11, 12, 2, 4, 4, 20,
                      1, 10, 15, 15, 16, 17, 6, 5], [50, 110, 120, 550, 295,
                      200, 375, 52, 100, 300, 412, 400, 500, 360, 205, 400,
                      600, 585, 540, 250, 290, 510, 590, 100,400]]).T
    # Define dependent variable vector y
    y = np.matrix([[9.95, 24.45, 31.75, 35.00, 25.02, 16.86, 14.38, 9.60,
                      24.35, 27.50, 17.08, 37.00, 41.95, 11.66, 21.65, 17.89,
                      69.00, 10.30, 34.93, 46.59, 44.88, 54.12, 56.63, 22.13,
                      21.15]]).T

#    X = np.array([[20, 25, 30, 35, 40, 50, 60, 65, 70, 75, 80, 90]])
#    X = np.matrix(np.concatenate((X, X ** 2), axis=0)).T
#    y = np.matrix([[1.81, 1.70, 1.65, 1.55, 1.48, 1.40, 1.30, 1.26, 1.24, 1.21, 1.20, 1.18]]).T

    # Print data to console to make sure it looks correct
    print 'X: \n', X, '\n'
    print 'y: \n', y, '\n'

    # Calculate Beta coefficients
    Beta = calculateBeta(X,y)
    print 'Beta: \n', Beta, '\n'

    # Create a human readable equation of the LSR fit line
    lsrEquation = createFormattedLSREquation(Beta)
    print 'LSR Equation: ', lsrEquation, '\n'
    
    # Calculate the predicted y-values (i.e., independent variable responses)
    yPred = calculatePredictY(X, y, Beta)
    print 'yPred: \n', yPred, '\n'

    # Calculate residual errors
    residuals = calculateResiduals(X,y,Beta)
    print 'residuals: \n', residuals, '\n'

    # Plot the observed data with the LSR fit line
    plotDataWithLSRLine(X, y, Beta)

    # Plot the residual errors
    plotResiduals(X, residuals)

    # Calculate the degrees of freedom for Regression, Residual Error, and Total
    dofRegress = calculateDOF(X, 'Regression')
    print 'DOF Regression: ', dofRegress, '\n'

    dofError = calculateDOF(X, 'Error')
    print 'DOF Error: ', dofError, '\n'

    dofTotal = calculateDOF(X, 'Total')
    print 'DOF Total: ', dofTotal, '\n'

    # Calculate the variance of y
    mleVar = calculateMLEVar(residuals, np.shape(X)[0])
    print 'MLE Variance: ', mleVar, '\n'

    # Calculate the covariance matrix of X 
    covMatrix = calculateCovMatrix(X)
    print 'Covariance Matrix: \n', covMatrix, '\n'

    # Calculate the beta covariance matrix
    betaCov = calculateBetaCov(mleVar, covMatrix)
    print 'Beta Covariance: \n', betaCov, '\n'

    # Calculate the sums of squares for Regression, Error, and Total
    SSR = calculateSSR(X, y, Beta, yPred)
    print 'SSR: ', SSR, '\n'

    SSE = calculateSSE(X, y, Beta)
    print 'SSE: ', SSE, '\n'

    SST = calculateSST(X, y, yPred)
    print 'SST: ', SST, '\n'

    # Calculate the mean squares for Regression and Error
    MSR = calculateMSR(SSR, dofRegress)
    print 'MSR: ', MSR, '\n'

    MSE = calculateMSE(SSE, dofError)
    print 'MSE: ', MSE, '\n'

    # Calculate the F Statistic
    fStat, fStatPVal = calculateFStat(MSR, MSE, dofRegress, dofError)
    print 'fStat: ', fStat, ' pVal: ', fStatPVal, '\n'

    # Calculate coefficient of determination
    rSquared = calculateR2(SSR, SSE, SST)
    print 'R-Squared: ', rSquared, '\n'

    # Calculate adjusted coefficient of determination
    rSquaredAdj = calculateR2Adj(SSR, SSE, SST, dofError, dofTotal)
    print 'Adjusted R-Squared: ', rSquaredAdj, '\n'

    # Calculate the T Statistic
    tStat, tStatPVal = calculateTStat(Beta, mleVar, covMatrix, dofRegress)
    print 'tStat: \n', tStat, '\n'
    print 'tStat pVal: \n', tStatPVal, '\n'

    # May need to be corrected
    betaCI = calculateBetaCI(X, Beta, mleVar, covMatrix, dofError)
    print 'Beta Confidence Intervals: \n', betaCI, '\n'

    # Calculate the 95% Confidence and Prediction Intervals
    meanResponseCI = calculateMeanResponseCI(X, Beta, mleVar, covMatrix, dofError, 0.95)
    print 'Mean Response Confidence Intervals: \n', meanResponseCI, '\n'

    predInterval = calculateMeanResponsePI(X, Beta, mleVar, covMatrix, dofError, 0.95)
    print 'Prediction Intervals: \n', predInterval, '\n'

    # Plot the 95% Confidence and Prediction Intervals
    # Need to write function to plot CIs and PIs and LSR fit line on same figure    
    plotMeanResponseCIWithData(X, y, Beta, mleVar, covMatrix, dofError, 0.95)

    plotMeanResponsePIWithData(X, y, Beta, mleVar, covMatrix, dofError, 0.95)

    # Residual Analysis
    # Calculate standardized residuals
    standardizedRes = calculateStandardizedResiduals(residuals, MSE)
    print 'Standardized Residuals: \n', standardizedRes, '\n'
    
    # Calculate the Hat Matrix
    hiiArray, fullHatMatrix = calculateHatMatrix(X, covMatrix)
#    print 'Hat Matrix: \n', fullHatMatrix, '\n'
    print 'Leverage: \n', hiiArray, '\n'

    # Calculate the studentized residuals
    studentRes = calculateStudentizedRes(residuals, mleVar, hiiArray)
    print 'Studentized Residuals: \n', studentRes, '\n'

    # Calculate Cook's distance for residuals
    cooksDist = calculateCooksDistance(dofRegress, studentRes, hiiArray)
    print 'Cook\'s Distance: \n', cooksDist, '\n'

    # Calculate the variance inflation factor
    VIF = calculateVIF(covMatrix)
    print 'VIF: \n', VIF, '\n'


########## ----- Main Function ----- ##########

def calculateBeta(X, y):
    '''
    calculateBeta
        Calculates Beta coefficients given X matrix of independent variables and
        y vector of dependent dependent variable

    Inputs:
        X       -   Array of independent variables (x_1 ... x_n) with a leading column of ones for intercept
        y       -   Dependent variable

    Outputs:
        Beta    -   Array of calculated Beta coefficients

    Example:

    Author:
        Curtis Neiderer
    '''
    # Add column of leading ones for Beta1 intercept
    X = np.concatenate((np.matrix(np.ones(np.shape(X)[0])).T,X), axis=1)
    # Calculate Beta coefficients
    Beta = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))

    return Beta

def calculateResiduals(X, y, Beta):
    '''
    calculateResiduals
        Calculate residual errors

    Inputs:
        X       -   Array of independent variables (x_1 ... x_n) with a leading column of ones for intercept
        y       -   Dependent variable
        Beta    -   Array of calculated Beta coefficients

    Outputs:
        residuals   -   Array of calculated y residual errors

    Example:

    Author:
        Curtis Neiderer
    '''
    yPred = calculatePredictY(X, y, Beta)
    residuals = y - yPred

    return residuals

def createFormattedLSREquation(Beta):
    '''
    createFormattedLSREquation
        Creates a human readable LSR equation

    Inputs:
        Beta    -   Array of beta coefficients

    Outputs:
        eqnString   -   Formated LSR equation string

    Example:

    Author:
        Curtis Neiderer
    '''
    eqnString = 'y = ' + str(round(Beta[0, 0], 5))
    for ii, coeff in enumerate(Beta[1:]):
        coeff = round(coeff[0,0], 5)
        if coeff == 0.0:
            continue
        elif (coeff < 0):
            sign = ' - '
        else:
            sign = ' + '
        if len(Beta) <= 2:
            eqnString += sign + str(abs(coeff)) + ' * x'
        else:
            eqnString += sign + str(abs(coeff)) + ' * x' + str(ii+1)

    return eqnString

def calculatePredictY(xObs, yObs, Beta):
    '''
    calculatePredictY
        Calculates predicted y given observed X and y values

    Inputs:
        xObs    -   Array of independent variables (x_1 ... x_n) with a leading column of ones for intercept
        yObs    -   Dependent variable
        Beta    -   Array of calculated Beta coefficients

    Outputs:
        yPred   -   Array of predicted y values

    Example:

    Author:
        Curtis Neiderer
    '''
    yPred = round(Beta[0,0], 5)
    for ii, coeff in enumerate(Beta[1:,0]):
        coeff = round(coeff[0,0], 5)
        yPred += coeff * xObs[:, ii]

    return yPred

def plotDataWithLSRLine(X, y, Beta):
    '''
    plotDataWithLSRLine
        Plot data with Least Squares Regression line

    Inputs:
        X       -   Array of independent variables (x_1 ... x_n) with a leading column of ones for intercept
        y       -   Dependent variable
        Beta    -   Array of calculated Beta coefficients

    Outputs:
        One plot for each independent variable (x_1 ... x_n) vs. y

    Example:

    Author:
    '''
    # Generate LSR fit data
    # Preallocate xFit array
    xCols = np.shape(X)[1]
    xRows = np.shape(X)[0]
    xFit = np.ones((xRows, xCols)) * -99
    # Calculate xFit array
    for ii in range(xCols):
        xFit[:,ii] = np.linspace(np.min(X[:,ii]) - 2.5, np.max(X[:,ii]) + 2.5, xRows)
    # Calculate xFit array
    yFit = calculatePredictY(xFit, y, Beta)

    # Plot observed data with LSR fit line
    for ii in range(np.shape(X)[1]):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.grid(True)
        # plot observed data
        ax.scatter(np.array(X[:,ii]), np.array(y))
        # plot fit line
        ax.plot(xFit[:,ii], yFit, '-k')
        fig.show()

def plotResiduals(X, residuals):
    '''
    plotResiduals
        Plot residual errors

    Inputs:
        xHeader     -   Array of independent variable variable names
        X           -   Array of independent variables (x_1 ... x_n) with a leading column of ones for intercept
        residuals   -   Array of calculated y residual errors

    Outputs:
        One plot of each independent variable vs. error residuals

    Example:

    Author:
        Curtis Neiderer
    '''
    for ii in range(np.shape(X)[1]):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.grid(True)
        ax.axhline(y=0, lw=2, color='black')
        ax.scatter(np.array(X[:,ii]), np.array(residuals))
        print ii
        fig.show()

def calculateDOF(X, source):
    '''
    calculateDOF
        Calculates the degrees of freedom for given source

    Inputs:
        X       -   Array of independent variables (x_1 ... x_n) with a leading column of ones for intercept
        source  -   String designating the source of the degrees of freedom ("Regression", "Error", "Total")

    Outputs:
        dof     -   Integer designating the number degrees of freedom from the specified source

    Example:

    Author:
        Curtis Neiderer
    '''
    if source == 'Regression':
        dof = np.shape(X)[1]
    elif source == 'Error':
        dof = np.shape(X)[0] - np.shape(X)[1] - 1
    elif source == 'Total':
        dof = np.shape(X)[0] - 1

    return dof

def calculateMLEVar(residuals, dofError):
    '''
    calculateMLEVar
        Calculates maximum likelihood estimate variance of y

    Inputs:
        X           -   Array of independent variables (x_1 ... x_n) with a leading column of ones for intercept
        residuals   -   Array of calculated y residual errors
        dofError    -   Integer designating the number of degrees of freedom from residual error

    Outputs:
        mle         -   Number designating the maximum likelihood estimate of the variance of y

    Example:

    Author:
        Curtis Neiderer
    '''
    residuals = np.array(residuals)
    mle = round(np.sum(residuals ** 2) / dofError, 5)

    return mle

def calculateCovMatrix(X):
    '''
    calculateCovMatrix
        Calculates covariance matrix of X

    Inputs:
        X           -   Array of independent variables (x_1 ... x_n) with a leading column of ones for intercept

    Outputs:
        covMatrix   -   Array representing the covariance matrix of X

    Example:

    Author:
        Curtis Neiderer
    '''
    # Add column of leading ones for Beta1 intercept
    X = np.concatenate((np.matrix(np.ones(np.shape(X)[0])).T,X), axis=1)

    covMatrix = np.linalg.inv(np.dot(X.T, X))

    return covMatrix

def calculateBetaCov(mleVar, covMatrix):
    '''
    calculateBetaCov
        Calculate the Beta covariance matrix

    Inputs:
        mleVar      -   Number designating the maximum likelihood estimate of the variance of y
        covMatrix   -   Array representing the covariance matrix of X

    Outputs:
        covBeta     -   Array representing the covariance matrix of Beta

    Example:

    Author:
        Curtis Neiderer
    '''
    covBeta = mleVar * covMatrix

    return covBeta

def calculateSSR(X, y, Beta, yPred):
    '''
    calculateSSR
        Calculate regression sum of squares

    Inputs:
        X       -   Array of independent variables (x_1 ... x_n) with a leading column of ones for intercept
        y       -   Array dependent variable y values
        Beta    -   Array of calculated beta coefficients
        yPred   -   Array of predicted y values

    Outputs:
        SSR     -   Regression sum of squares

    Example:

    Author:
        Curtis Neiderer
    '''
    # Add column of leading ones for Beta1 intercept
    X = np.concatenate((np.matrix(np.ones(np.shape(X)[0])).T,X), axis=1)

    SSR = (np.dot(np.dot(Beta.T, X.T), y) - (np.sum(yPred) ** 2 / len(yPred)))[0, 0]

    return SSR

def calculateSSE(X, y, Beta):
    '''
    calculateSSE
        Calculates residual error sum of squares

    Inputs:
        X       -   Array of independent variables (x_1 ... x_n) with a leading column of ones for intercept
        y       -   Array dependent variable y values
        Beta    -   Array of calculated beta coefficients

    Outputs:
        SSE     -   Residual error sum of squares

    Example:

    Author:
        Curtis Neiderer
    '''
    # Add column of leading ones for Beta1 intercept
    X = np.concatenate((np.matrix(np.ones(np.shape(X)[0])).T,X), axis=1)

    SSE = (np.dot(y.T,y) - np.dot(np.dot(Beta.T, X.T), y))[0, 0]

    return SSE

def calculateSST(X, y, yPred):
    '''
    calculateSST
        Calculates total sum of squares

    Inputs:
        X       -   Array of independent variables (x_1 ... x_n) with a leading column of ones for intercept
        y       -   Array dependent variable y values
        yPred   -   Array of predicted y values

    Outputs:
        SST     -   Total sum of squares

    Example:

    Author:
        Curtis Neiderer
    '''
    # Add column of leading ones for Beta intercept
    X = np.concatenate((np.matrix(np.ones(np.shape(X)[0])).T,X), axis=1)

    SST = (np.dot(y.T,y) - (np.sum(yPred) ** 2 / len(yPred)))[0,0]

    return SST

def calculateMSR(SSR, dofRegress):
    '''
    calculateMSR
        Calculates mean sum of squares from regression

    Inputs:
        SSR         -   Regression sum of squares
        dofRegress  -   Regression degrees of freedom

    Outputs:
        MSR         -   Mean sum of squares from regression
    Example:

    Author:
        Curtis Neiderer
    '''
    MSR = SSR / dofRegress

    return MSR

def calculateMSE(SSE, dofError):
    '''
    calculateMSE
        Calculates mean sum of squares from residual error

    Inputs:
        SSE         -   Residual error sum of squares
        dofError    -   Residual error degrees of freedom

    Outputs:
        MSE         -   Mean sum of squares from residual error
    Example:

    Author:
        Curtis Neiderer
    '''
    MSE = SSE / dofError

    return MSE

def calculateFStat(MSR, MSE, dofRegress, dofError):
    '''
    calculateFStat
        Calculate F statistic and corresponding p-Value

    Inputs:
        MSR     -   Mean square regression
        MSE     -   Mean square error

    Outputs:
        fStat   -   Calculated f statistic value
        pVal    -   p value corresponding to f statistic value

    Example:

    Author:
        Curtis Neiderer
    '''
    fStat = MSR / MSE
    pVal = sm.stats.stattools.stats.f.sf(fStat, dofRegress, dofError)
    return fStat, pVal

def calculateR2(SSR, SSE, SST):
    '''
    calculateR2
        Calculate coefficient of multiple determination "R-Squared"

    Inputs:
        SSR     -   Regression sum of squares
        SSE     -   Residual error sum of squares
        SST     -   Total sum of squares

    Outputsimp:
        R2      -   Coefficient of multiple determination

    Example:

    Author:
        Curtis Neiderer
    '''
    # Method 1
    R2 = SSR / SST
#    # Method 2
#    R2 = 1 - (SSE / SST)

    return R2

def calculateR2Adj(SSR, SSE, SST, dofError, dofTotal):
    '''
    calculateR2Adjsm.stats.stattools.stats.t.sf(
        Calculate the adjusted coefficient of multiple determination "R-Squared"

    Inputs:
        SSR         -   Regression sum of squares
        SSE         -   Residual error sum of squares
        SST         -   Total sum of squares
        dofError    -   Residual error degrees of freedom
        dofTotal    -   Total degrees of freedom

    Outputs:
        R2      -   Adjusted coefficient of multiple determination

    Example:

    Author:
        Curtis Neiderer
    '''
    R2Adj = 1 - ((SSE / dofError) / (SST / dofTotal))

    return R2Adj

def calculateTStat(Beta, mleVar, covMatrix, dofRegress):
    '''
    calculateTStat
        Calculate the T statistic and corresponding p-value

    Inputs:
        Beta        -   Array of beta coefficients
        mleVar      -   Maximum likelihood residual error variance
        covMatrix   -   Array representing covariance matrix of X

    Outputs:
        tStat       -   Calculated T statistic
        pVal        -   Corresponding p value

    Example:

    Author:
        Curtis Neiderer
    '''
    tStat = np.ones((len(covMatrix),1))
    pVal = np.ones((len(covMatrix),1))
    for ii in range(len(covMatrix)):
        tStat[ii] = Beta[ii,0] / np.sqrt(mleVar * covMatrix[ii,ii])
        pVal = sm.stats.stattools.stats.t.sf(tStat, dofRegress) * 2

    return tStat, pVal

def calculateBetaCI(X, Beta, mleVar, covMatrix, dofError):
    '''
    calculateBetaCI
        Calculate Beta confidence intervals

    Inputs:
        X           -
        Beta        -
        mleVar      -
        covMatrix   -
        dofError    -

    Outputs:
        CI          -

    Example:

    Author:
        Curtis Neiderer
    '''
    CI = np.ones((len(Beta), 2)) * -99
    for ii, coeff in enumerate(Beta):
        tStat = np.sqrt(mleVar * covMatrix[ii, ii])
        pVal = sm.stats.stattools.stats.t.sf(tStat, dofError) * 2 # Needs to be checked
        CI[ii, 0] = coeff - tStat * pVal
        CI[ii, 1] = coeff + tStat * pVal

    return CI

def calculateMeanResponseCI(X, Beta, mleVar, covMatrix, dofError, confLevel):
    '''
    calculateMeanResponseCI
        Calculate mean response confidence intervals

    Inputs:
        X           -   Array of independent variables
        Beta        -   Array of calculated Beta coeficients
        mleVar      -   Variance of dependent variable
        covMatrix   -   Array representing the covariance matrix for independent variables
        dofError    -   Degrees of freedom due to residual error

    Outputs:
        CI          -   Confidence interval

    Example:

    Author:
        Curtis Neiderer
    '''
    # Add column of leading ones for Beta1 intercept
    X = np.concatenate((np.matrix(np.ones(np.shape(X)[0])).T,X), axis=1)

#    X = np.matrix([1, 8, 275]) # Use for testing

    CI = np.ones((np.shape(X)[0], 2)) * -99
    for ii in range(np.shape(X)[0]):
        meanResponse = np.dot(X[ii, :], Beta)
        varMeanResponse = mleVar * np.dot(np.dot(X[ii, :], covMatrix), X[ii, :].T)
#        tStat = lookupTStat(confLevel, dofError)
        tStat = sm.stats.stattools.stats.t.interval(0.95, dofError)[1]
        CI[ii, 0] = round(meanResponse - tStat * np.sqrt(varMeanResponse)[0, 0], 5)
        CI[ii, 1] = round(meanResponse + tStat * np.sqrt(varMeanResponse)[0, 0], 5)

    return CI

def calculateMeanResponsePI(X, Beta, mleVar, covMatrix, dofError, confLevel):
    '''
    calculateMeanResponsePI
        Calculate mean response prediction intervals

    Inputs:
        X           -   Array of independent variables
        Beta        -   Array of calculated Beta coeficients
        mleVar      -   Variance of dependent variable
        covMatrix   -   Array representing the covariance matrix for independent variables
        dofError    -   Degrees of freedom due to residual error

    Outputs:
        PI          -   Prediction interval

    Example:

    Author:
        Curtis Neiderer
    '''
    # Add column of leading ones for Beta1 intercept
    X = np.concatenate((np.matrix(np.ones(np.shape(X)[0])).T,X), axis=1)

#    X = np.matrix([1, 8, 275]) # Use for testing

    PI = np.ones((np.shape(X)[0], 2)) * -99
    for ii in range(np.shape(X)[0]):
        meanResponse = np.dot(X[ii, :], Beta)
        varMeanResponse = mleVar * (1 + np.dot(np.dot(X[ii, :], covMatrix), X[ii, :].T))
#        tStat = lookupTStat(confLevel, dofError)
        tStat = sm.stats.stattools.stats.t.interval(0.95, dofError)[1]
        PI[ii, 0] = round(meanResponse - tStat * np.sqrt(varMeanResponse), 5)
        PI[ii, 1] = round(meanResponse + tStat * np.sqrt(varMeanResponse), 5)

    return PI

def plotMeanResponseCIWithData(X, y, Beta, mleVar, covMatrix, dofError, confLevel):
    '''
    plotMeanResponseConfidenceIntervals
        Plot mean response confidence intervals

    Inputs:
        X           -   Array of independent variables
        Beta        -   Array of calculated Beta coeficients
        mleVar      -   Variance of dependent variable
        covMatrix   -   Array representing the covariance matrix for independent variables
        dofError    -   Degrees of freedom due to residual error

    Outputs:
        Plot of data with confidence interval lines

    Example:

    Author:
        Curtis Neiderer
    '''
    # Generate CI line fit data
    # Preallocate xFit array
    xCols = np.shape(X)[1]
    xFit = np.ones((50, xCols)) * -99
    # Calculate xFit array
    for ii in range(xCols):
        xFit[:,ii] = np.linspace(np.min(X[:,ii]) - 2.5, np.max(X[:,ii]) + 2.5, 50)
    yFit = calculateMeanResponseCI(xFit, Beta, mleVar, covMatrix, dofError, confLevel)

    # Plot observed data with CI
    for jj in range(np.shape(X)[1]):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.grid(True)
        # plot observed data
        ax.scatter(np.array(X[:, jj]), np.array(y))
        # plot fit line
        ax.plot(xFit[:, jj], yFit[:, 0], '-k', xFit[:, jj], yFit[:, 1], '-k')
        fig.show()

def plotMeanResponsePIWithData(X, y, Beta, mleVar, covMatrix, dofError, confLevel):
    '''
    plotMeanResponsePIs
        Plot mean response prediction intervals

    Inputs:
        X           -   Array of independent variables
        Beta        -   Array of calculated Beta coeficients
        mleVar      -   Variance of dependent variable
        covMatrix   -   Array representing the covariance matrix for independent variables
        dofError    -   Degrees of freedom due to residual error

    Outputs:
        Plot of data with prediction interval lines

    Example:

    Author:
        Curtis Neiderer
    '''
    # Generate PI line fit data
    # Preallocate xFit array
    xCols = np.shape(X)[1]
    xFit = np.ones((50, xCols)) * -99
    # Calculate xFit array
    for ii in range(xCols):
        xFit[:,ii] = np.linspace(np.min(X[:,ii]) - 2.5, np.max(X[:,ii]) + 2.5, 50)
    yFit = calculateMeanResponsePI(xFit, Beta, mleVar, covMatrix, dofError, confLevel)

    # Plot observed data with PI
    for jj in range(np.shape(X)[1]):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.grid(True)
        # plot observed data
        ax.scatter(np.array(X[:, jj]), np.array(y))
        # plot fit line
        ax.plot(xFit[:, jj], yFit[:, 0], '-k', xFit[:, jj], yFit[:, 1], '-k')
        fig.show()

def calculateStandardizedResiduals(residuals, MSE):
    '''
    calculateStandardizedResiduals
        Calculates the standardized residuals

    Inputs:
        residuals       -   Array of residual errors
        MSE             -   Mean 

    Outputs:
        standardizedRes -   Standardized residuals

    Example:

    Author:
    '''
    standardizedRes = residuals / np.sqrt(MSE)

    return standardizedRes

def  calculateHatMatrix(X, covMatrix):
    '''
    calculateHatMatrix
        Calculates the hat matrix for leverage comparison

    Inputs:
        X           -   Array of independent variables
        covMatrix   -   Array representing the covariance matrix for independent variables

    Outputs:
        hatMatrix   -   Array representing the Hat Matrix
        hiiArray    -   Array of the diagonal elements of the Hat Matrix

    Example:

    Author:
        Curtis Neiderer
    '''
    # Add column of leading ones for Beta intercept
    X = np.concatenate((np.matrix(np.ones(np.shape(X)[0])).T,X), axis=1)
    hatMatrix = np.dot(np.dot(X, covMatrix), X.T)

    hiiArray = np.ones((np.shape(hatMatrix)[0], 1)) * -99
    for ii in range(np.shape(hatMatrix)[0]):
        hiiArray[ii, 0] = round(hatMatrix[ii, ii], 5)

    return hiiArray, hatMatrix

def calculateStudentizedRes(residuals, mleVar, hiiArray):
    '''
    calculateStudentRes
        Calculates the studentized residual errors

    Inputs:
        residuals   -   Array of residual errors of dependent variable
        mleVar      -   Variance of dependent variable
        hiiArray    -   Array of the diagonal elements of the Hat Matrix

    Outputs:
        studentizedRes  -   Array of studentized residuals

    Example:

    Author:
        Curtis Neiderer
    '''
    studentizedRes = np.ones((len(residuals), 1))
    for ii, res in enumerate(residuals):
        studentizedRes[ii, 0] = res / np.sqrt(mleVar * (1 - hiiArray[ii, 0]))

    return studentizedRes

def calculateCooksDistance(dofRegress, studentRes, hiiArray):
    '''
    calculateCooksDistance
        Calculates Cook's distance for the residual errors

    Inputs:
        dofRegress  -   Degrees of freedom due to regression
        studentRes  -   Array of studentized residuals
        hiiArray    -   Array of the diagonal elements of the Hat Matrix  

    Outputs:
        cooksD      -   Array of Cook's Distances for each residual

    Example:

    Author:
        Curtis Neiderer
    '''
    cooksD = np.ones((len(studentRes), 1)) * -99
    for ii in range(np.shape(studentRes)[0]):
        cooksD[ii, 0] = (studentRes[ii, 0] ** 2) * hiiArray[ii, 0] / ((dofRegress + 1) * (1 - hiiArray[ii, 0]))

    return cooksD

def calculateVIF(covMatrix):
    '''
    calculateVIF
        Calculates the variance inflation factor for the beta coefficients

    Inputs:
        covMatrix   -   Array representing the covariance matrix for independent variables

    Outputs:
        VIF         -   

    Example:

    Author:
        Curtis Neiderer
    '''
    corrMatrix = np.linalg.inv(covMatrix)

    VIF = np.ones((np.shape(corrMatrix)[0], 1))
    for ii in range(np.shape(corrMatrix)[0]):
        VIF[ii, 0] = 1 / (1 - corrMatrix[ii, ii])

    return VIF

def lookupTStat(confLevel, dof):
    '''
    lookupTStat
        Lookup T Statistic value in T distribution table
    
    Inputs:
        confLevel   -   Desired confidence level (i.e., 1 - alpha)
        dof         -   Degrees of freedom
    
    Outputs:
        TStat       -   T statistic value

    Author:
        Curtis Neiderer
    
    Notes:
        No longer necessary, as I figured out how to find this value using statsmodels and pylab
    '''

    #
    tDist = np.array([[0.325, 1.000, 3.078, 6.314, 12.706, 31.821, 63.657],
                      [0.289, 0.816, 1.886, 2.920, 4.303, 6.965, 9.925],
                      [0.277, 0.765, 1.638, 2.353, 3.182, 4.541, 5.841],
                      [0.271, 0.741, 1.533, 2.132, 2.776, 3.747, 4.604],
                      [0.267, 0.727, 1.476, 2.015, 2.571, 3.365, 4.032],
                      [0.265, 0.718, 1.440, 1.943, 2.447, 3.143, 3.707],
                      [0.263, 0.711, 1.415, 1.895, 2.365, 2.998, 3.499],
                      [0.262, 0.706, 1.397, 1.860, 2.306, 2.896, 3.355],
                      [0.261, 0.703, 1.383, 1.833, 2.262, 2.821, 3.250],
                      [0.260, 0.700, 1.372, 1.812, 2.228, 2.764, 3.169],
                      [0.260, 0.697, 1.363, 1.796, 2.201, 2.718, 3.106],
                      [0.259, 0.695, 1.356, 1.782, 2.179, 2.681, 3.055],
                      [0.259, 0.694, 1.350, 1.771, 2.160, 2.650, 3.012],
                      [0.258, 0.692, 1.345, 1.761, 2.145, 2.624, 2.997],
                      [0.258, 0.691, 1.341, 1.753, 2.131, 2.602, 2.947],
                      [0.258, 0.690, 1.337, 1.746, 2.120, 2.583, 2.921],
                      [0.257, 0.689, 1.333, 1.740, 2.110, 2.567, 2.898],
                      [0.257, 0.688, 1.330, 1.734, 2.101, 2.552, 2.878],
                      [0.257, 0.688, 1.328, 1.729, 2.093, 2.539, 2.845],
                      [0.257, 0.687, 1.325, 1.725, 2.086, 2.528, 2.845],
                      [0.257, 0.686, 1.323, 1.721, 2.080, 2.518, 2.831],
                      [0.256, 0.686, 1.321, 1.717, 2.074, 2.508, 2.819],
                      [0.256, 0.685, 1.319, 1.714, 2.069, 2.500, 2.807],
                      [0.256, 0.685, 1.318, 1.711, 2.064, 2.492, 2.797],
                      [0.256, 0.684, 1.316, 1.708, 2.060, 2.485, 2.787],
                      [0.256, 0.684, 1.315, 1.706, 2.056, 2.479, 2.779],
                      [0.256, 0.684, 1.314, 1.703, 2.052, 2.473, 2.771],
                      [0.256, 0.683, 1.313, 1.701, 2.048, 2.467, 2.763],
                      [0.256, 0.683, 1.311, 1.699, 2.045, 2.462, 2.756],
                      [0.256, 0.683, 1.310, 1.697, 2.042, 2.457, 2.750], 
                      [0.253, 0.674, 1.282, 1.645, 1.960, 2.326, 2.576]])

    # Get column range                      
    if confLevel == 0.60:
        confCol = 0
    elif confLevel == 0.75:
        confCol = 1
    elif confLevel == 0.90:
        confCol = 3
    elif confLevel == 0.95:
        confCol = 4
    elif confLevel == 0.975:
        confCol = 5
    elif confLevel == 0.99:
        confCol = 6
    else:
        print 'Confidence Level not found in table, \n'
        print 'please select level from table: 0.60, 0.75, 0.90, 0.95, 0.975, 0.99'
        
    # Get row
    if dof <=30 :
        dofRow = dof - 1
    else:
        dofRow = 30

    # Lookup value in tDist Table
    tStat = tDist[dofRow, confCol]      
        
    return tStat                     


########### ----- Enables Command Line Call ----- ##########
if __name__ == '__main__':
    main()
########### ----- Enables Command Line Call ----- ##########