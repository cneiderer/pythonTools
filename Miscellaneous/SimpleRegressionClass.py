# -*- coding: utf-8 -*-
"""
Created on Thu Feb 06 14:56:53 2014

@author: Curtis.Neiderer
"""

from __future__ import division
import pylab as py
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

########## ----- Main Function ----- ##########
def main():
    Xin = np.matrix([[70, 74, 72, 68, 58, 54, 82, 64, 80, 61]]).T
    Xin = np.concatenate((np.matrix(np.ones(np.max(np.shape(Xin)))).T,Xin), axis=1)
    yin = np.matrix([[77, 94, 88, 80, 71, 76, 88, 80, 90, 69]]).T
             
    X = np.matrix([[2, 8, 11, 10, 8, 4, 2, 2, 9, 8, 4, 11, 12, 2, 4, 4, 20, 
                      1, 10, 15, 15, 16, 17, 6, 5], [50, 110, 120, 550, 295, 
                      200, 375, 52, 100, 300, 412, 400, 500, 360, 205, 400, 
                      600, 585, 540, 250, 290, 510, 590, 100,400]]).T
    X = np.concatenate((np.matrix(np.ones(np.max(np.shape(X)))).T,X), axis=1)
    y = np.matrix([[9.95, 24.45, 31.75, 35.00, 25.02, 16.86, 14.38, 9.60, 
                      24.35, 27.50, 17.08, 37.00, 41.95, 11.66, 21.65, 17.89, 
                      69.00, 10.30, 34.93, 46.59, 44.88, 54.12, 56.63, 22.13, 
                      21.15]]).T
             
    lsrObj = lsrLine(X, y)
    print lsrObj
    
    lsrObj.plotDataWithLSRLine()
########## ----- Main Function ----- ##########   
    
def calculateBeta(Xin, yin):
    ''' 
    calculateBeta
        Calculate Beta coefficients given X matrix of independent variables and 
        y vector of dependent dependent variable
    
    Inputs:
    
    Outputs:
    
    Example:
    
    Author:
    '''
    Beta = np.dot(np.linalg.inv(np.dot(Xin.T, Xin)), np.dot(Xin.T, yin))
    
    return Beta      

def generateLSRFitData(self, xData, Beta):
    xCols = np.shape(xData)[1]
    xRows = 10
    xFit = np.ones((xRows, xCols)) * -99
    yFit = round(Beta[0,0], 5)                    
    for ii, coeff in enumerate(Beta[1:]):
        coeff = round(coeff[0,0], 5)
        xFit[:,ii] = np.linspace(np.min(xData[:,ii]) - 2.5, np.max(xData[:,ii]) + 2.5, xRows)
        yFit += coeff * xFit[:, ii]   
    
    return xFit, yFit

class lsrLine:
    '''
    Class to calculate and define LSR fit line
    '''
    def __init__(self, xData, yData):
        self.xData = xData
        self.yData = yData
        self.Beta = calculateBeta(xData, yData)
        
        
    def __str__(self):
        string = 'y = ' + str(round(self.Beta[0, 0], 5))
        for ii, coeff in enumerate(self.Beta[1:]):
            coeff = round(coeff[0,0], 5)
            if coeff == 0.0:
                continue
            elif (coeff < 0):
                sign = ' - '                
            else:
                sign = ' + '
            if len(self.Beta) <= 2:
                string += sign + str(abs(coeff)) + '*x'
            else:
                string += sign + str(abs(coeff)) + '*x' + str(ii+1)
        return string
    
    def calculatePredictY(self):
        y = round(self.Beta[0,0], 5)
        xData = self.xData[:,1:]
        for ii, coeff in enumerate(self.Beta[1:]):
            coeff = round(coeff[0,0], 5)
            y += coeff * xData[:, ii]
        self.yPred = y
        
        return y
    
    def plotDataWithLSRLine(self):
        ''' Plot data with Least Squares Regression line '''
        xData = self.xData[:,1:]
        yData = self.yData
        Beta = self.Beta
        self.xFit, self.yFit = generateLSRFitData(self, xData, Beta)
        xFit = self.xFit
        yFit = self.yFit
        
        for ii in range(np.shape(xData)[1]):
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.grid(True)
            # plot observed data
            ax.scatter(np.array(xData[:,ii]), np.array(yData))
            # plot fit line       
            ax.plot(xFit[:,ii], yFit, '-')
            fig.show()
        
    def generateFitData(self):
        xData = self.xData[:,1:]
        
        xCols = np.shape(xData)[1]
        xRows = 10
        xFit = np.ones((xRows, xCols)) * -99
        yFit = round(self.Beta[0,0], 5)                    
        for ii, coeff in enumerate(self.Beta[1:]):
            coeff = round(coeff[0,0], 5)
            xFit[:,ii] = np.linspace(np.min(xData[:,ii]) - 2.5, np.max(xData[:,ii]) + 2.5, xRows)
            yFit += coeff * xFit[:, ii]   
        self.xFit = xFit
        self.yFit = yFit
        
        return xFit, yFit
    
    def calculateResiduals(self):
        yData = self.yData
        yPred = self.yPred
        residuals = yData - yPred
        self.residuals = residuals
        
        return residuals
    
    def plotResiduals(self):
        residuals = self.residuals
        xData = self.xData[:,1:]
        
        for ii in range(np.shape(xData)[1]):
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.grid(True)
            ax.axhline(y=0, lw=2, color='black')
            ax.scatter(np.array(xData[:,ii]), np.array(residuals))
            print ii
            fig.show()

    def calculateCovarianceMatrix(self):
        xData = self.xData
        covMatrix = np.linalg.inv(np.dot(xData.T, xData))
        self.covMatrix = covMatrix
        
        return covMatrix
        
    def calculateMLEVar(self):
        xData = self.xData
        residuals = self.residuals
        mle = np.sum(np.dot(residuals.T, residuals)) / (np.shape(xData[:,1:])[0] - np.shape(xData)[1])
        self.mle = mle
        
        return mle
    
    def calculateCovarianceBeta(self):
        mle = self.mle
        covMatrix = self.mle
        covBeta = mle * covMatrix
        self.covBeta = covBeta
        
        return covBeta    
    
    def calculateDoFError(self):
        xData = self.xData
        dofError = np.shape(xData)[0] - np.shape(xData)[1]
        self.dofError = dofError
        
        return dofError
    
    def calculateDOFRegression(self):
        xData = self.xData
        dofRegress = np.shape(xData[:,1:])[1]    
        self.dofReg = dofRegress
        
        return dofRegress
    
    def calculateDoFTotal(self):
        xData = self.xData
        dofTotal = np.shape(xData[:,1:])[0] - 1
        self.dofTotal = dofTotal
        
        return dofTotal

    
########### ----- Enables Command Line Call ----- ##########          
if __name__ == '__main__':
    main()
########### ----- Enables Command Line Call ----- ##########