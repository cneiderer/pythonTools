# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 07:08:03 2014

@author: Curtis.Neiderer
"""

from __future__ import division
import numpy as np
import statsmodels.api as sm

########## ----- Main Function ----- ##########
def main():
        
        p1 = 0.5
        p2 = 0.51     
        alpha = 0.05
        beta = 0.1
        
        print '\np1 = ' + str(p1), ', p2 = ' + str(p2) + '\n'
        
        print 'Samples Necessary:'
        print 'Method 1: ', str(calculateSampleSizeForBinaryOutcomeData_M1(p1, p2)) + \
        ', alpha = 0.05, beta = 0.2'
        print 'Method 2: ', str(calculateSampleSizeForBinaryOutcomeData_M2(p1, p2, alpha, beta)) + \
        ', alpha = ' + str(alpha) + ', beta = ' + str(beta)
        print 'Method 3: ', str(calculateSampleSizeForBinaryOutcomeData_M3(p1, p2, alpha, beta)) + \
        ', alpha = ' + str(alpha) + ', beta = ' + str(beta)
        print 'Method 4: ', str(calculateSampleSizeForBinaryOutcomeData_M4(p1, p2, alpha, beta)) + \
        ', alpha = ' + str(alpha) + ', beta = ' + str(beta)
        
        print '\nUpper Limit on Sample Size:'
        print 'Method 5: ', str(calculateSampleSizeForBinaryOutcomeData_M5(p1, p2)) + \
        ', alpha = 0.05, beta = 0.2'
        print '\nSamples Necessary:'
        print 'Method 6: ', str(calculateSampleSizeForBinaryOutcomeData_M6(p1, p2)) + \
        ', alpha = 0.05, beta = 0.2'
        
        
########## ----- Main Function ----- ########## 

def calculateSampleSizeForBinaryOutcomeData_M1(pA, pB):
    '''
    CalculateSampleSizeForBinaryOutcomeData_M1
        An approximate calculation of sample size necessary for binary outcome
        data.  This equation is valid for alpha = 0.05 and beta = 0.2 (80% power).
        This equation also slightly overestimates sample size.
    
    Inputs:
        pA      -   proportions expected in group 1
        pB      -   proportions expected in group 2
        
    Outputs:
        m   -   necessary sample size
    
    Example:
    
    Author:
        Curtis Neiderer, 2/18/2014
    
    Reference:
        "Estimating Sample Sizes for Binary, Ordered Categorical, and
        Continuous Outcomes in Two Group Comparisons", M.J. Campbell, S.A. 
        Julious, D.G. Altman, 28 October 1995    
    '''
    pBar = (pA + pB) / 2
    m = (16 * pBar * (1 - pBar)) / (pA - pB) ** 2

    return int(np.ceil(m))
    
def calculateSampleSizeForBinaryOutcomeData_M2(pA, pB, alpha, beta):
    '''
    CalculateSampleSizeForBinaryOutcomeData_M2
        Calculation of sample size necessary for binary outcome data.  
        Equation 3 from reference.
    
    Inputs:
        pA      -   proportions expected in group 1
        pB      -   proportions expected in group 2
        alpha   -
        beta    -
        
    Outputs:
        m   -   necessary sample size
    
    Example:
    
    Author:
        Curtis Neiderer, 2/18/2014
    
    Reference:
        "Estimating Sample Sizes for Binary, Ordered Categorical, and
        Continuous Outcomes in Two Group Comparisons", M.J. Campbell, S.A. 
        Julious, D.G. Altman, 28 October 1995    
    '''
    delta = pA - pB
    pBar = (pA + pB) / 2
    zAlpha = sm.stats.stattools.stats.norm.ppf(1 - (alpha / 2))
    zBeta = sm.stats.stattools.stats.norm.ppf(1 - beta)
    m = ((zAlpha * np.sqrt(2 * pBar * (1 - pBar))) + (zBeta * np.sqrt(pA * (1 - pA) + pB * (1 - pB)))) ** 2 / delta ** 2
    
    return int(np.ceil(m))
    
def calculateSampleSizeForBinaryOutcomeData_M3(pA, pB, alpha, beta):
    '''
    CalculateSampleSizeForBinaryOutcomeData_M3
        Calculation of sample size necessary for binary outcome data.  
        Equation 4 from reference.  Approximate formula sufficiently accurate
        except when pA, pB are small (say < 0.05)
    
    Inputs:
        pA      -   proportions expected in group 1
        pB      -   proportions expected in group 2
        alpha   -
        beta    -
        
    Outputs:
        m   -   necessary sample size
    
    Example:
    
    Author:
        Curtis Neiderer, 2/18/2014
    
    Reference:
        "Estimating Sample Sizes for Binary, Ordered Categorical, and
        Continuous Outcomes in Two Group Comparisons", M.J. Campbell, S.A. 
        Julious, D.G. Altman, 28 October 1995    
    '''
    delta = pA - pB
    zAlpha = sm.stats.stattools.stats.norm.ppf(1 - (alpha / 2))
    zBeta = sm.stats.stattools.stats.norm.ppf(1 - beta)
    m = (((zAlpha + zBeta) ** 2) * ((pA * (1 - pA)) + (pB * (1 - pB)))) / delta ** 2
    
    return int(np.ceil(m))
        
def calculateSampleSizeForBinaryOutcomeData_M4(pA, pB, alpha, beta):
    '''
    CalculateSampleSizeForBinaryOutcomeData_M4
        Calculation of sample size necessary for binary outcome data.  
        Equation 5 from reference.  
    
    Inputs:
        pA      -   proportions expected in group 1
        pB      -   proportions expected in group 2
        alpha   -
        beta    -
        
     Outputs:
        m   -   necessary sample size
    
    Example:
    
    Author:
        Curtis Neiderer, 2/18/2014
    
    Reference:
        "Estimating Sample Sizes for Binary, Ordered Categorical, and
        Continuous Outcomes in Two Group Comparisons", M.J. Campbell, S.A. 
        Julious, D.G. Altman, 28 October 1995    
    '''
    oddsRatio = (pA * (1 - pB)) / (pB * (1 - pA))
    pBar = (pA + pB) / 2
    zAlpha = sm.stats.stattools.stats.norm.ppf(1 - (alpha / 2))
    zBeta = sm.stats.stattools.stats.norm.ppf(1 - beta)

    m = (2 * ((zAlpha + zBeta)** 2)) / ((np.log(oddsRatio) ** 2) * pBar * (1 - pBar))
    
    return int(np.ceil(m))

def calculateSampleSizeForBinaryOutcomeData_M5(p0, p1):
    '''
    CalculateSampleSizeForBinaryOutcomeData_M5
        Calculation for the upper limit sample size necessary for binary 
        outcome data.  Equation 2.28 from reference.  Valid only when 
        alpha = 0.05 and beta = 0.2 (80% power).
    
    Inputs:
        pA      -   proportions expected in group 1
        pB      -   proportions expected in group 2
        
     Outputs:
        n   -   necessary sample size
    
    Example:
    
    Author:
        Curtis Neiderer, 2/18/2014
    
    Reference:
        Chapter 2: Sample Size, 
        Section 2.9 - Sample Size Calculation for the Binomial Distribution         
    '''
    n = 4 / (p0 - p1) ** 2
    
    return int(np.ceil(n))
    
def calculateSampleSizeForBinaryOutcomeData_M6(p0, p1):
    '''
    CalculateSampleSizeForBinaryOutcomeData_M4
        Calculation of sample size necessary for binary outcome data.  
        Equation 2.30 from reference.  Valid only when alpha = 0.05 and
        beta = 0.2 (80% power)
    
    Inputs:
        pA      -   proportions expected in group 1
        pB      -   proportions expected in group 2
        
     Outputs:
        m   -   necessary sample size
    
    Example:
    
    Author:
        Curtis Neiderer, 2/18/2014
    
    Reference:
        Chapter 2: Sample Size, 
        Section 2.9 - Sample Size Calculation for the Binomial Distribution
    '''
    n = 4 / (np.arcsin(np.sqrt(p0)) - np.arcsin(np.sqrt(p1))) ** 2
    
    return int(np.ceil(n))
    
########### ----- Enables Command Line Call ----- ##########
if __name__ == '__main__':
    main()
########### ----- Enables Command Line Call ----- ##########