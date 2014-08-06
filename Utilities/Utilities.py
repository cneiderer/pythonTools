# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 08:46:14 2014

@author: Curtis.Neiderer
"""

import os
import matplotlib.pyplot as plt

########## ----- Main Function ----- ##########
def main():
    clc()
    clearConsole()
    pwd()
    closeAll()
    
########## ----- Main Function ----- ##########

def clc():
    '''
    clc() -> None
    
    Description:
        Command line clear (as in MATLAB), clears the console
    
    Inputs:
        None
        
    Output:
        None

    Reference / Notes:

        
    Author:
        Curtis Neiderer, 2/2014
    '''
    os.system(['clear', 'cls'][os.name == 'nt'])
    
def clearConsole():
    '''
    clearConsole() -> None
    
    Description:
        Command line clear, clears the console
    
    Inputs:
        None
        
    Output:
        None

    Reference / Notes:


    Author: 
        Curtis Neiderer, 2/2014
    '''
    os.system(['clear', 'cls'][os.name == 'nt'])    

def pwd():
    '''
    pwd() -> String representing the currect working directory
    
    Description:
        Print working directory (as in MATLAb and UNIX/LINUX), returns a string representing the current working directory
    
    Inputs:
        None
        
    Output:
        None

    Reference / Notes:


    Author: 
        Curtis Neiderer, 2/2014
    '''
    print os.getcwd()        

def closeAll():
    '''
    closeAll() -> None
    
    Description:
        Close all open figures
    
    Inputs:
        None
        
    Output:
        None

    Reference / Notes:


    Author: 
        Curtis Neiderer, 2/2014
    '''
    plt.close('all')
    
########### ----- Enables Command Line Call ----- ##########
if __name__ == '__main__':
    main()
########### ----- Enables Command Line Call ----- ##########