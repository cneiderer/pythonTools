# -*- coding: utf-8 -*-
"""
Created on Sat Jul 05 09:51:13 2014

@author: Curtis
"""
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import math
import os

########### ----- Begin Main Function ----- ##########
def main():

#    fig_h, ax_h = create_m_of_n_prob_curve(2, 3, single_prob=0.75, saveFig=False)
#    fig_h.show() 
#    fig_h, ax_h = create_m_of_n_prob_curve(3, 5, single_prob=0.75, saveFig=False)
#    fig_h.show()    
#    fig_h, ax_h = create_m_of_n_prob_curve(4, 7, single_prob=0.75, saveFig=False)
#    fig_h.show() 
#    fig_h, ax_h = create_m_of_n_prob_curve(5, 9, single_prob=0.75, saveFig=False)
#    fig_h.show() 
    
    fig_h, ax_h = create_m_of_n_prob_curves(single_prob=None, 
                                            saveFig=True, 
                                            savePath=None)                                         
    
########### ----- End Main Function ----- ##########



def create_m_of_n_prob_curves(single_prob=None, 
                              saveFig=True, 
                              savePath=None):
    '''
    create_m_of_n_prob_curves(float, 
                              boolean, 
                              string) 
                              -> handle, handle

    Description:
    
    Inputs:
    
    Outputs:
    
    Example:
    
    Author:
        Curtis Neiderer, July 2014
    '''                             
    
    # Initialize handles
    fig_h=None
    ax_h=None
    # Define m of n combo list
    combo_list = [[2, 3], [3, 5], [4, 7], [5, 9]]
    # Iterate through m of n combo list
    for combo in combo_list:
        # Add curve to axis
        [fig_h, ax_h] = create_single_m_of_n_prob_curve(combo[0], 
                                                        combo[1], 
                                                        fig_h=fig_h, 
                                                        ax_h=ax_h, 
                                                        single_prob=single_prob,
                                                        saveFig=False)
    # Add legend to axis                                                        
    ax_h.legend(loc='center right')
    
    # Define and add plot title
    plot_title = 'Binomial Aggregation Probability Curves'
    ax_h.set_title(plot_title)
    # Define and add figure window title
    fig_title = 'BinomialAggregationProbabilityCurves'
    fig_h.canvas.set_window_title(fig_title)
    
    #
    if not single_prob is None:
        plot_title = plot_title + ' at ' + str(int(round(single_prob * 100))) + '%'
        ax_h.set_title(plot_title) 
        fig_title = fig_title + '_' + str(int(round(single_prob * 100))) + 'percent'
        fig_h.canvas.set_window_title(fig_title)

    # Check if saveFig flag is set to True
    if saveFig is True:
        # Check if savePath string is defined
        if savePath is None:
            # Define savePath string if not defined
            savePath = os.getcwd() + '\\'
        # Save figure
        plt.savefig(fig_title + '.png')
    
        # Re-define axes limits
        ax_h.axes.set_xlim([0.5, 1.0])                                        
        ax_h.axes.set_ylim([0.5, 1.0]) 
        # Format x-ticks
        ax_h.set_xticks(np.arange(0.5, 1.05, 0.05)) # Major
        ax_h.set_xticks(np.arange(0.5, 1.025, 0.025), 
                        minor=True) # Minor
        # Format y-ticks
        ax_h.set_yticks(np.arange(0.5, 1.05, 0.05)) # Major
        ax_h.set_yticks(np.arange(0.5, 1.025, 0.025), 
                        minor=True) # Minor    
        plt.savefig(fig_title + '_Zoomed.png')                        
    
    return fig_h, ax_h
    
def m_of_n_prob(m, n, prob):
    '''
    m_of_n_prob(integer, integer, float) -> float
    
    Description:
    
    Inputs:
    
    Outputs:
    
    Example:
    
    Author:
        Curtis Neiderer, July 2014
    '''
    summation = 0
    for k in range(m):
        summation += n_choose_k(n, k) * (prob ** k) * ((1 - prob) ** (n - k))
        
    return 1 - summation


def n_choose_k(n, k):
    '''
    n_choose_k(integer, integer) -> integer
    
    Description:
    
    Inputs:
    
    Outputs:
    
    Example:
    
    Author:
        Curtis Neiderer, July 2014
    '''
    return math.factorial(n) / (math.factorial(k) * math.factorial(n - k))


def create_single_m_of_n_prob_curve(m, 
                                    n, 
                                    single_prob=None, 
                                    fig_h=None, 
                                    ax_h=None, 
                                    saveFig=True, 
                                    savePath=None):
    '''
    create_single_m_of_n_prob_curve(integer, 
                                    integer, 
                                    float, 
                                    boolean, 
                                    string) 
                                    -> handle, handle

    Description:
    
    Inputs:
    
    Outputs:
    
    Example:
    
    Author:
        Curtis Neiderer, July 2014
    '''                             
    # Allocate input probability 
    in_prob = np.arange(0, 1.01, 0.01)
    # Initialize output probability
    out_prob = []
    # Iterate across single observation probably range
    for prob in in_prob:
        # Calculate out probability range for desired m of n
        out_prob.append(m_of_n_prob(m, n, prob))

    # Create figure
    if fig_h is None:
        fig_h = plt.figure()
    # Add axis
    if ax_h is None:
        ax_h = fig_h.add_subplot(1, 1, 1)
        ax_h.grid(True)
        
#    fig_h, ax_h = plt.subplots(figsize=(12, 8), frameon=True)
    # Plot single observation probability vs. multi observation probability
    ax_h.plot(in_prob, 
              out_prob, 
              linewidth=2,
              label=str(m) + ' of ' + str(n))
    
    if not single_prob is None:    
        multi_prob = m_of_n_prob(m, n, single_prob)
        # Vertical Line
        ax_h.axvline(x=single_prob, 
                     ymin=0, 
                     ymax=multi_prob, 
                     linewidth=2, 
                     linestyle='--',
                     color='k')
        # Horizontal Line
        ax_h.axhline(y=multi_prob, 
                     xmin=0, 
                     xmax=single_prob, 
                     linewidth=2, 
                     linestyle='--',
                     color='k')
    
    # Add major gridlines
    ax_h.grid(b='on', which='major', axis='both')
    # Format x-ticks
    ax_h.set_xticks(np.arange(0, 1.25, 0.25)) # Major
    ax_h.set_xticks(np.arange(0, 1.05, 0.05), 
                    minor=True) # Minor
    # Format y-ticks
    ax_h.set_yticks(np.arange(0, 1.25, 0.25)) # Major
    ax_h.set_yticks(np.arange(0, 1.05, 0.05), 
                    minor=True) # Minor
    # Define and add plot title
    plot_title = 'Binomial Aggregation Probability Curve: ' + str(m) + ' of ' + str(n)
    ax_h.set_title(plot_title)
    # Define and add figure window title
    fig_title = 'BinomialAggregationProbabilityCurve_' + str(m) + 'of' + str(n)
    fig_h.canvas.set_window_title(fig_title)
    
    # Check if saveFig flag is set to True
    if saveFig is True:
        # Check if savePath string is defined
        if savePath is None:
            # Define savePath string if not defined
            savePath = os.getcwd() + '\\'
        # Save figure
        fig_h.savefig(fig_title + '.png')
    
    return fig_h, ax_h
    
########### ----- Enables Command Line Call ----- ##########
if __name__ == '__main__':
    main()