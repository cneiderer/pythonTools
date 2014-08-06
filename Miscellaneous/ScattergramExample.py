# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 13:43:11 2014

@author: Curtis.Neiderer
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd

########## ----- Main Function ----- ##########
def main():
    
    scatterGramExample(withPandasDF=0)
    
    scatterGramExample(withPandasDF=1)
    
    test = 1

########## ----- End Main Function ----- ##########

def scatterGramExample(withPandasDF=0):
    
    # the random data
    x = np.random.randn(1000)
    y = np.random.randn(1000)
    
    # Move data into DataFrame
    data = pd.DataFrame([x, y]).T
    
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
        fig.suptitle('Numpy Arrays')
    else:
        axHistx.hist(data[data.columns[0]], bins)
        axHisty.hist(data[data.columns[1]], bins=bins, orientation='horizontal')
        fig.suptitle('Pandas DataFrame Columns (i.e., Pandas Series)')
    
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
    
    ########### ----- Enables Command Line Call ----- ##########
if __name__ == '__main__':
    main()
########### ----- Enables Command Line Call ----- ##########