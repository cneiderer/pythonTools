# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 13:18:58 2014

@author: Curtis.Neiderer
"""
import numpy as np

x = [0.18, -1.54, 0.42, 0.95]
w = [2, 1, 3, 1]
m = [0.3, 1.077, 0.0025, 0.1471]

for idx, val in enumerate(m):
    total_sum = 0
    for i in range(len(x)):
        total_sum += w[i] * (x[i] - val) ** 2
    print "Mu = " + str(val) + ": " + str(total_sum)
