# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 21:19:46 2021

@author: TerryYang
"""
import math
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0,8*np.pi,0.1)   # start,stop,step
y = np.cos(x)-x/6

plt.plot(x,y)
plt.show()

x = np.arange(0,16*np.pi,0.1)   # start,stop,step
y = np.cos(x/2)-x/12

plt.plot(x,y)
plt.show()