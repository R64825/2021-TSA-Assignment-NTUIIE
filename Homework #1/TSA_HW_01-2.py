# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 17:22:52 2021

@author: TerryYang
"""
import math
import numpy as np
import matplotlib.pyplot as plt

def Predict_a_Time_Series(Time_length):
    #    predict a time series
    predictions = []
    for epoch in range(0,Time_length+1):
        value = Make_Prediction(epoch)
        predictions.append(value)
    return predictions

def Make_Prediction(epoch):
    #    return prediction value. 
    value = math.cos(2*math.pi*((epoch/12)+(np.random.uniform(0,1))))
    return value

Predictions = Predict_a_Time_Series(48)
plt.plot(Predictions)
plt.show()
