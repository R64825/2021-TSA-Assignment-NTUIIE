# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 17:07:35 2021

@author: TerryYang
"""
import statsmodels.api as sm
from statsmodels import tsa as TSA
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


'''
Q2
'''

df = pd.read_csv(r"C:\Users\TerryYang\Desktop\Github\2021-TSA-Assignment-NTUIIE\Homework #8\TSA HW06.robot.csv")
cut = 5
df_train = df[:-cut]
df_test = df[-cut:]

alpha = 0.05
#(a)
IMA = ARIMA(df_train, order=(0,1,1)).fit()
IMA.summary()

IMA_Prediction = IMA.get_prediction(start=len(df_train), end=len(df_train)+len(df_test)-1)
IMA_Prediction.summary_frame(alpha = alpha)

#(b)
Pred_coef = IMA_Prediction.predicted_mean
Pred_coef_itv = IMA_Prediction.conf_int(alpha=alpha)

fig = plt.figure()
ax = fig.add_subplot()
ax.plot(df_test, color = "red", marker = 'o', label="actual")
ax.plot(Pred_coef, color = "blue", marker = 'o', label="prediction")
ax.fill_between(x = Pred_coef_itv.index, y1 = Pred_coef_itv.iloc[:,0], y2 = Pred_coef_itv.iloc[:,1], color = "gray", alpha = 0.5)
ax.legend()
plt.show()

#(c)
ARIMA = ARIMA(df_train, order=(1,0,1)).fit()
ARIMA.summary()

ARIMA_Prediction = ARIMA.get_prediction(start=len(df_train), end=len(df_train)+len(df_test)-1)
ARIMA_Prediction.summary_frame(alpha = alpha)

Pred_coef = ARIMA_Prediction.predicted_mean
Pred_coef_itv = ARIMA_Prediction.conf_int(alpha=alpha)

fig = plt.figure()
ax = fig.add_subplot()
ax.plot(df_test, color = "red", marker = 'o', label="actual")
ax.plot(Pred_coef, color = "blue", marker = 'o', label="prediction")
ax.fill_between(x = Pred_coef_itv.index, y1 = Pred_coef_itv.iloc[:,0], y2 = Pred_coef_itv.iloc[:,1], color = "gray", alpha = 0.5)
ax.legend()
plt.show()

'''
Q3
'''
df = pd.read_csv(r"C:\Users\TerryYang\Desktop\Github\2021-TSA-Assignment-NTUIIE\Homework #8\TSA HW08.boardings.csv")

#(a)
plt.plot(df, marker = 'o')

#(b)
acf = TSA.stattools.acf(df, nlags=len(df), fft=False)
plt.plot(acf)

#(c)
SARIMA=sm.tsa.statespace.SARIMAX(endog=df,order=(0,0,3),seasonal_order=(1,0,0,12),trend='c',enforce_invertibility=False).fit()
print(SARIMA.summary())

'''
Q4
'''
df = pd.read_csv(r"C:\Users\TerryYang\Desktop\Github\2021-TSA-Assignment-NTUIIE\Homework #8\TSA HW08.airpass.csv")

#(a)
plt.plot(df, marker = 'o', markersize = 3)
plt.plot(np.log(df), marker = 'o', markersize = 3)

#(b)
plt.plot(np.log(df).diff(periods = 1), marker = 'o', markersize = 3)

#(c)
plt.plot(np.log(df).diff(periods = 12), marker = 'o', markersize = 3)

#(d)
series = np.log(df).diff(periods = 12)
series = series.dropna()
acf = TSA.stattools.acf(series, nlags=len(series)-1, fft=False)
fig = sm.graphics.tsa.plot_acf(series, lags=len(series)-1)

#(e)
SARIMA=sm.tsa.statespace.SARIMAX(endog=np.log(df),order=(0,0,3),seasonal_order=(1,0,0,12),trend='c',enforce_invertibility=False).fit()
print(SARIMA.summary())

SARIMA.plot_diagnostics
plt.show()

#(f)
SARIMA_pre = SARIMA.get_prediction(start=len(np.log(df)), end=len(np.log(df))+24)
SARIMA_pre.predicted_mean

Pred_coef = SARIMA_pre.predicted_mean
Pred_coef_itv = SARIMA_pre.conf_int(alpha=alpha)

fig = plt.figure(figsize = (15,5))
ax = fig.add_subplot()
ax.plot(np.log(df), color = "red", marker = 'o', label="history")
ax.plot(SARIMA_pre.predicted_mean, color = "blue", marker = 'o', label="prediction")
ax.fill_between(x = Pred_coef_itv.index, y1 = Pred_coef_itv.iloc[:,0], y2 = Pred_coef_itv.iloc[:,1], color = "gray", alpha = 0.5)
ax.legend()
plt.show()
