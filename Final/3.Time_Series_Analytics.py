# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 16:44:39 2022

@author: TerryYang
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as dates
import statsmodels.api as sm
import itertools
from pmdarima.arima import auto_arima
from math import sqrt
from sklearn.metrics import mean_squared_error
from pmdarima.arima.stationarity import ADFTest
import itertools
from statsmodels.tsa.arima_model import ARIMA
from pmdarima.arima import ndiffs
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf
import warnings
warnings.filterwarnings("ignore")

input_path = r"C:\Users\TerryYang\(2021)_Preprocessed.csv"
train_Ratio = 0.8
poi = "AQI"

def Plot_Time_Series(X, y, s , c):
    fig = plt.figure()
    plt.plot(X, y)
    plt.title(s+" "+ poi +" Changes at Wanhua, 2021")
    plt.xlabel('Time')
    plt.ylabel(poi)
    fig.set_figheight(8)
    fig.set_figwidth(15)

    plt.plot(X,y, color=c)
    plt.show()
    
def OLS_Res(y):
    X = []
    for i in range(0, len(y)):
        X.append(i)
    reg = sm.OLS(y, X)
    result = reg.fit()
    # print result
    print("parameters: \n", result.params)
    print("\n", result.summary())


# read and split dataset
df = pd.read_csv(input_path)
df['Date'] =pd.to_datetime(df.Date)
df = df.drop(columns = ['Unnamed: 0'])
df = df[[poi, "Date"]]

df_Hour = df

df_Day = df.groupby([df['Date'].dt.date]).mean()
df_Day['Date'] = df_Day.index
del df

# visialize
Plot_Time_Series(df_Hour.Date, df_Hour[poi], "Hourly" , "blue")
Plot_Time_Series(df_Day.Date, df_Day[poi], "Daily", "green")

# checking staionary
adf_test = ADFTest(alpha=0.05)
adf_test.should_diff(df_Day[poi])
train = df_Day[:int(train_Ratio*(len(df_Day)))]
test = df_Day[int(train_Ratio*(len(df_Day))):]
train.shape
test.shape
plt.plot(train.AQI)
plt.plot(test.AQI)
plt.title("Training and Test Data")
plt.show()

# find p, d, q
d_value = ndiffs(df_Day[poi],alpha=0.05, test='adf',max_d=1000)


plot_pacf(df_Day[poi], lags = (len(df_Day)/4), title="Daily PACF")
plot_acf(df_Day[poi], lags = (len(df_Day)/4), title="Daily ACF")
plot_pacf(df_Hour[poi], lags = (len(df_Day)/4), title="Hourly PACF")
plot_acf(df_Hour[poi], lags = (len(df_Day)/4), title="Hourly ACF")

#
train = df_Day[:int(train_Ratio*(len(df_Day)))]
test = df_Day[int(train_Ratio*(len(df_Day))):]


p_value = 0
q_value = 0
aic_value = np.Inf
for p in range(0,2):
    for q in range(0,22):
        print(p, q)
        model = sm.tsa.SARIMAX(train[poi], trend='c', order=(p,0,q), enforce_stationarity=False, enforce_invertibility=False)
        #ARIMA(train[poi],order=(p,0,q),freq=None)   
        model_fit = model.fit(disp=0)      
        if model_fit.aic < aic_value:           
            p_value = p       
            q_value = q
            aic_value = model_fit.aic
print(p_value, q_value, aic_value)
best_AIC = ARIMA_AIC(df_Day.AQI, 5, 1, 5)

model = sm.tsa.SARIMAX(train[poi], trend='c', order=(p_value,d_value,q_value), enforce_stationarity=False, enforce_invertibility=False)
model_fit = model.fit(disp=0)
model_fit.aic
model_fit.summary()

model_fit.plot_diagnostics(figsize = (15,15))
plt.show()

SARIMA_pre = model_fit.get_prediction(start=len(train), end=len(train)+len(test))
SARIMA_pre.predicted_mean

Pred_coef = SARIMA_pre.predicted_mean
Pred_coef_itv = SARIMA_pre.conf_int(alpha=0.05)

fig = plt.figure(figsize = (15,5))
ax = fig.add_subplot()
ax.plot(train[poi], color = "red", label="train")
ax.plot(test[poi], color = "orange" , label="test", alpha = 0.5)
ax.plot(SARIMA_pre.predicted_mean, color = "blue" , label="prediction")
ax.fill_between(x = Pred_coef_itv.index, y1 = Pred_coef_itv.iloc[:,0], y2 = Pred_coef_itv.iloc[:,1], color = "gray", alpha = 0.2)
ax.legend()
plt.show()


