# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 21:45:30 2022

@author: TerryYang
"""

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pmdarima.arima import auto_arima
from math import sqrt
from sklearn.metrics import mean_squared_error

# settings
input_path = r"C:\Users\TerryYang\(2021)_Preprocessed.csv"
train_Ratio = 0.8
poi = "AQI"
'''
read dataset
'''
df = pd.read_csv(input_path)
df['Date'] =pd.to_datetime(df.Date)
df = df.drop(columns = ['Unnamed: 0'])


'''
generate date data
'''
df = df[["AQI", "Date"]]
df = df.groupby([df['Date'].dt.date]).mean()
df['Date'] = df.index

'''
visialize
'''
poi = "AQI"
X = df['Date']
y = df[poi]

fig = plt.figure()
plt.plot(X, y)
plt.title(poi +" Changes at Wanhua, 2021")
plt.xlabel('Time')
plt.ylabel(poi)
fig.set_figheight(6)
fig.set_figwidth(10)

plt.plot(X,y, color='blue')
plt.show()

'''
OLS regression
'''
X = df['Date_NO']
y = df[poi]
reg = sm.OLS(y, X)
result = reg.fit()
# print result
print("parameters: \n", result.params)
print("\n", result.summary())

'''
auto arima
'''
# split train and test
train = df[:int(train_Ratio*(len(df)))]
test = df[int(train_Ratio*(len(df))):]

Auto_ARIMA = auto_arima(train.Date_NO, trace=True, error_action='ignore', suppress_warnings=True).fit(train.AQI)

forecast = Auto_ARIMA.predict(n_periods=len(test))
forecast = pd.DataFrame(forecast,index = test.index,columns=['Prediction'])

#plot the predictions for validation set
fig = plt.figure(figsize = (10,6))
ax = fig.add_subplot()
ax.plot(train.AQI, color = "blue", label="Train")
ax.plot(test.AQI, color = "gray", label="test")
ax.plot(forecast, color = "green", label="Prediction")
ax.legend()
plt.show()

#calculate rmse
rms = sqrt(mean_squared_error(test.AQI,forecast))
print(rms)