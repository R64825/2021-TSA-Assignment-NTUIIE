# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 15:52:00 2021

@author: TerryYang
"""
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.holtwinters import ExponentialSmoothing as HWES
'''a'''
x = np.arange(0,8*np.pi,0.1)   # start,stop,step
y = 10+np.cos(x)-x/6

y_noise = []

for i in y:
    n = 0.3
    noise = np.random.uniform(-n,n)
    y_noise.append(i + noise)

plt.plot(x,y_noise)
plt.show()

'''b'''
df_original_series = pd.DataFrame({
    'period': x,
    'demand': y_noise
})

# include average interger.
df_interger_series = df_original_series[df_original_series['period'] % 1 == 0]
df_interger_series['period'] = df_interger_series['period'].div(np.pi).round(1)
plt.plot(df_interger_series['period'],df_interger_series['demand'])

# deseasonalize
series_deseasonalization = df_interger_series.loc[:, 'demand'].rolling(3).mean().dropna()
series_deseasonalization=series_deseasonalization.drop([series_deseasonalization.index[22],series_deseasonalization.index[23]])

# deseasonalize dataframe
df_deseasonalization = pd.DataFrame({
    'Quater': sum([[1,2,3]*8,[1,2]],[]),
    'Period': df_interger_series['period'] ,
    'Demand': df_interger_series['demand'] ,
    'Deseasonalized_Demand': series_deseasonalization
})

'''c'''
#build up regressing model
reg = LinearRegression().fit(np.asarray(df_deseasonalization.loc[20:230, 'Period']).reshape(-1, 1),
                            df_deseasonalization.loc[:, 'Deseasonalized_Demand'].dropna())
#predict nan value
values = pd.Series(reg.predict(np.asarray(df_deseasonalization.loc[:, 'Period']).reshape(-1,1)))

df_deseasonalization.loc[0,['Deseasonalized_Demand']] = values[0]
df_deseasonalization.loc[10,['Deseasonalized_Demand']] = values[10]
df_deseasonalization.loc[240,['Deseasonalized_Demand']] = values[24]
df_deseasonalization.loc[250,['Deseasonalized_Demand']] = values[25]

# calculate seansonality factor
df_deseasonalization.loc[:, 'Seasonality'] = (df_deseasonalization.loc[:, 'Demand'] / df_deseasonalization.loc[:, 'Deseasonalized_Demand'])

df_Seasonality_bar= pd.DataFrame({
    'Quater': sum([[1,2,3]*8,[1,2]],[]),
    'Period': df_interger_series['period'] ,
    'Demand': df_interger_series['demand'] ,
    'Deseasonalized_Demand': series_deseasonalization
})

df_seasonality = df_deseasonalization.groupby(['Quater'], as_index=False).mean()
df_seasonality.loc[:, 'Seasonality_bar'] = df_seasonality.loc[:, 'Seasonality']
df_seasonality = df_seasonality[['Quater','Seasonality_bar']]

df_deseasonalization = pd.merge(df_deseasonalization,df_seasonality).sort_values('Period')


'''d'''
df_deseasonalization.loc[:, 'Forecast'] =  (reg.predict(np.asarray(df_deseasonalization.loc[:, 'Period']).reshape(-1,1)) * df_deseasonalization.loc[:, 'Seasonality_bar'])
df_deseasonalization.loc[:, 'Error'] =  (df_deseasonalization.loc[:, 'Demand']-df_deseasonalization.loc[:, 'Forecast'])
df_deseasonalization.loc[:, 'Error_Squre'] =  (df_deseasonalization.loc[:, 'Error']*df_deseasonalization.loc[:, 'Error'])
MSE = df_deseasonalization['Error_Squre'].sum()/len(df_deseasonalization)
MAPE = ((abs(df_deseasonalization.loc[:, 'Error']) / abs(df_deseasonalization.loc[:, 'Demand'])).sum())*100/len(df_deseasonalization)

'''e'''
x = np.arange(26.0,10*np.pi,0.1)
y = np.cos(x)-x/6

y_noise = []

for i in y:
    n = 0.3
    noise = np.random.uniform(-n,n)
    y_noise.append(i + noise)
    
#true model
df_true_model = pd.DataFrame({
    'Period': x,
    'Demand': y_noise,
    'Forecast':y,
})
df_true_model.loc[:, 'Error'] =  (df_true_model.loc[:, 'Demand']-df_true_model.loc[:, 'Forecast'])
df_true_model.loc[:, 'Error_Squre'] =  (df_true_model.loc[:, 'Error']*df_true_model.loc[:, 'Error'])
MSE = df_true_model['Error_Squre'].sum()/len(df_true_model)
MAPE = ((abs(df_true_model.loc[:, 'Error']) / abs(df_true_model.loc[:, 'Demand'])).sum())*100/len(df_true_model)

#time-series model
df_time_series_model = pd.DataFrame({   
    'Period': x,
    'Demand': y_noise
})


df_interger_series = df_time_series_model[df_time_series_model['Period'].round(1) % 1.0 == 0]
df_interger_series['Period'] = df_interger_series['Period'].div(np.pi).round(1)
plt.plot(df_interger_series['Period'],df_interger_series['Demand'])

df_time_series_deseasonalization= pd.DataFrame({   })
df_time_series_deseasonalization = pd.merge(df_interger_series,df_deseasonalization, how="outer").sort_values('Period')
df_time_series_deseasonalization.loc[:,'Quater']=sum([[1,2,3]*10,[1,2]],[])

df_time_series_deseasonalization=df_time_series_deseasonalization.drop(['Seasonality_bar'], axis=1)
df_time_series_deseasonalization = pd.merge(df_time_series_deseasonalization,df_seasonality,how = "outer").sort_values('Period')

df_time_series_deseasonalization.loc[:, 'Forecast'] =  (reg.predict(np.asarray(df_time_series_deseasonalization.loc[:, 'Period']).reshape(-1,1)) * df_time_series_deseasonalization.loc[:, 'Seasonality_bar'])
df_time_series_deseasonalization.loc[:, 'Error'] =  (df_time_series_deseasonalization.loc[:, 'Demand']-df_time_series_deseasonalization.loc[:, 'Forecast'])
df_time_series_deseasonalization.loc[:, 'Error_Squre'] =  (df_time_series_deseasonalization.loc[:, 'Error']*df_time_series_deseasonalization.loc[:, 'Error'])
MSE = df_time_series_deseasonalization['Error_Squre'].sum()/len(df_time_series_deseasonalization)
MAPE = ((abs(df_time_series_deseasonalization.loc[:, 'Error']) / abs(df_time_series_deseasonalization.loc[:, 'Demand'])).sum())*100/len(df_deseasonalization)
'''f'''


'''Q2'''
x = np.arange(0,8*np.pi,0.1)   # start,stop,step
y = 10+np.cos(x)-x/6

y_noise = []

for i in y:
    n = 0.3
    noise = np.random.uniform(-n,n)
    y_noise.append(i + noise)

plt.plot(x,y_noise)
plt.show()

'''b'''
df_original_series = pd.DataFrame({
    'Period': x,
    'Demand': y_noise
})

# include average interger.
df_interger_series = df_original_series[df_original_series['Period'] % 1 == 0]
df_interger_series['Period'] = df_interger_series['Period'].div(np.pi).round(1)
plt.plot(df_interger_series['Period'],df_interger_series['Demand'])

# train the data with Holt-Winters algorithms with statsmodels module.
HWES_model = HWES(df_interger_series.loc[:, 'Demand'], seasonal_periods=8, trend='add', seasonal='mul')
HWES_fit_report = HWES_model.fit()
print(HWES_fit_report.summary())
