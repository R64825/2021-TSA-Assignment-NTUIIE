# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 23:26:17 2021

@author: TerryYang
"""
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import itertools

# function definition
def Return_SARIMA_Model(y, p, d, q ,P, D, Q, S):
    order = [p,d,q]
    s_order = [P,D,Q,S]
    model=sm.tsa.statespace.SARIMAX(endog=y,order=order,seasonal_order=s_order)
    results=model.fit()
    return results

def Add_Month_Datecolumn(df, No):
    month_NO = []
    for i in range(1, len(df)+1):
        if  (i - No)%12 == 0:
            month_NO.append(1)
        else:
            month_NO.append(0)
    df.loc[:, str(No)] = month_NO
    
# gernerating dataset
df = pd.read_csv(r"C:\Users\TerryYang\Desktop\Github\2021-TSA-Assignment-NTUIIE\Homework #7\TSA HW07.co2.csv")
for i in range(1,12):
    Add_Month_Datecolumn(df, i)
plt.plot(df["time_trend"].tolist(), df["co2_level"].tolist())

'''
(a)
Fit a deterministic regression model in terms of months and time. Are the regression coefficients
significant? What is the adjusted R-squared? (Note that the month variable should be treated as
categorical and transformed into 11 dummy variables.) 
'''
X = sm.add_constant(df.drop(['month', "co2_level"], axis=1))
y = df["co2_level"]
reg = sm.OLS(y, X)

result = reg.fit()
print(result.params)
print(result.summary())

'''
(b)
Identify, estimate the SARIMA model for the co2 level. 
'''

p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
s_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

aic_list = []
pdq_index = []
s_pdq_index = []
for i in range(0,len(pdq)):
    for j in range(0,len(s_pdq)):
        order = pdq[i]
        s_order = s_pdq[j]
        result = Return_SARIMA_Model(y, *order, *s_order)
        aic_list.append(result.aic)
        pdq_index.append(i)
        s_pdq_index.append(j)
        
min_aic = min(aic_list)
min_index = aic_list.index(min_aic)
op_order = pdq[pdq_index[min_index]]
op_s_order = s_pdq[s_pdq_index[min_index]]
  
orders = []               
s_orders = []  
indeies = []
for i in range(0, len(aic_list)):
    orders.append(pdq[pdq_index[i]])
    s_orders.append(s_pdq[s_pdq_index[i]])
    indeies.append(i)
data = {'index':  indeies,
        'p, d, q': orders,
        'P, D, Q, S': s_orders,
        'AIC': aic_list
        }

df_aic = pd.DataFrame(data)

result = Return_SARIMA_Model(y, *op_order, *op_s_order)
result.aic
result.summary()

'''
(c)
Compare the two models above, what do you observe?
'''