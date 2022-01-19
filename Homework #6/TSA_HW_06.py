# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 16:51:51 2021

@author: TerryYang
"""
import statsmodels.api as sm
from statsmodels import tsa as TSA
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.ar_model import AutoReg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
'''
'''
'''
Q1 
Simulate a time series 𝑦𝑡 of length 𝑛 = 100 
following an ARMA(1,1) model with 𝜙 = 0.8 and 𝜃 = 0.4.
'''
'''
'''
def MSE(arr):
    MSE = 0
    for i in range(0, len(arr)):
        MSE += arr[i] * arr[i]
    return MSE

def Rnd_standard_t( dof ):
    np.random.standard_t(dof)
    
def Generate_ARMA_Series( theta, phi, n_samples, distrvs):
    # parameters
    arparams = np.array([phi])
    maparams = np.array([theta])
    ar = np.r_[1, -arparams]
    ma = np.r_[1, maparams]
    # generate arma(1,1) model
    if (distrvs != None):
        arma = sm.tsa.arma_generate_sample(ar, ma, n_samples, distrvs = Rnd_standard_t(distrvs))   
    else:
        arma = sm.tsa.arma_generate_sample(ar, ma, n_samples)   
        
    return {
        'arma': arma,
        'ar': ar,
        'ma': ma,
        'n' : n_samples
    }

def EACF(arma, ar_max, ma_max, show):
    def lag1(arma, lag=1):
        return pd.Series(arma).shift(lag)
    
    def reupm(m, nrow, ncol):
        k = ncol - 1
        m2 = np.empty((m.shape[0], k))
        for i in range(k):
            i1 = i + 1
            work = lag1(m1[:, i])
            work[0] = -1
            temp = m1[:, i1] - work * m1[i1, i1]/m1[i, i]
            temp[i1] = 0
            m2[:, i] = temp
        return m2
    
    def ceascf(m, cov1, nar, ncol, count, ncov, arma, armam):
        result = np.zeros(nar+1)
        result[0] = cov1[ncov + count - 1]
        for i in range(1, nar+1):
            A = np.empty((len(arma) - i, i+1))
            A[:, 0] = arma[i:]
            A[:, 1:] = armam[i:, :i]
            b = np.r_[1, -m[:i, i-1]]
            temp = A @ b
            result[i] = acf(temp, nlags=count, fft=False)[count]
        return result
    
    ar_max = ar_max + 1
    ma_max = ma_max + 1
    nar = ar_max - 1
    nma = ma_max
    ncov = nar + nma + 2
    nrow = nar + nma + 1
    ncol = nrow - 1
    arma = np.array(arma) - np.mean(arma)
    armam = np.empty((len(arma), nar))
    for i in range(nar):
        armam[:, i] = lag1(arma, lag=i+1)
    cov1 = acf(arma, nlags=ncov, fft=False)
    cov1 = np.r_[np.flip(cov1[1:]), cov1]
    ncov = ncov + 1
    m1 = np.zeros((nrow, ncol))
    for i in range(ncol):
        m1[:i+1, i] = AutoReg(arma, lags=i+1, trend='c').fit().params[1:]
        
    eacf = np.empty((ar_max, nma))
    for i in range(nma):
        m2 = reupm(m = m1, nrow = nrow, ncol = ncol)
        ncol = ncol - 1
        eacf[:, i] = ceascf(m2, cov1, nar, ncol, i+1, ncov, arma, armam)
        m1 = m2
    
    work = np.arange(1, nar+2)
    work = len(arma) - work + 1
    symbol = np.empty(eacf.shape, dtype=object)
    for i in range(nma):
        work = work - 1
        symbol[:, i] = np.where(np.abs(eacf[:, i]) > 2/np.sqrt(work), 'x', 'o')
    
    symbol = pd.DataFrame(symbol)
    if show:
        print('AR / MA')
        print(symbol)
    
    return {
        'eacf': eacf,
        'ar.max': ar_max,
        'ma.max': ma_max,
        'symbol': symbol
    }

generate_result = Generate_ARMA_Series( theta = 0.4, phi = 0.8, n_samples = 100, distrvs= None)
arma = generate_result["arma"]
plt.plot(arma)
plt.title('ARMA(1, 1), \u03A6 = 0.8 & \u0398 = 0.4')
plt.xlabel('t')
plt.ylabel('y')
plt.show() 

'''
(a) Calculate and plot the theoretical autocorrelation function for this model.
Plot sufficient lags until the correlations are negligible.
'''
arma_theoretical_acf = TSA.arima_process.arma_acf(generate_result["ar"], generate_result["ma"] , lags=20)
plt.plot(range(1,len(arma_theoretical_acf)+1), arma_theoretical_acf, 'bD', markersize=5, linestyle='dashed')
plt.xticks(range(1,len(arma_theoretical_acf)+1))
plt.show()
'''
(b) Calculate and plot the sample ACF for your simulated series. 
How well do the values and patterns match the theoretical ACF from part (a)?
'''
plt.show(sm.graphics.tsa.plot_acf(arma, lags=generate_result["n"]-1))

'''
(c) Calculate and interpret the sample EACF for this series. 
Does the EACF help you specify the correct orders for the model?
'''
EACF_result = EACF(arma, 5, 5, True)
print('\n\n EACF Matrix: \n', np.around(EACF_result['eacf'], decimals=3))

'''
(d) Repeat parts (b) and (c) 
with a new simulation using the same parameter values but sample size 𝑛 = 48.
'''
generate_result = Generate_ARMA_Series( 0.8, 0.4, 48, None)
arma = generate_result["arma"]

#b
arma_sample_acf = TSA.stattools.acf(arma, nlags=generate_result["n"])
plt.show(sm.graphics.tsa.plot_acf(arma, lags=generate_result["n"]-1))

#c
EACF_result = EACF(arma, 5, 5, True)
print('\n\n EACF Matrix: \n', np.around(EACF_result['eacf'], decimals=3))

'''
(e) Repeat parts (b) and (c) with a new simulation 
using the same parameter values but sample size 𝑛 = 200.
'''
generate_result = Generate_ARMA_Series( 0.8, 0.4, 200, None)
arma = generate_result["arma"]

#b
arma_sample_acf = TSA.stattools.acf(arma, nlags=generate_result["n"])
plt.show(sm.graphics.tsa.plot_acf(arma, lags=generate_result["n"]-1))

#c
EACF_result = EACF(arma, 5, 5, True)
print('\n\n EACF Matrix: \n', np.around(EACF_result['eacf'], decimals=3))


'''
'''
'''
Q2
Simulate an ARMA(1,1) series with 𝜙 = 0.7, 𝜃 = −0.6, 𝑛 = 48 
but with error terms from a tdistribution with degrees of freedom 6.
'''
'''
'''
generate_result = Generate_ARMA_Series( theta = -0.6, phi = 0.7, n_samples = 48, distrvs = 6.0)
generate_result = Generate_ARMA_Series( -0.6, 0.7, 48, 6)
arma = generate_result["arma"]

'''
(a) show the sample EACF of the series. Is an ARMA(1,1) model suggested?
'''
EACF_result =  EACF(arma, 5, 5, True)
print('\n\n EACF Matrix: \n', np.around(EACF_result['eacf'], decimals=3))

'''
(b) Estimate 𝜙 and 𝜃 from the series and comment on the results.
'''
model_ARMA11_fit = sm.tsa.ARMA(arma , (1, 1)).fit(trend='nc', disp=0)
print(model_ARMA11_fit.params)

model_ARMA51_fit = sm.tsa.ARMA(arma , (5, 1)).fit(trend='nc', disp=0)
print(model_ARMA51_fit.params)

'''
'''
'''
Q3
The data file named robot contains a time series obtained from an industrial robot. 
The robot was put through a sequence of maneuvers, 
and the distance from a desired ending point was recorded in inches.
'''
'''
'''
df_robot = pd.read_csv("C:\\Users\\TerryYang\\Desktop\\Github\\2021-TSA-Assignment-NTUIIE\\Homework #6\\TSA HW06.robot.csv")

'''
(a) Display the time series plot of the data. 
Based on this information, do these data appear to come from a 
stationary or nonstationary process?
'''
plt.plot(df_robot)

'''
(b) Calculate and plot the sample ACF and PACF for these data. 
Based on this additional information, do these 
data appear to come from a stationary or nonstationary process?
'''
# acf
data_acf = TSA.stattools.acf(df_robot, nlags=40, fft=False)
print("ACF of the data:\n", data_acf, "\n")
plt.plot(data_acf, "bD", linestyle='dashed')
plt.show(sm.graphics.tsa.plot_acf(df_robot, lags=100))

# pacf
data_pacf = TSA.stattools.pacf(df_robot, nlags=40)
print("PACF of the data:\n", data_pacf, "\n")
plt.plot(data_pacf, "bD", linestyle='dashed')
plt.show(sm.graphics.tsa.plot_pacf(df_robot, lags=100))

'''
(c) Calculate and interpret the sample EACF.
'''
result_EACF =  EACF(df_robot.iloc[:, 0], 5, 5, True)
print('\n\n ECAF Matrix: \n', np.around(result_EACF['eacf'], decimals=4))

'''
(d) Estimate the parameters of an AR(1) model and IMA(1, 1) for these data, 
respectively.
'''
# AR(1)
model_AR1 = AutoReg(df_robot, lags=1)
model_AR1_fit = model_AR1.fit()
print("AR(1) Model:\n", model_AR1_fit.params)
print(model_AR1_fit.summary())

#IMA(1, 1)
model_IMA = ARIMA(df_robot, order=(0,1,1))
model_IMA_fit = model_IMA.fit()
print("\n\nIMA(1,1) Model:\n", model_IMA_fit.params)
print(model_IMA_fit.summary())

'''
(e) Compare the results from parts (d) in terms of AIC and discuss the residual tests.
'''
print("AIC of AR(1) model", model_AR1_fit.aic)

print("\n\nAIC of IMA(1, 1) model", model_IMA_fit.aic, "\n\n\n")

# residual test
predict_AR = model_AR1.predict(start=0, end=(len(df_robot)), params = model_AR1_fit.params).tolist()
predict_IMA = model_IMA_fit.predict( params = model_IMA_fit.params).tolist()

for i in range(len(df_robot)):
    df_robot.loc[i, 'residual_AR'] = predict_AR[i] - df_robot['robot'][i]
    df_robot.loc[i, 'residual_IMA'] = predict_IMA[i] - df_robot['robot'][i]
    
plt.plot(df_robot.loc[:, 'residual_AR'])
plt.plot(df_robot.loc[:, 'residual_IMA'])
plt.show()

MSE(df_robot['residual_AR'])
MSE(df_robot['residual_IMA'])
