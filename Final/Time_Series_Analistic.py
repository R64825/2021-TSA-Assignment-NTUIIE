# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 13:53:44 2022

@author: TerryYang
"""
import pandas as pd
import matplotlib.pyplot as plt

#read
input_path = r"C:\Users\TerryYang\(2021).csv"
df = pd.read_csv(input_path)

df['Date'] =pd.to_datetime(df.DataCreationDate)
df['Date'] = pd.to_datetime(df['Date'], format="%Y%m%d:%H:%M:%S.%f").sort_values()
df = df.sort_values(by='Date',ascending=True)
df = df.drop(columns=['DataCreationDate'])
df = df.drop(columns=['Unnamed: 0'])
df.head()