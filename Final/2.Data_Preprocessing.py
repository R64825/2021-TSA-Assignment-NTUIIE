# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 13:53:44 2022

@author: TerryYang
"""
import pandas as pd

output_name = "(2021)_Preprocessed"
output_path = r"C:\Users\TerryYang\Desktop\Github\2021-TSA-Assignment-NTUIIE\Final"
input_path = r"C:\Users\TerryYang\(2021).csv"

# read dataset
df = pd.read_csv(input_path)

# sort by date time
df['Date'] =pd.to_datetime(df.DataCreationDate)
df['Date'] = pd.to_datetime(df['Date'], format="%Y%m%d:%H:%M:%S.%f").sort_values()
df = df.sort_values(by='Date',ascending=True)
df.head()

# finding missing value
before = len(df)
df = df.set_index('Date')
df = df.reindex(pd.date_range(start=df.index[0], end=df.index[-1], freq='1h'))
df['Date'] = df.index
df = df.reset_index()
df = df.drop(columns = ['index'])
df = df.drop(columns = ['DataCreationDate'])
df = df.drop(columns = ['Unnamed: 0'])
print("Missing: "+ str(len(df)-before)+" data")

# filling missing data with int(mean)
df['AQI']=df['AQI'].fillna(int(df['AQI'].mean()))
df['PM2.5_AVG']=df['PM2.5_AVG'].fillna(int(df['PM2.5_AVG'].mean()))
df['PM2.5']=df['PM2.5'].fillna(int(df['PM2.5'].mean()))

# output csv
df.to_csv(output_name + ".csv")