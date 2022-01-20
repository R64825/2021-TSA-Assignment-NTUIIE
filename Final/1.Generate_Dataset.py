# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 13:08:04 2022

@author: TerryYang
"""
import pandas as pd
import os


# merge 12 datasets to df
output_name = "(2021)"
output_path = r"C:\Users\TerryYang\Desktop\Github\2021-TSA-Assignment-NTUIIE\Final"
input_path = r"C:\Users\TerryYang\Desktop\Github\2021-TSA-Assignment-NTUIIE\Final\Original_Datasets"
df = pd.DataFrame()  
directory = os.path.join(input_path)
for root,dirs,files in os.walk(directory):
    for i in range(0,len(files)):      
        df_new = pd.read_csv(input_path + '\\' + files[i])
        df = df.append(df_new)
        print("append file:" + files[i])

df.to_csv(output_name + ".csv")
