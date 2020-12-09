#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 13:17:27 2020

@author: james
"""
# Import libraries
import pandas as pd
import os
substrings = ['election','biden','trump']
file_list = os.listdir('datasets/vader_senti')
# Loop through the vader_senti directory and get files to read in for combining.
for folder in file_list:
    second_file_list = os.listdir('datasets/vader_senti/'+folder)
    for strings in substrings:
        files = [file for file in second_file_list if strings in file]
        df = pd.DataFrame()
        for file in files:
            data = pd.read_excel('datasets/vader_senti/'+folder+'/'+file, index_col=[0])
            data = data.drop(data.columns[0], axis=1)
            data = data.reset_index(drop=True)
            df = pd.concat([df, data], axis=0)
            df = df.reset_index(drop=True)
        print("Saving combined "+ folder+" for "+ strings)
        # Save combined datasets into directory combined.
        df.to_excel('datasets/combined/'+folder+'/'+strings+'_'+folder+'_combined.xlsx')
