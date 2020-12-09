#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 18:45:48 2020

@author: james
"""


# Load Libraries
print("loading libraries")
import pandas as pd
import numpy as np
import os
from afinn import Afinn


# Get a list of files from datasets/tweets
file_list = os.listdir('datasets/clustered')

# Loop through all the clustered datasets.
for i in file_list:
    print("Reading in dataset "+ i)
    data = pd.read_excel('datasets/clustered/' + i) 
    reviews = np.array(data['new_text'])
        
    afn = Afinn(emoticons=True)
    print("Running AFINN on "+i)    
    af_sentiment_polarity = [afn.score(review) for review in reviews]
    af_pred_list = []
    for score in af_sentiment_polarity:
        if score >= 1.0:
            af_sent = 'positive'
            af_pred_list.append(af_sent)
        elif score == 0.0:
            af_sent = 'neutral'
            af_pred_list.append(af_sent)
        else:
            af_sent = 'negative'
            af_pred_list.append(af_sent)
    af_predicted_sentiments = af_pred_list
    data['af_sentiment_polarity'] = af_sentiment_polarity
    data['af_predicted_sentiments'] = af_predicted_sentiments
    # Save the AFINN sentiment scored dataframe to an excel spreadsheet in afinn_senti directory
    data.to_excel('datasets/Afinn_senti/'+i[:-15]+'_afinn.xlsx')