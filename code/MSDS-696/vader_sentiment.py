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
from nltk.sentiment.vader import SentimentIntensityAnalyzer
# Get a list of files from datasets/tweets
print("Reading in file list")
file_list = os.listdir('datasets/vader_senti')
file_list2 = file_list[20:]

print("Defining functions")
# Function to analyze the data with the vader lexicon.
def analyze_sentiment_vader_lexicon(review,
                                    threshold=0.0,
                                    verbose=False):
    # analyze the sentiment for review
    analyzer = SentimentIntensityAnalyzer()
    try:
        scores = analyzer.polarity_scores(review)
        # get aggregate scores and final sentiment
        agg_score = scores['compound']
        #final_sentiment = 'positive' if agg_score >= threshold else 'negative'
        if agg_score > threshold:
            final_sentiment = 'positive'
        elif agg_score == threshold:
            final_sentiment = 'neutral'
        else:
            final_sentiment = 'negative'
        if verbose:
            # display detailed sentiment statistics
            positive = str(round(scores['pos'], 2) * 100) + '%'
            final = round(agg_score, 2)
            negative = str(round(scores['neg'], 2) * 100) + '%'
            neutral = str(round(scores['neu'], 2) * 100) + '%'
            sentiment_frame = pd.DataFrame([[final_sentiment, final, positive,
                                             negative, neutral]],
                                           columns=pd.MultiIndex(levels=[['SENTIMENT STATS:'],
                                                                         ['Predicted Sentiment', 'Polarity Score',
                                                                          'Positive', 'Negative', 'Neutral']],
                                                            
                                                                 codes=[[0, 0, 0, 0, 0], [0, 1, 2, 3, 4]]))
    
        return final_sentiment
    except:
        return "NaN"
        pass

# Function to analyze the polarity for a positive score.
def analyze_sentiment_vader_pos(review,
                                threshold=0.1,
                                verbose=False):
    # analyze the sentiment for review
    analyzer = SentimentIntensityAnalyzer()
    try:
        scores = analyzer.polarity_scores(review)
        positive = str(round(scores['pos'], 2) * 100) + '%'
        return positive
    except:
        return "NaN"
        pass

# Function to analyze the polarity for a negative score.    
def analyze_sentiment_vader_neg(review,
                                threshold=0.1,
                                verbose=False):
    # analyze the sentiment for review
    analyzer = SentimentIntensityAnalyzer()
    try:
        scores = analyzer.polarity_scores(review)
        negative = str(round(scores['neg'], 2) * 100) + '%'
        return negative
    except:
        return "NaN"
        pass

# Function to analyze the polarity for a neutral score.
def analyze_sentiment_vader_neut(review,
                                 threshold=0.1,
                                 verbose=False):
    # analyze the sentiment for review
    analyzer = SentimentIntensityAnalyzer()
    try:
        scores = analyzer.polarity_scores(review)
        neutral = str(round(scores['neu'], 2) * 100) + '%'
        return neutral
    except:
        return "Nan"
        pass

# Function to analyze the polarity for a final score.
def analyze_sentiment_vader_fin(review,
                                threshold=0.1,
                                verbose=False):
    # analyze the sentiment for review
    analyzer = SentimentIntensityAnalyzer()
    try:
        scores = analyzer.polarity_scores(review)
        agg_score = scores['compound']
        final = round(agg_score, 2)
        return final
    except:
        return "NaN"
        pass

# Function for the vader predicted sentiment based on the percentage scores.
def vd_sentiment(row):
    sentiment = ""
    try:
        x = float(row['vd_positive_sentiment'].strip("%"))/100
    except:
        x = row['vd_positive_sentiment']
    try:
        y = float(row['vd_negative_sentiment'].strip("%"))/100
    except:
        y = row['vd_negative_sentiment']
    try:
        z = float(row['vd_neutral_sentiment'].strip("%"))/100
    except:
        z = row['vd_neutral_sentiment']
    #print(type(x),":",x," ,",type(y),":",y, " ,",type(z), ":",z )
    if x > y and x > z:
        sentiment = "positive"
        return sentiment
    elif z > x and z > y:
        sentiment = "neutral"
        return sentiment
    elif y > x and y > z:
        sentiment = "negative"
        return sentiment

print("Looping through file list")

# data = pd.read_excel('datasets/vader_senti/biden_am_vader.xlsx', index_col=[0])
# data = data.drop(data.columns[0], axis=1)
# data['vader_sentiment'] = ""
# data['vader_sentiment'] = [vd_sentiment(row) for i, row in data.iterrows()]

# data.rename(columns={'vd_sentiment':'vd__polarity_sentiment'}, inplace=True)

# Loop through all the afinn_senti datasets and run the vader sentiment analysis on them.
for i in file_list2:
    print("Reading in dataset "+ i)
    data = pd.read_excel('datasets/vader_senti/' + i)     
    data = data.drop(data.columns[0], axis=1)
    vd_reviews = np.array(data['tweet_text'])
    # for review in vd_reviews:
    #     pred = analyze_sentiment_vader_lexicon(review, threshold=0.4, verbose=True)
    data['vd_positive_sentiment'] = [analyze_sentiment_vader_pos(review) for review in vd_reviews]
    data['vd_negative_sentiment'] = [analyze_sentiment_vader_neg(review) for review in vd_reviews]
    data['vd_neutral_sentiment'] = [analyze_sentiment_vader_neut(review) for review in vd_reviews]
    data['vd_polarity_score'] = [analyze_sentiment_vader_fin(review) for review in vd_reviews]
    data['vd_polarity_sentiment'] = [analyze_sentiment_vader_lexicon(review) for review in vd_reviews]
    #data.rename(columns={'vd_sentiment':'vd__polarity_sentiment'}, inplace=True)
    data['vader_sentiment'] = ""
    data['vader_sentiment'] = [vd_sentiment(row) for i, row in data.iterrows()]
    # Save the vader sentiment analysis datasets to vader_Senti directory.
    data.to_excel('datasets/vader_senti/'+i[:-11]+'_vader.xlsx')
