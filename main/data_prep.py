# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 13:53:56 2020

@author: jnels
"""

# standard library
# pydata
import pandas as pd
%matplotlib inline
from pymongo import MongoClient




client = MongoClient()
#point the client at mongo URI
client = MongoClient('mongodb://localhost')
#select database
db = client['tweets']
# get list of collections
col_list = db.list_collection_names()

# loop through collections and download them all as excel spreadsheets
for i in col_list:
    tweets = db.i
    df = pd.DataFrame(list(tweets.find()))
    # Save the extracted data to the tweets directory.
    df.to_excel('datasets/tweets/'+ str(i)+'.xlsx')
   
    
### Testing code
#select the collection within the database  
# tweets = db.election_am
# #convert entire collection to Pandas dataframe
# df = pd.DataFrame(list(tweets.find()))

# df.to_excel('datasets/election_am.xlsx')

