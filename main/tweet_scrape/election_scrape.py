# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 08:53:17 2020

@author: jnels
"""
import tweepy
from pymongo import MongoClient
from datetime import datetime
from credentials import cred

# Application Settings
consumer_key = cred["consumer_key"]
consumer_secret = cred["consumer_secret"]
# Your Access Token
access_token = cred["access_token"]
access_token_secret = cred["access_token_secret"]

# API from OAuth
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

# Class for the streamlistener from tweepy
count = 0
class StreamListener(tweepy.StreamListener):
    
    def __init__(self, api=None):
        super(StreamListener, self).__init__()
        self.num_tweets = 0
    
    # Function for the mongoDB client
    def on_status(self, status):
        connection = MongoClient("mongodb://localhost")
        # Change the database after connection. when storing in a different DB
        database = connection.tweets

        retweet = status._json.get("retweeted_status")
        
        # If is a Retweet or a commom tweet
        if retweet:
            if retweet.get('full_text'):
                tweet_type = 'retweet_extended'
                tweet_text = retweet.get('full_text')
            else:
                tweet_type = 'retweet'
                tweet_text = retweet.get('text')
        else:
            extended_tweet = status._json.get('extended_tweet')
            if extended_tweet and 'full_text' in extended_tweet:
                tweet_type = 'tweet_extended'
                tweet_text = status.extended_tweet['full_text']
            else:
                tweet_type = 'tweet'
                tweet_text = status.text
        
        tweet = { "_id": status.id_str,
                  "tweet_@": status.user.screen_name,
                  "tweet_type": tweet_type,
                  "tweet_text": tweet_text,
                  "tweet_created_at": status.created_at,
                  "created_at": datetime.utcnow() }

        self.num_tweets += 1
        if self.num_tweets < 10000:
            # Change the collection after database. when storing into a different collection
            database.biden_aft_3d.insert(tweet)
            return True
        else:
            return False
        
        

# Stream
# Change the track= as needed for the filter
stream = tweepy.Stream(auth = api.auth,
                       listener=StreamListener(), tweet_mode='extended')
stream.filter(track=['biden'], languages=['en'])
