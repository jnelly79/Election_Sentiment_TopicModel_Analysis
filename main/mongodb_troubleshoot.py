# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 16:43:13 2020

@author: jnels
"""




from pymongo import MongoClient, errors

import time

# create a timestamp before making API call

start = time.time()

# check the seconds that have elapsed

try:

    client = MongoClient(host = ["localhost:27017"], serverSelectionTimeoutMS = 2000)
    
    client.server_info() # will throw an exception

except:
    
    print ("connection error")
    
    # print the time difference
    
    print (time.time() - start)

# define a function that validates the client instance

def mongodb_connect(domain, port):

# create a domain and port string for the MongoDB server

    domain_str = str(domain) + ":" + str(port)

    try:
    
    # print the port and domain to connect to
    
        print ("nTrying to connect to MongoDB server:", domain, "on port:", port)
        
        # connect to a MongoDB server that doesn't exist
        
        client = MongoClient(
        
        host = [domain_str],
        
        # timeout is in milliseconds
        
        serverSelectionTimeoutMS = 2000
        
        )
        
        # server_info() should raise exception if host settings are invalid
        
        print ("server_info():", client.server_info())
    
    except errors.ServerSelectionTimeoutError as err:
    
    # catch pymongo.errors.ServerSelectionTimeoutError
    
        print ("pymongo ERROR:", err)
        
        # set the client instance to None in case of connection error
        
        client = None
        
        return client
        
        # call the mongodb_connect() function
        
client = mongodb_connect("localhost", 27017)


connection = MongoClient("mongodb://localhost")
#database = connection.tweets
print ("client:", connection)
    
    # use a try-except block when getting database/collection

try:

# create new database and collection instances

    db = client.SomeDatabase
    
    col = db["Some Collection"]

except AttributeError as error:

# should raise AttributeError if mongodb_connect() function returned "None"

    print ("Get MongoDB database and collection ERROR:", error)
    
    # or evaluate the client object instead

if client != None:

    # create new database and collection instances
    
    db = client.SomeDatabase
    
    col = db["Some Collection"]

else:

    print ("Client instance is invalid")
    
    # connect to a MongoDB server that doesn't exist
    
    client = MongoClient(
    
    host = ["DOES_NOT_EXIST"],
    
    # timeout is in milliseconds
    
    serverSelectionTimeoutMS = 2000
    
    )
    
    print ("nCreating database and collection form 'bad' client instance")
    
    # create new database and collection instances
    
    db = client.SomeDatabase
    
    col = db["Some Collection"]

# try to make a find_one() method call to the collection

try:

    one_doc= col.find_one()
    
    print ("nfind_one():", one_doc)

except errors.ServerSelectionTimeoutError as err:

    print ("nfind_one() ERROR:", err)