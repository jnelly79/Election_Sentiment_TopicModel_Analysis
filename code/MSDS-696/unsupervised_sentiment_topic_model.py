#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 10:53:30 2020

@author: james
"""
# Load Libraries

print("loading libraries")
import pandas as pd
import numpy as np
import text_normalizer as tn
import warnings
import nltk
import gensim
from tqdm import tqdm
import os

# Declare working directory
directory = os.listdir('datasets/combined')
# elect_results_dir = os.listdir('datasets/combined/elect_results')
# day_three_dir = os.listdir('datasets/combined/day_three')
senti_words = ['positive', 'negative', 'neutral']

# Declare stopwords
stop_words = nltk.corpus.stopwords.words('english')
stop_words.remove('but')
stop_words.append('httpst')
#stop_words.append('purge')

warnings.filterwarnings("ignore")

# Loop through the datasets from the combined directory and normalize for topic modeling.
for folder in directory:
    second_folder = os.listdir('datasets/combined/'+folder)
    for file in second_folder:
        data = pd.read_excel('datasets/combined/'+folder+'/'+file)

#     for file in elect_results_dir:    
#         data = pd.read_excel('datasets/combined/elect_results/'+file, index_col=[0])
# #data = data.drop(data.columns[0], axis=1)
# data = pd.read_excel('datasets/combined/day_one/election_day_one_combined.xlsx', index_col=[0])
    # for file in day_three_dir:    
    #     data = pd.read_excel('datasets/combined/day_three/'+file, index_col=[0])
        # Separate tweets from retweets and then normalize/
        mask = data['tweet_type'] == 'tweet'
        reg_tweets = data[mask]
        #reg_tweets = data[~data['tweet_text'].str.contains('RT')]
        # reg_tweets['tweet_text'] = [tweet['tweet_text'] for i, tweet in tweets_senti.iterrows() if tweet['tweet_type'] == 'tweet']
        # reg_tweets['vader_sentiment'] = [tweet['vd__polarity_sentiment'] for i, tweet in tweets_senti.iterrows() if tweet['tweet_type'] == 'tweet']
        reg_tweets['norm_tweets'] = tn.normalize_corpus(reg_tweets['tweet_text'], stopwords=stop_words)
        
        tweets = [[],[],[]]
        
        # Create lists for subtopics for positive, negative, and neutral vader sentiments.
        tweets[0] = [sentiment['norm_tweets'] for review, sentiment in reg_tweets.iterrows() if sentiment['vd__polarity_sentiment'] == 'positive']
        tweets[1] = [sentiment['norm_tweets'] for review, sentiment in reg_tweets.iterrows() if sentiment['vd__polarity_sentiment'] == 'negative']
        tweets[2] = [sentiment['norm_tweets'] for review, sentiment in reg_tweets.iterrows() if sentiment['vd__polarity_sentiment'] == 'neutral']
        
        tweets_senti = [[],[],[]]
        tweets_senti[0] = [sentiment['norm_tweets'] for review, sentiment in reg_tweets.iterrows() if sentiment['vd__polarity_sentiment'] == 'positive']
        tweets_senti[1] = [sentiment['norm_tweets'] for review, sentiment in reg_tweets.iterrows() if sentiment['vd__polarity_sentiment'] == 'negative']
        tweets_senti[2] = [sentiment['norm_tweets'] for review, sentiment in reg_tweets.iterrows() if sentiment['vd__polarity_sentiment'] == 'neutral']
        new_df = pd.DataFrame()
        
        # Loop through the data and run through the topic model.
        for i,s,n in zip(tweets,tweets_senti,senti_words):
            temp_mask = reg_tweets['vd__polarity_sentiment'] == str(n)
            temp_df = reg_tweets[temp_mask]
            temp_df['Dominant Topic'] = ""
            temp_df['Topic Desc'] = ""
            
            bigram_ = gensim.models.Phrases(i, min_count=1, threshold=20, delimiter=b', ') # higher threshold fewer phrases.
            bigram_model_ = gensim.models.phrases.Phraser(bigram_)
            norm_corpus_bigrams_= [bigram_model_[doc] for doc in i]
            dictionary_ = gensim.corpora.Dictionary(norm_corpus_bigrams_)
            dictionary_.filter_extremes(no_below=20, no_above=0.5)
            bow_corpus_ = [dictionary_.doc2bow(text) for text in norm_corpus_bigrams_]
                
            TOTAL_TOPICS = 10
            
            # Directory to mallet
            MALLET_PATH = '/home/james/mallet-2.0.8/bin/mallet'
            lda_mallet = gensim.models.wrappers.LdaMallet(mallet_path=MALLET_PATH, corpus=bow_corpus_, 
                                                          num_topics=TOTAL_TOPICS, id2word=dictionary_,
                                                          iterations=500, workers=16)
            
            topics = [[(term, round(wt, 3)) 
                           for term, wt in lda_mallet.show_topic(n, topn=20)] 
                               for n in range(0, TOTAL_TOPICS)]
            
            
            cv_coherence_model_lda_mallet = gensim.models.CoherenceModel(model=lda_mallet, corpus=bow_corpus_, 
                                                                         texts=norm_corpus_bigrams_,
                                                                         dictionary=dictionary_, 
                                                                         coherence='c_v')
            avg_coherence_cv = cv_coherence_model_lda_mallet.get_coherence()
            
            umass_coherence_model_lda_mallet = gensim.models.CoherenceModel(model=lda_mallet, corpus=bow_corpus_, 
                                                                            texts=norm_corpus_bigrams_,
                                                                            dictionary=dictionary_,  
                                                                            coherence='u_mass')
            avg_coherence_umass = umass_coherence_model_lda_mallet.get_coherence()
            
            
            # Function to get coherence scores for topics.
            def topic_model_coherence_generator(corpus, texts, dictionary, 
                                                start_topic_count=2, end_topic_count=10, step=1,
                                                cpus=1):
                
                models = []
                coherence_scores = []
                for topic_nums in tqdm(range(start_topic_count, end_topic_count+1, step)):
                    mallet_lda_model = gensim.models.wrappers.LdaMallet(mallet_path=MALLET_PATH, corpus=corpus,
                                                                        num_topics=topic_nums, id2word=dictionary_,
                                                                        iterations=500, workers=cpus)
                    cv_coherence_model_mallet_lda = gensim.models.CoherenceModel(model=mallet_lda_model, corpus=corpus, 
                                                                                 texts=texts, dictionary=dictionary_, 
                                                                                 coherence='c_v')
                    coherence_score = cv_coherence_model_mallet_lda.get_coherence()
                    coherence_scores.append(coherence_score)
                    models.append(mallet_lda_model)
                
                return models, coherence_scores
            
            lda_models, coherence_scores = topic_model_coherence_generator(corpus=bow_corpus_, texts=norm_corpus_bigrams_,
                                                                           dictionary=dictionary_, start_topic_count=2,
                                                                           end_topic_count=30, step=1, cpus=16)
            
            coherence_df = pd.DataFrame({'Number of Topics': range(2, 31, 1),
                                         'Coherence Score': np.round(coherence_scores, 4)})
            coherence_df.sort_values(by=['Coherence Score'], ascending=False).head(10)
            
            # Choose the number of topics you want.
            best_model_idx = coherence_df[coherence_df['Number of Topics'] == 6].index[0]
            best_lda_model = lda_models[best_model_idx]
            
            topics = [[(term, round(wt, 3)) 
                           for term, wt in best_lda_model.show_topic(n, topn=10)] 
                               for n in range(0, best_lda_model.num_topics)]
            
            # for idx, topic in enumerate(topics):
            #     #print('Topic #'+str(idx+1)+':')
            #     print([term for term, wt in topic])
            #     print()
                
            topics_df = pd.DataFrame([[term for term, wt in topic] 
                                          for topic in topics], 
                                     columns = ['Term'+str(i) for i in range(1,11)],
                                     index=['Topic '+str(t) for t in range(1, best_lda_model.num_topics+1)]).T
            topics_df    
            
            pd.set_option('display.max_colwidth', -1)
            topics_df = pd.DataFrame([', '.join([term for term, wt in topic])  
                                          for topic in topics],
                                     columns = ['Terms per Topic'],
                                     index=['Topic'+str(t) for t in range(1, best_lda_model.num_topics+1)]
                                     )
            topics_df
            
            tm_results = best_lda_model[bow_corpus_]
            
            corpus_topics = [sorted(topics, key=lambda record: -record[1])[0] 
                                 for topics in tm_results]
            
            # corpus_topic_df = pd.DataFrame()
            
            # Add topic models to a dataframe    
            temp_df['Dominant Topic'] = [item[0]+1 for item in corpus_topics]
            temp_df['Contribution %'] = [round(item[1]*100, 2) for item in corpus_topics]
            temp_df['Topic Desc'] = [topics_df.iloc[t[0]]['Terms per Topic'] for t in corpus_topics]
            new_df = pd.concat([new_df, temp_df], axis=0)
            new_df = new_df.reset_index(drop=True)
        # Save dataframe to an excel spreadsheet and export to the analysis directory.
        new_df.to_excel('datasets/analysis/'+folder+'/'+file[:-14]+'_analysis.xlsx')
        #new_df.to_excel('datasets/analysis/day_three/'+file[:-14]+'_analysis.xlsx')
        

