#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 09:07:04 2020

@author: james
"""

# Load Libraries
print("loading libraries")
import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_similarity
import random
from matplotlib.font_manager import FontProperties
from tqdm import tqdm
tqdm.pandas()
# Get a list of files from datasets/tweets
file_list = os.listdir('datasets/preprocessed')

print("Defining function for building feature matrix")
## Build Feature Extraction
def build_feature_matrix(documents, feature_type='frequency',
                         ngram_range=(1, 3), min_df=0.0, max_df=1.0):
    feature_type = feature_type.lower().strip()
    if feature_type == 'binary':
        vectorizer = CountVectorizer(binary=False, min_df=min_df,
                                     max_df=max_df, ngram_range=ngram_range)
    elif feature_type == 'frequency':
        vectorizer = CountVectorizer(binary=False, min_df=min_df,
                                     max_df=max_df, ngram_range=ngram_range)
    elif feature_type == 'tfidf':
        vectorizer = TfidfVectorizer(binary=False, min_df=min_df,
                                     max_df=max_df, ngram_range=ngram_range)
    else:
        raise Exception("Wrong feature type entered. Possible values: 'binary', 'frequency', 'tfidf'")
    feature_matrix = vectorizer.fit_transform(documents).astype(float)
    return vectorizer, feature_matrix
# Function for K-Means clustering
def k_means(feature_matrix, num_clusters=5):
        km = KMeans(n_clusters=num_clusters,
                    max_iter=10000)
        km.fit(feature_matrix)
        clusters = km.labels_
        return km, clusters
# Function to get cluster data
def get_cluster_data(clustering_obj, reviews,
                          feature_names, num_clusters,
                          topn_features=10):
        cluster_details = {}
        ordered_centroids = clustering_obj.cluster_centers_.argsort()[:, ::-1]
        for cluster_num in range(num_clusters):
            cluster_details[cluster_num] = {}
            cluster_details[cluster_num]['cluster_num'] = cluster_num
            key_features = [feature_names[index]
                            for index
                            in ordered_centroids[cluster_num, :topn_features]]
            cluster_details[cluster_num]['key_features'] = key_features
        return cluster_details
# Function to plot clusters    
def plot_clusters(num_clusters, feature_matrix,
                  cluster_data, reviews,
                  plot_size=(16, 8)):
    # Function to generate random color
    def generate_random_color():
        color = '#%06x' % random.randint(0, 0xFFFFFF)
        return color

    markers = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd']
    cosine_distance = 1 - cosine_similarity(feature_matrix)
    mds = MDS(n_components=2, dissimilarity="precomputed",
              random_state=1)
    plot_positions = mds.fit_transform(cosine_distance)
    x_pos, y_pos = plot_positions[:, 0], plot_positions[:, 1]
    cluster_color_map = {}
    cluster_name_map = {}
    for cluster_num, cluster_details in cluster_data.items():
        cluster_color_map[cluster_num] = generate_random_color()
        cluster_name_map[cluster_num] = ', '.join(cluster_details['key_features'][:5]).strip()
    cluster_plot_frame = pd.DataFrame({'x': x_pos,
                                       'y': y_pos,
                                       'label': data['Cluster'].values.tolist(),
                                       })
    grouped_plot_frame = cluster_plot_frame.groupby('label')
    # set plot figure sizes and axes
    fig, ax = plt.subplots(figsize=plot_size)
    ax.margins(0.05)
    for cluster_num, cluster_frame in grouped_plot_frame:
        marker = markers[cluster_num] if cluster_num < len(markers) \
            else np.random.choice(markers, size=1)[0]
        ax.plot(cluster_frame['x'], cluster_frame['y'],
                marker=marker, linestyle='', ms=12,
                label=cluster_name_map[cluster_num],
                color=cluster_color_map[cluster_num], mec="none")
        ax.set_aspect('auto')
        ax.tick_params(axis='x', which='both', bottom='off', top='off',
                       labelbottom='off')
        ax.tick_params(axis='y', which='both', bottom='off', top='off',
                       labelbottom='off')
    fontP = FontProperties()
    fontP.set_size('small')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.01),
              fancybox=True,
              shadow=True, ncol=5, numpoints=1, prop=fontP)
    for index in range(len(cluster_plot_frame)):
        ax.text(cluster_plot_frame.ix[index]['x'],
                cluster_plot_frame.ix[index]['y'], size=8)
    plt.show()

# Loop through all the preprocessed datasets    
for i in file_list:
    print("Reading in dataset "+ i)
    data = pd.read_excel('datasets/preprocessed/' + i)   
    data = data.drop(data.columns[0], axis=1)
    data = data.drop(data.columns[0], axis=1)
    ##Call text_prepare Function into new column in dataset
    
    
    print("Vectorizing tokenzed text")
    
    vectorizer, feature_matrix = build_feature_matrix(data['new_text'],
                                                      feature_type='tfidf',
                                                      min_df=0.05, max_df=0.95,
                                                      ngram_range=(1, 3))   
    feature_names = vectorizer.get_feature_names()
    feature_names
    
    
    
    print("Running kmeans clustering")
    num_clusters = 2
    km_obj, clusters = k_means(feature_matrix=feature_matrix,
                               num_clusters=num_clusters)    
    data['Cluster'] = clusters
    
    
    
    
    ####def cluster data --> pick km clusters or affinity propagation object
    cluster_data = get_cluster_data(clustering_obj=km_obj,
                                    reviews=data,
                                    feature_names=feature_names,
                                    num_clusters=num_clusters,
                                    topn_features=5)
    # Save the clustered data to the clustered directory
    data.to_excel('datasets/clustered/'+i[:-15]+'_clustered.xlsx')
