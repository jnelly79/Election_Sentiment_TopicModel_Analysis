#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 14:26:52 2020

@author: james
"""

# Load Libraries
print("loading libraries")
import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import re
import nltk
import spacy
from nltk.tokenize.toktok import ToktokTokenizer
from bs4 import BeautifulSoup
import collections
from tqdm import tqdm
tqdm.pandas()
# Get a list of files from datasets/tweets
file_list = os.listdir('datasets/tweets')

# Test_list 
#test_list = ['election_even.xlsx']

# Create nlp objects spacy en_core_web_sm and stopwords
print("Setting up nlp objects")
nlp = spacy.load('en_core_web_sm', parse=False, tag=False, entity=False)
tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words('english')
stopword_list.remove('no')
stopword_list.remove('not')
stopword_list.extend(['Gets', 'Let', 'Make', 'Just', 'Things', 'Use', '90'])

##def CONTRACTIONS_MAP
print("Creating contractions map object and setting up all dictionaries for text_preprocessing")
CONTRACTION_MAP = {
    "ain't": "are not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "I'd": "I would",
    "I'd've": "I would have",
    "I'll": "I will",
    "I'll've": "I will have",
    "I'm": "I am",
    "I've": "I have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so is",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you shall have",
    "you're": "you are",
    "you've": "you have",
    "doin'": "doing",
    "goin'": "going",
    "nothin'": "nothing",
    "somethin'": "something",
}
contractions_re_keys = [x.replace("'", "['’]") for x in CONTRACTION_MAP]
CONTRACTION_MAP.update({k.replace("'", "’"): v for k, v in CONTRACTION_MAP.items()})
leftovers_dict = {
    "'all": '',
    "'am": '',
    "'cause": 'because',
    "'d": " would",
    "'ll": " will",
    "'re": " are",
    "'em": " them",
}
leftovers_dict.update({k.replace("'", "’"): v for k, v in leftovers_dict.items()})
safety_keys = set(["he's", "he'll", "we'll", "we'd", "it's", "i'd", "we'd", "we're"])
unsafe_dict = {
    k.replace("'", ""): v for k, v in CONTRACTION_MAP.items() if k.lower() not in safety_keys
}
slang = {
    "ima": "I am going to",
    "gonna": "going to",
    "gotta": "got to",
    "wanna": "want to",
    "woulda": "would have",
    "gimme": "give me",
    "asap": "as soon as possible",
    "u": "you",
    "r ": "are ",
}
unsafe_dict.update(slang)
leftovers_re = re.compile('|'.join(sorted(leftovers_dict.keys())), re.IGNORECASE)
contractions_re = re.compile('|'.join(sorted(contractions_re_keys)), re.IGNORECASE)
unsafe_re = re.compile(r"\b" + r"\b|\b".join(sorted(unsafe_dict)) + r"\b", re.IGNORECASE)

#  Function for replacing
def _replacer(dc):
    def replace(match):
        v = match.group()
        if v in dc:
            return dc[v]
        v = v.lower()
        if v in dc:
            return dc[v]
        return v

    return replace

slang_re = re.compile(
    r"\b" + r"\b|\b".join(sorted(list(slang) + list(unsafe_dict))) + r"\b", re.IGNORECASE
)
LIM_RE = re.compile("['’]")
rc = _replacer(CONTRACTION_MAP)
rl = _replacer(leftovers_dict)
ru = _replacer(unsafe_dict)

print("Defining all functions for text_preprocessing")
def fix(s, leftovers=True, slang=True):
    # when not expecting a lot of matches, this will be 30x faster
    # otherwise not noticeably slower even in benchmarks
    if not LIM_RE.search(s):
        if slang and slang_re.search(s):
            pass
        else:
            # ensure str like expected from re.sub
            return str(s)
    s = contractions_re.sub(rc, s)
    if leftovers:
        s = leftovers_re.sub(rl, s)
    if slang:
        s = unsafe_re.sub(ru, s)
    return s

## Cleaning Text - strip HTML
def strip_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text()
    return stripped_text

# # Removing accented characters
def remove_accented_chars(sentence, keep_apostrophes=False):
    sentence = sentence.strip()
    if keep_apostrophes:
        PATTERN = r'[?|$|&|*|%|@|(|)~|`|^|*|#|!|<|>|-|_|/|[|]|{\}|]'
        filtered_sentence = re.sub(PATTERN, r' ', sentence)
    else:
        PATTERN = r'[^a-zA-Z0-9]'
        filtered_sentence = re.sub(PATTERN, r' ', sentence)
    return filtered_sentence

# # Expanding Contractions
def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
                                      flags=re.IGNORECASE | re.DOTALL)

    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match) \
            if contraction_mapping.get(match) \
            else contraction_mapping.get(match.lower())
        expanded_contraction = str(expanded_contraction)
        expanded_contraction = first_char + expanded_contraction[1:]
        return expanded_contraction

    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text

## Lemmatizing text
def lemmatize_text(text):
    text = nlp(text)
    text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
    return text

## Removing Stopwords
def remove_stopwords(text, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text

## Correct Spelling
def tokens(text):
    return re.findall('[a-z]+', text.lower())

WORDS = tokens(open('big.txt').read())
WORD_COUNTS = collections.Counter(WORDS)

def edits0(word):
    return {word}

def edits1(word):
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    def splits(word):
        return [(word[:i], word[i:])
                for i in range(len(word) + 1)]
    pairs = splits(word)
    deletes = [a + b[1:] for (a, b) in pairs if b]
    transposes = [a + b[1] + b[0] + b[2:] for (a, b) in pairs if len(b) > 1]
    replaces = [a + c + b[1:] for (a, b) in pairs for c in alphabet
                if b]
    inserts = [a + c + b for (a, b) in pairs for c in alphabet]
    return set(deletes + transposes + replaces + inserts)

def edits2(word):
    return {e2 for e1 in edits1(word) for e2 in edits1(e1)}

## Normalize text corpus - tying it all together
def text_prepare(text, html_stripping=True, contraction_expansion=True,
                 accented_char_removal=True, text_lower_case=True,
                 text_lemmatization=True, special_char_removal=True,
                 stopword_removal=True, tokenize_text=True):
    normalized_corpus = []
    # count = 0
    # count1 = 0
    for doc in text:
        doc = str(doc)
        if html_stripping:
            doc = strip_html_tags(doc)
        if contraction_expansion:
            doc = expand_contractions(doc)
        if accented_char_removal:
            doc = remove_accented_chars(doc)
        if text_lower_case:
            doc = doc.lower()
        # remove extra newlines
        doc = re.sub(r'[\r|\n|\r\n]+', ' ', doc)
        # insert spaces between special characters to isolate them
        special_char_pattern = re.compile(r'([{.(-)!}])')
        doc = special_char_pattern.sub(" \\1 ", doc)
        if text_lemmatization:
            doc = lemmatize_text(doc)
        # remove extra whitespace
        doc = re.sub(' +', ' ', doc)
        doc = doc.strip()
        if stopword_removal:
            doc = remove_stopwords(doc, is_lower_case=text_lower_case)
        if tokenize_text:
            doc = tokens(doc)
        normalized_corpus.append(doc)
        # count += 1
        # print("count1", count)
    return normalized_corpus

print("Defining function for building feature matrix")
##Feature Extraction
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

# Loop through all the datasets
for i in file_list:
    print("Reading in dataset "+ i)
    data = pd.read_excel('datasets/tweets/' + i)   
    
    ##Call text_prepare Function into new column in dataset
    print("Preparing text")
    data['new_text'] = text_prepare(data.tweet_text)
    data['new_text'] = data['new_text'].astype(str)
    # Save the prepocessed data to the preprocessed directory.
    data.to_excel('datasets/preprocessed/'+i[:-5]+'_processed.xlsx')
    
    