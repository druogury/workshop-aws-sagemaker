# coding: utf-8
import pandas as pd
import numpy as np

from pathlib import Path
from time import time
from subprocess import Popen, PIPE, STDOUT
import simplejson
import pickle

import nltk
nltk.download('wordnet')
nltk.download('stopwords')
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer

import spacy
spacy.load('en')
from spacy.lang.en import English
parser = English()

from gensim import corpora
import gensim


HOME = str(Path.home()) + '/'
PROJ = HOME + 'proj/workshop-aws-sagemaker/lda/'
DATA = PROJ + 'data/'
SRC  = PROJ + 'src/'

# Mallet’s version often gives a better quality of topics
MALLET_PATH = HOME + 'mallet-2.0.8/bin/mallet' # update this path


en_stop = nltk.corpus.stopwords.words('english')
en_stop.extend(['from', 'subject', 're', 'edu', 'use'])
en_stop.extend(pd.read_csv(DATA + "stopwordsApps.txt", header=0)["stopwords"].tolist())
en_stop.extend(pd.read_csv(DATA + "stopwordsEnglishBIG.txt", header=0, encoding='unicode_escape')["stopwords"].tolist())
en_stop = set(en_stop)


def jz_read(fn):
    with open(fn, mode="rb") as f:
        p = Popen(['pigz', '-dc'], stdin=f, stdout=PIPE)
        return simplejson.loads(p.communicate()[0])


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)

    print()


def tokenize(text):
    lda_tokens = []
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        elif token.like_url:
            lda_tokens.append('URL')
        elif token.orth_.startswith('@'):
            lda_tokens.append('SCREEN_NAME')
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens


def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma
    

def get_lemma2(word):
    return WordNetLemmatizer().lemmatize(word)


def prepare_text_for_lda(text, lemmatize=get_lemma):
    tokens = tokenize(text)
    tokens = [token for token in tokens if len(token) > 4]
    tokens = [token for token in tokens if token not in en_stop]
    tokens = [lemmatize(token) for token in tokens]
    return tokens

    

if __name__ == '__main__':
    # Read data    
    # raw_data = jz_read(DATA + 'matters.en.2018-06.json.gz')
    raw_data = jz_read(DATA + 'playstore-00.tar.gz')

    df = pd.DataFrame.from_dict(raw_data, orient='index')
    print(0, df.head())
    
    # random shuffle
    df = df.sample(frac=1, random_state=0)
    print(1, df.head())

    # remove uncategorized
    df = df[df['cat_key'] != 'UNCATEGORIZED']
    
    categories = sorted(df['cat_key'].unique())
    print('{} categories :'.format(len(categories)), categories)

    # remove games
    if False:
        print(len(df))
        df = df[~ df['cat_key'].str.startswith('GAME_')]
        print(len(df))

    dff = df.copy() # .iloc[:1000]
    print('{} documents'.format(len(dff)))
    
    dff['tokens'] = dff.apply(
        lambda row: prepare_text_for_lda('{}\n{}\n{}\n{}'.format(
            row['title'], row['category'], row['description'], row['short_desc'])), axis=1)

    dff['nb_of_tokens'] = dff['tokens'].apply(lambda lst : len(lst))
    dff = dff[dff['nb_of_tokens'] > 0]
    
    BASE_NAME = DATA + 'gensim/raw_desc_{}samples'.format(len(dff))
    # df.to_csv(BASE_NAME + ".df.bz2", compression='bz2')
    dff.to_csv(BASE_NAME + ".dff.bz2", compression='bz2')

    # save list of nb of tokens to plot distribution
    with open(BASE_NAME + "nb_of_tokens.list", "wb") as fp:
        pickle.dump(dff['nb_of_tokens'], fp)

    stop
    
    text_data = dff['tokens'].tolist()
    # print(text_data[:5])
    dictionary = corpora.Dictionary(text_data)
    corpus = [dictionary.doc2bow(text) for text in text_data]
    

    pickle.dump(corpus, open(BASE_NAME + '.corpus', 'wb'))
    dictionary.save(BASE_NAME + '.dict')
    
    # optimal number of topics is 50 without games
    for NUM_TOPICS in [len(df['cat_key'].unique()), 100]:
        print('*** Mallet === {} topics'.format(NUM_TOPICS))
        # https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/
        model = gensim.models.wrappers.LdaMallet(MALLET_PATH, corpus=corpus, num_topics=NUM_TOPICS, id2word=dictionary)
        model.save(BASE_NAME + '_{}topics.model_mallet_nogames'.format(NUM_TOPICS))

    # gensim implementation
    for NUM_TOPICS in [len(df['cat_key'].unique()), 100]:
        print('*** Gensim === {} topics'.format(NUM_TOPICS))
        # default parameters
        model = gensim.models.ldamodel.LdaModel(corpus, num_topics=NUM_TOPICS, id2word=dictionary, passes=1, random_state=123456, chunksize=10000)
        model.save(BASE_NAME + '_{}topics.model_gensim'.format(NUM_TOPICS))

