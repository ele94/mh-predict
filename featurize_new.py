import pickle
import pandas as pd
import numpy as np

from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

import pandas, xgboost, numpy, textblob, string
from sklearn.model_selection import train_test_split


def main():


    #with open('data/pickles/test.users.pkl', 'rb') as test_users_file:
    #    test_users = pickle.load(test_users_file)
    #with open('data/pickles/train.users.pkl', 'rb') as train_users_file:
    #    train_users = pickle.load(train_users_file)


    with open('data/pickles/train.x.pkl', 'rb') as train_df_file:
        train_DF = pickle.load(train_df_file)

    with open('data/pickles/test.x.pkl', 'rb') as test_df_file:
        test_DF = pickle.load(test_df_file)


    train_featsDF = create_features(train_DF)
    test_featsDF = create_features(test_DF)

    print(train_featsDF)

    with open('data/pickles/train.df.feats.pkl', 'wb') as train_feats_file:
        pickle.dump(train_featsDF, train_feats_file)

    with open('data/pickles/test.df.feats.pkl', 'wb') as test_feats_file:
        pickle.dump(test_featsDF, test_feats_file)


def create_features(trainDF):
    newFeats = pd.DataFrame()
    newFeats['char_count'] = trainDF['clean_text'].apply(len)
    newFeats['word_count'] = trainDF['clean_text'].apply(lambda x: len(x.split()))
    newFeats['word_density'] = newFeats['char_count'] / (newFeats['word_count'] + 1)
    newFeats['punctuation_count'] = trainDF['clean_text'].apply(
        lambda x: len("".join(_ for _ in x if _ in string.punctuation)))
    newFeats['upper_case_word_count'] = trainDF['clean_text'].apply(
        lambda x: len([wrd for wrd in x.split() if wrd.isupper()]))

    # pos_family = {
    #     'noun': ['NN', 'NNS', 'NNP', 'NNPS'],
    #     'pron': ['PRP', 'PRP$', 'WP', 'WP$'],
    #     'verb': ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
    #     'adj': ['JJ', 'JJR', 'JJS'],
    #     'adv': ['RB', 'RBR', 'RBS', 'WRB']
    # }
    #
    # # function to check and get the part of speech tag count of a words in a given sentence
    # def check_pos_tag(x, flag):
    #     cnt = 0
    #     try:
    #         wiki = textblob.TextBlob(x)
    #         for tup in wiki.tags:
    #             ppo = list(tup)[1]
    #             if ppo in pos_family[flag]:
    #                 cnt += 1
    #     except:
    #         pass
    #     return cnt
    #
    # trainDF['noun_count'] = trainDF['clean_text'].apply(lambda x: check_pos_tag(x, 'noun'))
    # trainDF['verb_count'] = trainDF['clean_text'].apply(lambda x: check_pos_tag(x, 'verb'))
    # trainDF['adj_count'] = trainDF['clean_text'].apply(lambda x: check_pos_tag(x, 'adj'))
    # trainDF['adv_count'] = trainDF['clean_text'].apply(lambda x: check_pos_tag(x, 'adv'))
    # trainDF['pron_count'] = trainDF['clean_text'].apply(lambda x: check_pos_tag(x, 'pron'))

    return newFeats





if __name__ == '__main__':
    main()