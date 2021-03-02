# calculates features related to text characteristics

# after: windowfy.py
# next: combine_features.py OR train.py

import pickle
import pandas as pd
import numpy as np
from utils import load_pickle
from utils import save_pickle
from utils import remove_pickle

from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

import pandas, xgboost, numpy, textblob, string, re
from sklearn.model_selection import train_test_split
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nssi_corpus_path: str = "data/nssicorpus.txt"
train_x_filename = "train.x.pkl"
test_x_filename = "test.x.pkl"
train_df_feats_filename = "train.df.feats.pkl"
test_df_feats_filename = "test.df.feats.pkl"
train_feats_filename = "train.feats.pkl"
test_feats_filename = "test.feats.pkl"
normalize_param = True


def main():

    remove_pickle(train_df_feats_filename)
    remove_pickle(test_df_feats_filename)

    train_x = load_pickle(train_x_filename)
    test_x = load_pickle(test_x_filename)

    print(train_x.isnull().values.any())
    train_feats = create_features(train_x, normalize_param)
    print(train_feats.isnull().values.any())

    test_feats = create_features(test_x, normalize_param)

    save_pickle(train_df_feats_filename, train_feats)
    save_pickle(test_df_feats_filename, test_feats)

    #train_feats.to_csv(r'train_feats.csv')
    #test_feats.to_csv(r'test_feats.csv')


def create_features(users_df, normalize=True):

    normalize_exceptions = ['char_count', 'word_density']
    exclude_features = ['char_count', 'word_count']

    with open(nssi_corpus_path, 'r') as file:
        nssi_corpus = file.read().replace('*', '')
    nssi_corpus = nssi_corpus.split('\n')
    nssi_corpus.remove('')

    new_feats = pd.DataFrame()

    text_length = users_df['clean_text'].map(len)
    print(text_length)

    new_feats['char_count'] = users_df['clean_text'].map(len)
    new_feats['word_count'] = users_df['clean_text'].map(lambda x: len(x.split()))
    new_feats['word_density'] = text_length / (text_length + 1)

    new_feats['punctuation_count'] = users_df['clean_text'].map(
        lambda x: len("".join(_ for _ in x if _ in string.punctuation)))
    new_feats['upper_case_count'] = users_df['clean_text'].map(
        lambda x: len([wrd for wrd in x.split() if wrd.isupper()]))

    #my old features:
    #text features
    new_feats['questions_count'] = users_df['text'].map(lambda x: len(re.findall(r'\?', x)))
    new_feats['exclamations_count'] = users_df['text'].map(lambda x: len(re.findall(r'\!', x)))
    new_feats['smilies'] = users_df['text'].map(lambda x: len(re.findall(r'\:\)+|\(+\:', x)))
    new_feats['sad_faces'] = users_df['text'].map(lambda x: len(re.findall(r'\:\(+|\)+\:', x)))

    reg = r'\bI\b|\bme\b|\bmine\b|\bmy\b|\bmyself\b'
    new_feats['first_prons'] = users_df['clean_text'].map(lambda x: len(re.findall(reg, x)))
    print(new_feats.isnull().values.any())

    sid = SentimentIntensityAnalyzer()
    new_feats['sentiment'] = users_df['clean_text'].map(lambda x: round(sid.polarity_scores(x)['compound'], 2))
    print(new_feats.isnull().values.any())
    new_feats['nssi_words'] = users_df['tokens'].map(lambda x: sum((' '.join(x)).count(word) for word in nssi_corpus))
    print(new_feats.isnull().values.any())
    pos_family = {
        'noun': ['NN', 'NNS', 'NNP', 'NNPS'],
        'pron': ['PRP', 'PRP$', 'WP', 'WP$'],
        'verb': ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
        'adj': ['JJ', 'JJR', 'JJS'],
        'adv': ['RB', 'RBR', 'RBS', 'WRB']
    }
    # se pueden anhadir mas!! # TODO

    #x es una lista de tuplas
    def check_pos_tag(x, flag):
        test_list = [tag for (word, tag) in x if tag in pos_family[flag]]
        count = len(test_list)
        return count

    new_feats['noun_count'] = users_df['pos_tags'].map(lambda x: check_pos_tag(x, 'noun'))
    new_feats['pron_count'] = users_df['pos_tags'].map(lambda x: check_pos_tag(x, 'pron'))
    new_feats['verb_count'] = users_df['pos_tags'].map(lambda x: check_pos_tag(x, 'verb'))
    new_feats['adj_count'] = users_df['pos_tags'].map(lambda x: check_pos_tag(x, 'adj'))
    new_feats['adv_count'] = users_df['pos_tags'].map(lambda x: check_pos_tag(x, 'adv'))
    print(new_feats.isnull().values.any())

    #normalize features by text length:
    #newFeats['word_count'] = newFeats['word_count'] / text_length

    # def normalize_feature(feature, normalizer):
    #     return feature / normalizer

    print("Before normalizing", new_feats.isnull().values.any())
    if normalize:
        for feature in new_feats.columns:
            if feature not in normalize_exceptions:
                new_feats[feature] = new_feats[feature] / text_length

    print("After normalizing", new_feats.isnull().values.any())

    for feat in exclude_features:
        new_feats.drop(feat, inplace=True, axis=1)

    # new features ideas:
    # calcular la media de longitud de todos los usuarios en otro lado y ver las desviaciones

    return new_feats



if __name__ == '__main__':
    main()