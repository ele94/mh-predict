# calculates features related to text characteristics

# after: windowfy.py
# next: combine_features.py OR train.py

import pickle
import pandas as pd
import numpy as np
from utils import load_pickle
from utils import save_pickle
from utils import load_parameters
from utils import remove_pickle
import filenames as fp

from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

import pandas, xgboost, numpy, textblob, string, re
from sklearn.model_selection import train_test_split
from nltk.sentiment.vader import SentimentIntensityAnalyzer


normalize_param = True


def main():

    params = load_parameters()
    window_size = params["feats_window_size"]
    feats_path = fp.get_feats_path()
    window_path = fp.get_window_path()

    remove_pickle(feats_path, fp.train_df_feats_filename)
    remove_pickle(feats_path, fp.test_df_feats_filename)

    train_x = load_pickle(window_path, fp.train_x_filename)
    test_x = load_pickle(window_path, fp.test_x_filename)

    train_feats = create_features(train_x, normalize_param)

    test_feats = create_features(test_x, normalize_param)

    save_pickle(feats_path, fp.train_df_feats_filename, train_feats)
    save_pickle(feats_path, fp.test_df_feats_filename, test_feats)

    #train_feats.to_csv(r'train_feats.csv')
    #test_feats.to_csv(r'test_feats.csv')


def create_features(users_df, normalize=True):

    normalize_exceptions = ['char_count', 'word_density']
    exclude_features = ['char_count', 'word_count']

    with open(fp.nssi_corpus_path, 'r') as file:
        nssi_corpus = file.read().replace('*', '')
    nssi_corpus = nssi_corpus.split('\n')
    nssi_corpus.remove('')

    new_feats = pd.DataFrame()

    text_length = users_df['clean_text'].map(len)

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

    sid = SentimentIntensityAnalyzer()
    new_feats['sentiment'] = users_df['clean_text'].map(lambda x: round(sid.polarity_scores(x)['compound'], 2))
    new_feats['nssi_words'] = users_df['tokens'].map(lambda x: sum((' '.join(x)).count(word) for word in nssi_corpus))
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

    #normalize features by text length:
    #newFeats['word_count'] = newFeats['word_count'] / text_length

    # def normalize_feature(feature, normalizer):
    #     return feature / normalizer

    if normalize:
        for feature in new_feats.columns:
            if feature not in normalize_exceptions:
                new_feats[feature] = new_feats[feature] / text_length

    for feat in exclude_features:
        new_feats.drop(feat, inplace=True, axis=1)

    # new features ideas:
    # calcular la media de longitud de todos los usuarios en otro lado y ver las desviaciones

    return new_feats



if __name__ == '__main__':
    main()