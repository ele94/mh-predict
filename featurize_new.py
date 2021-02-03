import pickle
import pandas as pd
import numpy as np

from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

import pandas, xgboost, numpy, textblob, string, re
from sklearn.model_selection import train_test_split
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nssi_corpus_path: str = "data/nssicorpus.txt"

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

    with open(nssi_corpus_path, 'r') as file:
        nssi_corpus = file.read().replace('*', '')
    nssi_corpus = nssi_corpus.split('\n')
    nssi_corpus.remove('')

    newFeats = pd.DataFrame()

    text_length = trainDF['clean_text'].apply(len)

    newFeats['char_count'] = trainDF['clean_text'].apply(len)
    newFeats['word_count'] = trainDF['clean_text'].apply(lambda x: len(x.split()))
    newFeats['word_density'] = text_length / (text_length + 1)
    #newFeats['word_count'] = newFeats['word_count'] / text_length

    newFeats['punctuation_count'] = trainDF['clean_text'].apply(
        lambda x: len("".join(_ for _ in x if _ in string.punctuation)))
    #newFeats['punctuation_count'] = newFeats['punctuation_count'] / text_length
    newFeats['upper_case_count'] = trainDF['clean_text'].apply(
        lambda x: len([wrd for wrd in x.split() if wrd.isupper()]))
    #newFeats['upper_case_count'] = newFeats['upper_case_count'] / text_length


    #my old features:
    #text features
    newFeats['questions_count'] = trainDF['text'].apply(lambda x: len(re.findall(r'\?', x)))
    #newFeats['questions_count'] = newFeats['questions_count'] / text_length
    newFeats['exclamations_count'] = trainDF['text'].apply(lambda x: len(re.findall(r'\!', x)))
    #newFeats['exclamations_count'] = newFeats['exclamations_count'] / text_length

    newFeats['smilies'] = trainDF['text'].apply(lambda x: len(re.findall(r'\:\)+|\(+\:', x)))
    #newFeats['smilies'] = newFeats['smilies'] / text_length
    newFeats['sad_faces'] = trainDF['text'].apply(lambda x: len(re.findall(r'\:\(+|\)+\:', x)))
    #newFeats['sad_faces'] = newFeats['sad_faces'] / text_length

    reg = r'\bI\b|\bme\b|\bmine\b|\bmy\b|\bmyself\b'
    newFeats['first_prons'] = trainDF['clean_text'].apply(lambda x: len(re.findall(reg, x)))
    #newFeats['first_prons'] = newFeats['first_prons'] / text_length

    sid = SentimentIntensityAnalyzer()
    newFeats['sentiment'] = trainDF['clean_text'].apply(lambda x: round(sid.polarity_scores(x)['compound'],2))
    #newFeats['sentiment'] = newFeats['sentiment'] / text_length

    newFeats['nssi_words'] = trainDF['tokens'].apply(lambda x: sum((' '.join(x)).count(word) for word in nssi_corpus))
    #newFeats['nssi_words'] = newFeats['nssi_words'] / text_length

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

    # newFeats['noun_count'] = trainDF['pos_tags'].apply(lambda x: check_pos_tag(x, 'noun'))
    # newFeats['noun_count'] = newFeats['noun_count'] / text_length
    # newFeats['pron_count'] = trainDF['pos_tags'].apply(lambda x: check_pos_tag(x, 'pron'))
    # newFeats['pron_count'] = newFeats['pron_count'] / text_length
    # newFeats['verb_count'] = trainDF['pos_tags'].apply(lambda x: check_pos_tag(x, 'verb'))
    # newFeats['verb_count'] = newFeats['verb_count'] / text_length
    # newFeats['adj_count'] = trainDF['pos_tags'].apply(lambda x: check_pos_tag(x, 'adj'))
    # newFeats['adj_count'] = newFeats['adj_count'] / text_length
    # newFeats['adv_count'] = trainDF['pos_tags'].apply(lambda x: check_pos_tag(x, 'adv'))
    # newFeats['adv_count'] = newFeats['adv_count'] / text_length




    # new features ideas:
    # calcular la media de longitud de todos los usuarios en otro lado y ver las desviaciones

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