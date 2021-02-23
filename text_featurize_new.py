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

    normalize = True

    #with open('data/pickles/test.users.pkl', 'rb') as test_users_file:
    #    test_users = pickle.load(test_users_file)
    #with open('data/pickles/train.users.pkl', 'rb') as train_users_file:
    #    train_users = pickle.load(train_users_file)


    with open('data/pickles/train.x.pkl', 'rb') as train_df_file:
        train_x = pickle.load(train_df_file)

    with open('data/pickles/test.x.pkl', 'rb') as test_df_file:
        test_x = pickle.load(test_df_file)






    train_feats = create_features(train_x, normalize)
    test_feats = create_features(test_x, normalize)

    print(train_feats)

    with open('data/pickles/train.df.feats.pkl', 'wb') as train_feats_file:
        pickle.dump(train_feats, train_feats_file)

    with open('data/pickles/test.df.feats.pkl', 'wb') as test_feats_file:
        pickle.dump(test_feats, test_feats_file)


def create_features(df, normalize=True):

    normalize_exceptions = ['char_count', 'word_density']
    exclude_features = ['char_count', 'word_count']

    with open(nssi_corpus_path, 'r') as file:
        nssi_corpus = file.read().replace('*', '')
    nssi_corpus = nssi_corpus.split('\n')
    nssi_corpus.remove('')

    newFeats = pd.DataFrame()

    newFeats['char_count'] = df['clean_text'].apply(lambda x: list(map(lambda y: len(y), x)))
    print(df['text'][0])
    print(df['title'][0])
    print(df['clean_text'][0])
    print(newFeats['char_count'][0])
    newFeats['word_count'] = df['clean_text'].apply(lambda x: list(map(lambda y: len(y.split()), x)))
    #newFeats['word_density'] = newFeats['char_count'] / (newFeats['char_count'] + 1)  # ???

    newFeats['word_density'] = newFeats['char_count'].apply(lambda x: list(map(lambda y: y/(y+1), x)))
    print(newFeats['word_density'][0])

    newFeats['punctuation_count'] = df['clean_text'].apply(
        lambda x: list(map(lambda y: len("".join(_ for _ in y if _ in string.punctuation)), x)))
    newFeats['upper_case_count'] = df['clean_text'].apply(
        lambda x: list(map(lambda y: len([wrd for wrd in y.split() if wrd.isupper()]), x)))


    #my old features:
    #text features
    newFeats['questions_count'] = df['text'].apply(lambda x: list(map(lambda y: len(re.findall(r'\?', y)), x)))
    newFeats['exclamations_count'] = df['text'].apply(lambda x: list(map(lambda y: len(re.findall(r'\!', y)), x)))
    newFeats['smilies'] = df['text'].apply(lambda x: list(map(lambda y: len(re.findall(r'\:\)+|\(+\:', y)), x)))
    newFeats['sad_faces'] = df['text'].apply(lambda x: list(map(lambda y: len(re.findall(r'\:\(+|\)+\:', y)), x)))

    reg = r'\bI\b|\bme\b|\bmine\b|\bmy\b|\bmyself\b'
    newFeats['first_prons'] = df['clean_text'].apply(lambda x: list(map(lambda y: len(re.findall(reg, y)), x)))

    sid = SentimentIntensityAnalyzer()
    newFeats['sentiment'] = df['clean_text'].apply(lambda x: list(map(lambda y: round(sid.polarity_scores(y)['compound'], 2), x)))

    newFeats['nssi_words'] = df['tokens'].apply(lambda x: list(map(lambda y: sum((' '.join(y)).count(word)
                                                                                 for word in nssi_corpus), x)))

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

    newFeats['noun_count'] = df['pos_tags'].apply(lambda x: list(map(lambda y: check_pos_tag(y, 'noun'), x)))
    newFeats['pron_count'] = df['pos_tags'].apply(lambda x: list(map(lambda y: check_pos_tag(y, 'pron'), x)))
    newFeats['verb_count'] = df['pos_tags'].apply(lambda x: list(map(lambda y: check_pos_tag(y, 'verb'), x)))
    newFeats['adj_count'] = df['pos_tags'].apply(lambda x: list(map(lambda y: check_pos_tag(y, 'adj'), x)))
    newFeats['adv_count'] = df['pos_tags'].apply(lambda x: list(map(lambda y: check_pos_tag(y, 'adv'), x)))


    #normalize features by text length:
    #newFeats['word_count'] = newFeats['word_count'] / text_length

    #def normalize_feature(feature, normalizer):
    #    return feature / normalizer

    # nose hacer esto de forma mas limpia
    if normalize:
        for feature in newFeats.columns:
            if feature not in normalize_exceptions:
                for index, user in enumerate(newFeats[feature]):
                    for i, feat in enumerate(user):
                        if newFeats['char_count'][index][i] > 0:
                            newFeats[feature][index][i] = newFeats[feature][index][i] / newFeats['char_count'][index][i]

                #newFeats[feature].apply(lambda x: list(map(lambda y: x/y, newFeats['char_count'])))
                #newFeats[feature] = newFeats[feature].apply(lambda x: lambda y: list(map(normalize_feature(y, newFeats['char_count']), x)))

    for feat in exclude_features:
        newFeats.drop(feat, inplace=True, axis=1)

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

    print(newFeats.columns)
    return newFeats



if __name__ == '__main__':
    main()