from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

import pandas, xgboost, numpy, textblob, string
import pickle
from utils import load_pickle
from utils import save_pickle
from utils import remove_pickle
from utils import load_parameters
import os

train_x_file = "train.x.pkl"
test_x_file = "test.x.pkl"
train_feats_file = "tfidf.train.pkl"
test_feats_file = "tfidf.test.pkl"


def main():

    params = load_parameters()
    window_size = params["feats_window_size"]

    train_word_file = str(window_size) + "." + train_feats_file
    test_word_file = str(window_size) + "." + test_feats_file

    remove_pickle(train_word_file)
    remove_pickle(test_word_file)

    train_x = load_pickle(train_x_file)
    test_x = load_pickle(test_x_file)

    max_features = load_parameters()["max_features"]

    print("Word-level tf-idf")
    # word level tf-idf
    tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=max_features)
    tfidf_vect.fit(train_x['clean_text'])
    xtrain_tfidf = tfidf_vect.transform(train_x["clean_text"])
    xtest_tfidf = tfidf_vect.transform(test_x["clean_text"])
    tfidf_vect = None

    save_pickle(train_word_file, xtrain_tfidf)
    save_pickle(test_word_file, xtest_tfidf)

    xtrain_tfidf = None
    xtest_tfidf = None

    # print("Ngram-level tf-idf")
    # # ngram level tf-idf
    # tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2, 3), max_features=max_features)
    # tfidf_vect_ngram.fit(train_x['clean_text'])
    # xtrain_tfidf_ngram = tfidf_vect_ngram.transform(train_x["clean_text"])
    # xtest_tfidf_ngram = tfidf_vect_ngram.transform(test_x["clean_text"])
    #
    # tfidf_vect_ngram = None
    #
    # save_pickle(train_ngram_file, xtrain_tfidf_ngram)
    # save_pickle(test_ngram_file, xtest_tfidf_ngram)



if __name__ == '__main__':
    main()