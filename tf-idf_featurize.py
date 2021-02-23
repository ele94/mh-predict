from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

import pandas, xgboost, numpy, textblob, string
import pickle


def main():

    #with open('data/pickles/users.df.pkl', 'rb') as users_df_file:
    #    trainDF = pickle.load(users_df_file)

    with open('data/pickles/train.x.pkl', 'rb') as train_x_file:
        train_x = pickle.load(train_x_file)

    with open('data/pickles/test.x.pkl', 'rb') as test_x_file:
        test_x = pickle.load(test_x_file)



    # word level tf-idf
    tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
    tfidf_vect.fit(train_x['clean_text'])
    xtrain_tfidf = tfidf_vect.transform(train_x["clean_text"])
    xtest_tfidf = tfidf_vect.transform(test_x["clean_text"])

    # ngram level tf-idf
    tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2, 3), max_features=5000)
    tfidf_vect_ngram.fit(train_x['clean_text'])
    xtrain_tfidf_ngram = tfidf_vect_ngram.transform(train_x["clean_text"])
    xtest_tfidf_ngram = tfidf_vect_ngram.transform(test_x["clean_text"])


    with open('data/pickles/xtrain.tfidf.word.pkl', 'wb') as xtrain_tfidf_file:
        pickle.dump(xtrain_tfidf, xtrain_tfidf_file)
    with open('data/pickles/xtest.tfidf.word.pkl', 'wb') as xtest_tfidf_file:
        pickle.dump(xtest_tfidf, xtest_tfidf_file)

    with open('data/pickles/xtrain.tfidf.ngram.pkl', 'wb') as xtrain_tfidf_file:
        pickle.dump(xtrain_tfidf_ngram, xtrain_tfidf_file)
    with open('data/pickles/xtest.tfidf.ngram.pkl', 'wb') as xtest_tfidf_file:
        pickle.dump(xtest_tfidf_ngram, xtest_tfidf_file)



if __name__ == '__main__':
    main()