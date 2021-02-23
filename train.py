import pickle
import nltk
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn import naive_bayes, ensemble
import pandas as pd
import numpy as np


def main():
    strategy = 'balanced'  # TODO parametrizar
    combine = False
    tfidf = True

    if combine:
        features_file = 'data/pickles/train_cfeats.pkl'
    elif tfidf:
        features_file = "data/pickles/xtrain.tfidf.ngram.pkl"
    else:
        features_file = 'data/pickles/train.df.feats.pkl'

    print(features_file)

    with open(features_file, 'rb') as train_feats_file:
        train_x = pickle.load(train_feats_file)

    with open('data/pickles/train.y.pkl', 'rb') as train_y_file:
        train_labels = pickle.load(train_y_file)


    classifier = SVC(class_weight=strategy)
    #classifier = SVC()
    #classifier = naive_bayes.MultinomialNB()
    #classifier = ensemble.RandomForestClassifier()

    classifier.fit(train_x, train_labels)

    with open('data/pickles/classifier.pkl', 'wb') as classifier_file:
        pickle.dump(classifier, classifier_file)





if __name__ == '__main__':
    main()