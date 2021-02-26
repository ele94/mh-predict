# trains the model

# after: text_featurize.py / tfidf_featurize.py / combine_features.py
# next: classify.py

import pickle
import nltk
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn import naive_bayes, ensemble
import pandas as pd
import numpy as np
from utils import load_pickle
from utils import save_pickle

train_feats_file = "train.feats.pkl"
train_labels_file = "train.y.pkl"
classifier_file = "classifier.pkl"


def main():
    strategy = 'balanced'  # TODO parametrizar

    train_feats = load_pickle(train_feats_file)
    train_labels = load_pickle(train_labels_file)


    classifier = SVC(class_weight=strategy)
    #classifier = SVC()
    #classifier = naive_bayes.MultinomialNB()
    #classifier = ensemble.RandomForestClassifier()

    # if train_feats.isnull().values.any():
    #     train_feats = train_feats.fillna(value=0,axis=0)

    classifier.fit(train_feats, train_labels)

    save_pickle(classifier_file, classifier)





if __name__ == '__main__':
    main()