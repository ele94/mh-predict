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
from utils import load_parameters
from utils import remove_pickle
import xgboost
import filenames as fp

#classifier_file = "classifier.pkl"


def main():
    params = load_parameters()
    strategy = params["strategy"]
    feats = params["feats"]
    classifier_name = params["classifier"]
    classifier_path = fp.get_classifier_path()
    feats_path = fp.get_feats_path()
    window_path = fp.get_window_path()

    classifier_file = classifier_name + ".pkl"
    train_feats_file = feats + "." + fp.train_feats_file

    remove_pickle(classifier_path, classifier_file)

    train_feats = load_pickle(feats_path, train_feats_file)
    train_labels = load_pickle(window_path, fp.train_y_filename)

    if classifier_name == "svm":
        classifier = SVC(class_weight=strategy)
    elif classifier_name == "linear_svm":
        classifier = LinearSVC(class_weight=strategy)
    elif classifier_name == "forest":
        classifier = ensemble.RandomForestClassifier()
    elif classifier_name == "xgboost":
        classifier = xgboost.XGBClassifier(class_weight=strategy)
    else:
        classifier = naive_bayes.MultinomialNB()

    if feats != 'text':
        train_feats = train_feats.tocsc()

    # if train_feats.isnull().values.any():
    #     train_feats = train_feats.fillna(value=0,axis=0)

    classifier.fit(train_feats, train_labels)
    save_pickle(classifier_path, classifier_file, classifier)





if __name__ == '__main__':
    main()