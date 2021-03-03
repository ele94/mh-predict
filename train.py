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

train_feats_filename = "train.pkl"
train_labels_filename = "train.y.pkl"
#classifier_file = "classifier.pkl"


def main():
    params = load_parameters()
    strategy = params["strategy"]
    classifier_name = params["classifier"]
    file_params = str(params["feats_window_size"]) + "." + params["feats"] + "."

    classifier_file = file_params + classifier_name + ".pkl"
    train_feats_file = file_params + train_feats_filename
    train_labels_file = str(params["feats_window_size"]) + "." + train_labels_filename

    remove_pickle(classifier_file)

    train_feats = load_pickle(train_feats_file)
    train_labels = load_pickle(train_labels_file)

    if classifier_name == "svm":
        classifier = SVC(class_weight=strategy)
    elif classifier_name == "linear_svm":
        classifier = LinearSVC(class_weight=strategy)
    elif classifier_name == "forest":
        classifier = ensemble.RandomForestClassifier(class_weight=strategy)
    elif classifier_name == "xgboost":
        classifier = xgboost.XGBClassifier(class_weight=strategy)
    else:
        classifier = naive_bayes.MultinomialNB()

    if not load_parameters()["feats"] == "text":
        train_feats = train_feats.tocsc()

    # if train_feats.isnull().values.any():
    #     train_feats = train_feats.fillna(value=0,axis=0)

    classifier.fit(train_feats, train_labels)
    save_pickle(classifier_file, classifier)





if __name__ == '__main__':
    main()