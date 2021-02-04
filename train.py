import pickle
import nltk
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import pandas as pd


def main():
    strategy = 'balanced'  # TODO parametrizar
    combine = False

    if combine:
        features_file = 'data/pickles/train_cfeats.pkl'
    else:
        features_file = 'data/pickles/xtrain.tfidf.word.pkl'

    with open(features_file, 'rb') as train_feats_file:
        train_x = pickle.load(train_feats_file)

    with open('data/pickles/train.y.pkl', 'rb') as train_y_file:
        train_labels = pickle.load(train_y_file)


    classifier = SVC(class_weight=strategy)

    classifier.fit(train_x, train_labels)

    with open('data/pickles/classifier.pkl', 'wb') as classifier_file:
        pickle.dump(classifier, classifier_file)





if __name__ == '__main__':
    main()