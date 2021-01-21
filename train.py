import pickle
import nltk
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import pandas as pd


def main():
    strategy = 'balanced'  # TODO parametrizar

    features = "train.df.feats"

    #with open('data/pickles/train.feats.mod.pkl', 'rb') as train_feats_file:
    #    train_x = pickle.load(train_feats_file)

    with open('data/pickles/train_cfeats.pkl', 'rb') as train_feats_file:
        train_x = pickle.load(train_feats_file)

    with open('data/pickles/train.y.pkl', 'rb') as train_y_file:
        train_labels = pickle.load(train_y_file)



    #print(train_feats[0]) # TODO delete later

    classifier = SVC(class_weight=strategy)

    classifier.fit(train_x, train_labels)

    with open('data/pickles/classifier.pkl', 'wb') as classifier_file:
        pickle.dump(classifier, classifier_file)





if __name__ == '__main__':
    main()