import pickle
from sklearn.svm import SVC
import pandas as pd


def main():

    combine = False
    tfidf = True

    if combine:
        features_file = 'data/pickles/test_cfeats.pkl'
    elif tfidf:
        features_file = "data/pickles/xtest.tfidf.ngram.pkl"
    else:
        features_file = 'data/pickles/test.df.feats.pkl'

    print(features_file)

    with open('data/pickles/classifier.pkl', 'rb') as classifier_file:
        classifier = pickle.load(classifier_file)

    with open(features_file, 'rb') as test_feats_file:
        test_feats = pickle.load(test_feats_file)




    #test_resul = classifier.classify_many(test_feats)

    test_x = test_feats

    predictions = classifier.predict(test_x)

    #with open('data/pickles/test.resul.pkl', 'wb') as test_resul_file:
    #    pickle.dump(test_resul, test_resul_file)

    with open('data/pickles/test.resul.mod.pkl', 'wb') as test_resul_mod_file:
        pickle.dump(predictions, test_resul_mod_file)


if __name__ == '__main__':
    main()