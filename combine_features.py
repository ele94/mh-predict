from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
import pickle
from numpy import hstack
import scipy

with open('data/pickles/train.df.feats.pkl', 'rb') as train_feats_file:
    train_word_feats = pickle.load(train_feats_file)

with open('data/pickles/xtrain.tfidf.word.pkl', 'rb') as train_tfidf_file:
    train_tfidf_feats = pickle.load(train_tfidf_file)

with open('data/pickles/xtrain.tfidf.ngram.pkl', 'rb') as train_ngram_file:
    train_ngram_feats = pickle.load(train_ngram_file)

with open('data/pickles/test.df.feats.pkl', 'rb') as test_feats_file:
    test_word_feats = pickle.load(test_feats_file)

with open('data/pickles/xtest.tfidf.word.pkl', 'rb') as test_tfidf_file:
    test_tfidf_feats = pickle.load(test_tfidf_file)

with open('data/pickles/xtest.tfidf.ngram.pkl', 'rb') as test_ngram_file:
    test_ngram_feats = pickle.load(test_ngram_file)



train_combined_features = scipy.sparse.hstack((train_tfidf_feats,train_ngram_feats,train_word_feats))

test_combined_features = scipy.sparse.hstack((test_tfidf_feats,test_ngram_feats,test_word_feats))



with open('data/pickles/train_cfeats.pkl', 'wb') as train_combined_feats:
    pickle.dump(train_combined_features, train_combined_feats)

with open('data/pickles/test_cfeats.pkl', 'wb') as test_combined_feats:
    pickle.dump(test_combined_features, test_combined_feats)