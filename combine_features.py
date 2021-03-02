from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
import pickle
from numpy import hstack
import scipy
from utils import load_pickle
from utils import save_pickle
from utils import load_parameters

train_df_feats_file = "train.df.feats.pkl"
test_df_feats_file = "test.df.feats.pkl"
train_ngram_file = "train.tfidf.ngram.pkl"
test_ngram_file = "test.tfidf.ngram.pkl"
train_word_file = "train.tfidf.pkl"
test_word_file = "test.tfidf.pkl"
train_combined_file = "train.cfeats.pkl"
test_combined_file = "test.cfeats.pkl"
train_feats_file = "train.feats.pkl"
test_feats_file = "test.feats.pkl"


def main():
    train_text_feats = load_pickle(train_df_feats_file)
    test_text_feats = load_pickle(test_df_feats_file)
    train_tfidf_feats = load_pickle(train_word_file)
    test_tfidf_feats = load_pickle(test_word_file)
    train_ngram_feats = load_pickle(train_ngram_file)
    test_ngram_feats = load_pickle(test_ngram_file)


    train_combined_features = scipy.sparse.hstack((train_tfidf_feats,train_ngram_feats,train_text_feats))
    test_combined_features = scipy.sparse.hstack((test_tfidf_feats,test_ngram_feats,test_text_feats))

    save_pickle(train_combined_file, train_combined_features)
    save_pickle(test_combined_file, test_combined_features)



if __name__ == '__main__':
    main()