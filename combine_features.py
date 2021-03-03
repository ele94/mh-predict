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
from utils import remove_pickle
from utils import load_parameters

train_df_feats_file = "text.train.pkl"
test_df_feats_file = "text.test.pkl"
train_word_file = "tfidf.train.pkl"
test_word_file = "tfidf.test.pkl"
train_combined_file = "combined.train.pkl"
test_combined_file = "combined.test.pkl"
train_feats_file = "train.pkl"
test_feats_file = "test.pkl"


def main():

    params = load_parameters()
    file_params = str(params["feats_window_size"]) + "."

    remove_pickle(train_combined_file)
    remove_pickle(test_combined_file)

    train_text_feats = load_pickle(file_params+train_df_feats_file)
    test_text_feats = load_pickle(file_params+test_df_feats_file)
    train_tfidf_feats = load_pickle(file_params+train_word_file)
    test_tfidf_feats = load_pickle(file_params+test_word_file)
    # train_ngram_feats = load_pickle(train_ngram_file)
    # test_ngram_feats = load_pickle(test_ngram_file)


    train_combined_features = scipy.sparse.hstack((train_tfidf_feats,train_text_feats))
    test_combined_features = scipy.sparse.hstack((test_tfidf_feats,test_text_feats))

    save_pickle(file_params + train_combined_file, train_combined_features)
    save_pickle(file_params + test_combined_file, test_combined_features)



if __name__ == '__main__':
    main()