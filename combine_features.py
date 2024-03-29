from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
import pickle
from numpy import hstack
from scipy.sparse import hstack
from utils import load_pickle
from utils import save_pickle
from utils import remove_pickle
from utils import check_pickle
from utils import load_parameters
import filenames as fp


def main():

    params = load_parameters()
    feats_path = fp.get_feats_path()
    window_path = fp.get_window_path()

    remove_pickle(feats_path, fp.train_combined_file)
    remove_pickle(feats_path, fp.test_combined_file)

    train_feats = []
    test_feats = []

    train_text_feats = load_pickle(feats_path, fp.train_df_feats_filename)
    test_text_feats = load_pickle(feats_path, fp.test_df_feats_filename)

    if params["tfidf_ngrams"]:
        train_ngram_feats = load_pickle(feats_path, fp.train_ngram_file)
        test_ngram_feats = load_pickle(feats_path, fp.test_ngram_file)
        train_feats.append(train_ngram_feats)
        test_feats.append(test_ngram_feats)
    else:
        train_tfidf_feats = load_pickle(feats_path, fp.train_word_file)
        test_tfidf_feats = load_pickle(feats_path, fp.test_word_file)
        train_feats.append(train_tfidf_feats)
        test_feats.append(test_tfidf_feats)

    train_feats.append(train_text_feats)
    test_feats.append(test_text_feats)



    # train_combined_features = scipy.sparse.hstack((train_tfidf_feats,train_ngram_feats,train_text_feats))
    # test_combined_features = scipy.sparse.hstack((test_tfidf_feats,test_ngram_feats,test_text_feats))

    train_combined_features = hstack(train_feats)
    test_combined_features = hstack(test_feats)

    print("Is the combined different from tfidf: {}".format(
        train_feats[0].toarray() == train_combined_features.toarray()))

    save_pickle(feats_path, fp.train_combined_file, train_combined_features)
    save_pickle(feats_path, fp.test_combined_file, test_combined_features)



if __name__ == '__main__':
    main()