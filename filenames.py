from utils import load_parameters
import os


# filenames
train_filename = "clean.train.data.pkl"
test_filename = "clean.test.data.pkl"
train_x_filename = "train.x.pkl"
test_x_filename = "test.x.pkl"
train_y_filename = "train.y.pkl"
test_y_filename = "test.y.pkl"
train_df_feats_filename = "text.train.pkl"
test_df_feats_filename = "text.test.pkl"
train_word_file = "tfidf.train.pkl"
test_word_file = "tfidf.test.pkl"
train_ngram_file = "ngram.train.pkl"
test_ngram_file = "ngram.test.pkl"
train_combined_file = "combined.train.pkl"
test_combined_file = "combined.test.pkl"
train_feats_file = "train.pkl"
test_feats_file = "test.pkl"
resul_file = "test.resul.pkl"
score_file = "test.scores.pkl"
train_weights_file = "train.weights.pkl"

g_truth_filename = "/home/elena/Documentos/UNED/erisk/2021/data/erisk2021_test_data/golden_truth.txt"

nssi_corpus_path = "data/nssicorpus.txt"


#filepaths
pickles_path = "data/pickles"


def get_window_path():
    filepath = os.path.join(pickles_path, str(load_parameters()["range_max"]), str(load_parameters()["feats_window_size"]))
    return filepath

def get_feats_path():
    filepath = get_window_path()
    return filepath

def get_classifier_path():
    params = load_parameters()
    return os.path.join(get_feats_path(), str(params["feats"]), str(params["strategy"]), str(params["classifier"]))

def get_resuls_path():
    return get_classifier_path()
