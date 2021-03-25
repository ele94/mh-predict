from sklearn.feature_extraction.text import TfidfVectorizer

from utils import load_pickle
from utils import save_pickle
from utils import remove_pickle
from utils import load_parameters
from utils import logger
import filenames as fp


def tfidf():
    params = load_parameters()
    feat = "tfidf"
    max_features = params["max_features"]
    type = params["tfidf_type"]
    if type == "all":
        only_positives = False
    else:
        only_positives = True

    get_features(feat, max_features, only_positives=only_positives)


def ngrams():
    params = load_parameters()
    feat = "ngram"
    max_features = params["max_features"]

    get_features(feat, max_features)




def get_features(feat, max_features, only_positives=True):

    feats_path = fp.get_feats_path()
    window_path = fp.get_window_path()

    if feat == "tfidf":
        train_file = fp.train_word_file
        test_file = fp.test_word_file
        ngram_range = (1, 1)
    else:
        train_file = fp.train_ngram_file
        test_file = fp.test_ngram_file
        ngram_range = (2, 3)

    train_x = load_pickle(window_path, fp.train_x_filename)
    test_x = load_pickle(window_path, fp.test_x_filename)

    remove_pickle(feats_path, train_file)
    remove_pickle(feats_path, test_file)
    # word level tf-idf
    tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=max_features, ngram_range=ngram_range)

    if only_positives:
        g_truth = {line.split()[0]: int(line.split()[1]) for line in open(fp.train_g_truth_filename)}
        train_x_positives = [text for text, user in zip(train_x['clean_text'], train_x['user']) if g_truth[user] == 1]
        tfidf_vect.fit(train_x_positives)
    else:
        tfidf_vect.fit(train_x['clean_text'])  # aqui pasar solo los positivos???

    xtrain_tfidf = tfidf_vect.transform(train_x["clean_text"])
    xtest_tfidf = tfidf_vect.transform(test_x["clean_text"])
    del tfidf_vect

    save_pickle(feats_path, train_file, xtrain_tfidf)
    save_pickle(feats_path, test_file, xtest_tfidf)

    del xtrain_tfidf
    del xtest_tfidf


def main():
    return tfidf()



if __name__ == '__main__':
    main()