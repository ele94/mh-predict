from sklearn.feature_extraction.text import TfidfVectorizer

from utils import load_pickle
from utils import save_pickle
from utils import remove_pickle
from utils import load_parameters
from utils import logger
import filenames as fp


def main():

    params = load_parameters()
    window_size = params["feats_window_size"]
    feats_path = fp.get_feats_path()
    window_path = fp.get_window_path()
    feats = params["feats"]

    train_x = load_pickle(window_path, fp.train_x_filename)
    test_x = load_pickle(window_path, fp.test_x_filename)

    max_features = params["max_features"]

    if feats == "tfidf":

        remove_pickle(feats_path, fp.train_word_file)
        remove_pickle(feats_path, fp.test_word_file)

        logger("Word-level tf-idf")
        # word level tf-idf
        tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=max_features)
        tfidf_vect.fit(train_x['clean_text'])
        xtrain_tfidf = tfidf_vect.transform(train_x["clean_text"])
        xtest_tfidf = tfidf_vect.transform(test_x["clean_text"])
        del tfidf_vect

        save_pickle(feats_path, fp.train_word_file, xtrain_tfidf)
        save_pickle(feats_path, fp.test_word_file, xtest_tfidf)

        del xtrain_tfidf
        del xtest_tfidf

    else:

        logger("Ngram-level tf-idf")
        # ngram level tf-idf
        tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2, 3), max_features=max_features)
        tfidf_vect_ngram.fit(train_x['clean_text'])
        xtrain_tfidf_ngram = tfidf_vect_ngram.transform(train_x["clean_text"])
        xtest_tfidf_ngram = tfidf_vect_ngram.transform(test_x["clean_text"])

        del tfidf_vect_ngram

        save_pickle(feats_path, fp.train_ngram_file, xtrain_tfidf_ngram)
        save_pickle(feats_path, fp.test_ngram_file, xtest_tfidf_ngram)

        del xtrain_tfidf_ngram
        del xtest_tfidf_ngram


if __name__ == '__main__':
    main()