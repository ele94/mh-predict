from utils import load_pickle
from utils import save_pickle
from utils import remove_pickle
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

    params = load_parameters()
    param = params["feats"]

    remove_pickle(train_feats_file)
    remove_pickle(test_feats_file)

    # change here
    if param == "text":
        print("Selecting text feats")
        train_text_feats = load_pickle(train_df_feats_file)
        test_text_feats = load_pickle(test_df_feats_file)
        train_feats = train_text_feats
        test_feats = test_text_feats
    elif param == "tfidf":
        print("Selecting tfidf feats")
        train_tfidf_feats = load_pickle(train_word_file)
        test_tfidf_feats = load_pickle(test_word_file)
        train_feats = train_tfidf_feats
        test_feats = test_tfidf_feats
    elif param == "ngram":
        print("Selecting ngram feats")
        train_ngram_feats = load_pickle(train_ngram_file)
        test_ngram_feats = load_pickle(test_ngram_file)
        train_feats = train_ngram_feats
        test_feats = test_ngram_feats
    else:
        print("Selecting combined feats")
        train_c_feats = load_pickle(train_combined_file)
        test_c_feats = load_pickle(test_combined_file)
        train_feats = train_c_feats
        test_feats = test_c_feats


    save_pickle(train_feats_file, train_feats)
    save_pickle(test_feats_file, test_feats)


if __name__ == '__main__':
    main()