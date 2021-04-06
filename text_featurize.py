# calculates features related to text characteristics

# after: windowfy.py
# next: combine_features.py OR train.py

import pandas as pd
from utils import load_pickle
from utils import save_pickle
from utils import load_parameters
from utils import remove_pickle
import filenames as fp

import string, re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.preprocessing import KBinsDiscretizer


normalize_param = True


def main():

    params = load_parameters()
    window_size = params["feats_window_size"]
    feats_path = fp.get_feats_path()
    window_path = fp.get_window_path()

    remove_pickle(feats_path, fp.train_df_feats_filename)
    remove_pickle(feats_path, fp.test_df_feats_filename)

    train_x = load_pickle(window_path, fp.train_x_filename)
    test_x = load_pickle(window_path, fp.test_x_filename)

    if params["text_features"] == "all":
        train_feats = create_features(train_x, normalize_param)
        test_feats = create_features(test_x, normalize_param)
    else:
        prons = params["prons"]
        nssi = params["nssi"]
        train_feats = create_selects2_features(train_x, normalize_param, prons=prons, nssi=nssi)
        test_feats = create_selects2_features(test_x, normalize_param, prons=prons, nssi=nssi)
        print(train_feats)

    if params["discretize"] == True:
        size = params["discretize_size"]
        strategy = params["discretize_strategy"]
        encode = params["discretize_encode"]

        train_feats, test_feats = discretize_features(train_feats, test_feats,
                                                      size=size, strategy=strategy,
                                                      encode=encode)

    save_pickle(feats_path, fp.train_df_feats_filename, train_feats)
    save_pickle(feats_path, fp.test_df_feats_filename, test_feats)




    #train_feats.to_csv(r'train_feats.csv')
    #test_feats.to_csv(r'test_feats.csv')


# def add_user_sequence(users_df, feats):
#     feats["user"] = users_df["user"]
#     feats["sequence"] = users_df["sequence"]
#     feats["g_truth"] = users_df["g_truth"]
#     return feats

def create_features(users_df, normalize=True):

    normalize_exceptions = ['char_count', 'word_density']
    exclude_features = ['char_count', 'word_count']

    nssi_corpus = load_nssi_corpus()

    new_feats = pd.DataFrame()

    text_length = users_df['clean_text'].map(len)

    new_feats['char_count'] = users_df['clean_text'].map(len)
    new_feats['word_count'] = users_df['clean_text'].map(lambda x: len(x.split()))
    new_feats['word_density'] = text_length / (text_length + 1)

    new_feats['punctuation_count'] = users_df['clean_text'].map(
        lambda x: len("".join(_ for _ in x if _ in string.punctuation)))
    new_feats['upper_case_count'] = users_df['clean_text'].map(
        lambda x: len([wrd for wrd in x.split() if wrd.isupper()]))

    #my old features:
    #text features
    new_feats['questions_count'] = users_df['text'].map(lambda x: len(re.findall(r'\?', x)))
    new_feats['exclamations_count'] = users_df['text'].map(lambda x: len(re.findall(r'\!', x)))
    new_feats['smilies'] = users_df['text'].map(lambda x: len(re.findall(r'\:\)+|\(+\:', x)))
    new_feats['sad_faces'] = users_df['text'].map(lambda x: len(re.findall(r'\:\(+|\)+\:', x)))

    reg = r'\bI\b|\bme\b|\bmine\b|\bmy\b|\bmyself\b'
    new_feats['first_prons'] = users_df['clean_text'].map(lambda x: len(re.findall(reg, x)))

    sid = SentimentIntensityAnalyzer()
    new_feats['sentiment'] = users_df['clean_text'].map(lambda x: round(sid.polarity_scores(x)['compound'], 2))
    for key, values in nssi_corpus.items():
        new_feats[key] = users_df['stems'].map(lambda x: sum((' '.join(x)).count(word) for word in values))
    #new_feats['nssi_words'] = users_df['stems'].map(lambda x: sum((' '.join(x)).count(word) for word in nssi_corpus))
    pos_family = {
        'noun': ['NN', 'NNS', 'NNP', 'NNPS'],
        'pron': ['PRP', 'PRP$', 'WP', 'WP$'],
        'verb': ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
        'adj': ['JJ', 'JJR', 'JJS'],
        'adv': ['RB', 'RBR', 'RBS', 'WRB']
    }
    # se pueden anhadir mas!! # TODO

    #x es una lista de tuplas
    def check_pos_tag(x, flag):
        test_list = [tag for (word, tag) in x if tag in pos_family[flag]]
        count = len(test_list)
        return count

    new_feats['noun_count'] = users_df['pos_tags'].map(lambda x: check_pos_tag(x, 'noun'))
    new_feats['pron_count'] = users_df['pos_tags'].map(lambda x: check_pos_tag(x, 'pron'))
    new_feats['verb_count'] = users_df['pos_tags'].map(lambda x: check_pos_tag(x, 'verb'))
    new_feats['adj_count'] = users_df['pos_tags'].map(lambda x: check_pos_tag(x, 'adj'))
    new_feats['adv_count'] = users_df['pos_tags'].map(lambda x: check_pos_tag(x, 'adv'))

    #normalize features by text length:
    #newFeats['word_count'] = newFeats['word_count'] / text_length

    # def normalize_feature(feature, normalizer):
    #     return feature / normalizer

    if normalize:
        for feature in new_feats.columns:
            if feature not in normalize_exceptions:
                new_feats[feature] = new_feats[feature] / text_length

    for feat in exclude_features:
        new_feats.drop(feat, inplace=True, axis=1)

    # new features ideas:
    # calcular la media de longitud de todos los usuarios en otro lado y ver las desviaciones

    return new_feats


def create_selects2_features(users_df, normalize=True, nssi=True, prons=True):
    new_feats = pd.DataFrame()
    text_length = users_df['clean_text'].map(len)

    if prons:
        reg = r'\bI\b|\bme\b|\bmine\b|\bmy\b|\bmyself\b'
        new_feats['first_prons'] = users_df['clean_text'].map(lambda x: len(re.findall(reg, x)))

    if nssi:
        nssi_corpus = load_nssi_corpus()
        for key, values in nssi_corpus.items():
            new_feats[key] = users_df['stems'].map(lambda x: sum((' '.join(x)).count(word) for word in values))

    if normalize:
        for feature in new_feats.columns:
            new_feats[feature] = new_feats[feature] / text_length

    return new_feats


def create_select_features(users_df, normalize=True):

    normalize_exceptions = ['char_count', 'word_density']
    exclude_features = ['char_count', 'word_count']

    nssi_corpus = load_nssi_corpus()

    new_feats = pd.DataFrame()

    text_length = users_df['clean_text'].map(len)

    new_feats['char_count'] = users_df['clean_text'].map(len)
    new_feats['word_count'] = users_df['clean_text'].map(lambda x: len(x.split()))
    #new_feats['word_density'] = text_length / (text_length + 1)

    new_feats['punctuation_count'] = users_df['clean_text'].map(
        lambda x: len("".join(_ for _ in x if _ in string.punctuation)))
    # new_feats['upper_case_count'] = users_df['clean_text'].map(
    #     lambda x: len([wrd for wrd in x.split() if wrd.isupper()]))

    #my old features:
    #text features
    new_feats['questions_count'] = users_df['text'].map(lambda x: len(re.findall(r'\?', x)))
    new_feats['exclamations_count'] = users_df['text'].map(lambda x: len(re.findall(r'\!', x)))
    new_feats['smilies'] = users_df['text'].map(lambda x: len(re.findall(r'\:\)+|\(+\:', x)))
    new_feats['sad_faces'] = users_df['text'].map(lambda x: len(re.findall(r'\:\(+|\)+\:', x)))

    reg = r'\bI\b|\bme\b|\bmine\b|\bmy\b|\bmyself\b'
    new_feats['first_prons'] = users_df['clean_text'].map(lambda x: len(re.findall(reg, x)))

    sid = SentimentIntensityAnalyzer()
    new_feats['sentiment'] = users_df['clean_text'].map(lambda x: round(sid.polarity_scores(x)['compound'], 2))
    for key, values in nssi_corpus.items():
        new_feats[key] = users_df['stems'].map(lambda x: sum((' '.join(x)).count(word) for word in values))
    #new_feats['nssi_words'] = users_df['stems'].map(lambda x: sum((' '.join(x)).count(word) for word in nssi_corpus))
    pos_family = {
        'noun': ['NN', 'NNS', 'NNP', 'NNPS'],
        'pron': ['PRP', 'PRP$', 'WP', 'WP$'],
        'verb': ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
        'adj': ['JJ', 'JJR', 'JJS'],
        'adv': ['RB', 'RBR', 'RBS', 'WRB']
    }
    # se pueden anhadir mas!! # TODO

    #x es una lista de tuplas
    def check_pos_tag(x, flag):
        test_list = [tag for (word, tag) in x if tag in pos_family[flag]]
        count = len(test_list)
        return count

    # new_feats['noun_count'] = users_df['pos_tags'].map(lambda x: check_pos_tag(x, 'noun'))
    # new_feats['pron_count'] = users_df['pos_tags'].map(lambda x: check_pos_tag(x, 'pron'))
    # new_feats['verb_count'] = users_df['pos_tags'].map(lambda x: check_pos_tag(x, 'verb'))
    # new_feats['adj_count'] = users_df['pos_tags'].map(lambda x: check_pos_tag(x, 'adj'))
    # new_feats['adv_count'] = users_df['pos_tags'].map(lambda x: check_pos_tag(x, 'adv'))

    #normalize features by text length:
    #newFeats['word_count'] = newFeats['word_count'] / text_length

    # def normalize_feature(feature, normalizer):
    #     return feature / normalizer

    if normalize:
        for feature in new_feats.columns:
            if feature not in normalize_exceptions:
                new_feats[feature] = new_feats[feature] / text_length

    for feat in exclude_features:
        new_feats.drop(feat, inplace=True, axis=1)

    # new features ideas:
    # calcular la media de longitud de todos los usuarios en otro lado y ver las desviaciones

    return new_feats


def discretize_features(train_feats, test_feats, size=3, strategy='uniform', encode='ordinal'):
    est = KBinsDiscretizer(n_bins=size, encode=encode, strategy=strategy)
    train = est.fit_transform(train_feats)
    test = est.transform(test_feats)

    return train, test


def load_nssi_corpus():

    with open(fp.nssi_corpus_path, 'r') as file:
        nssi_corpus_original = file.read()

    nssi_corpus = nssi_corpus_original.replace('*', '')
    nssi_corpus = nssi_corpus.replace("Methods of NSSI", '')
    nssi_corpus = nssi_corpus.replace("NSSI Terms", '')
    nssi_corpus = nssi_corpus.replace("Instruments Used", '')
    nssi_corpus = nssi_corpus.replace("Reasons for NSSI", '')

    keys = ["methods", "terms", "instruments", "reasons"]

    nssi_corpus = nssi_corpus.split(':')
    nssi_corpus.remove('')
    nssi_corpus = [corpus.split("\n") for corpus in nssi_corpus]
    new_nssi_corpus = {}
    for idx, corpus in enumerate(nssi_corpus):
        new_list = [word for word in corpus if word != ""]
        new_nssi_corpus[keys[idx]] = new_list

    return new_nssi_corpus

if __name__ == '__main__':
    main()