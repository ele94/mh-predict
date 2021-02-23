import pickle
import contractions
from nltk import word_tokenize
import nltk
import re, itertools
from nltk.corpus import stopwords
import pandas as pd
from pprint import pprint
import itertools
from sklearn import preprocessing


def main():

    with open('data/pickles/train.x.pkl', 'rb') as train_df_file:
        train_x = pickle.load(train_df_file)

    with open('data/pickles/test.x.pkl', 'rb') as test_df_file:
        test_x = pickle.load(test_df_file)

    with open('data/pickles/train.y.pkl', 'rb') as train_y_file:
        train_y = pickle.load(train_y_file)

    with open('data/pickles/test.y.pkl', 'rb') as test_y_file:
        test_y = pickle.load(test_y_file)

    window_type = "count"  # (count, size or time)
    window_size = -1

    train_window = windowfy(train_x, window_size)
    test_window = windowfy(test_x, window_size)




    train_window_frame = pd.DataFrame(train_window)
    test_window_frame = pd.DataFrame(test_window)
    print(train_window_frame)

    train_x, train_y = train_window_frame[
                           ["user", "date", "text", "title", "g_truth", "clean_text", "tokens", "pos_tags"]], \
                       train_window_frame[['g_truth']]
    test_x, test_y = test_window_frame[
                         ["user", "date", "text", "title", "g_truth", "clean_text", "tokens", "pos_tags"]], \
                     test_window_frame[['g_truth']]

    encoder = preprocessing.LabelEncoder()
    train_y = encoder.fit_transform(train_y)
    test_y = encoder.fit_transform(test_y)

    with open('data/pickles/train.x.pkl', 'wb') as train_x_file:
        pickle.dump(train_x, train_x_file)

    with open('data/pickles/train.y.pkl', 'wb') as train_y_file:
        pickle.dump(train_y, train_y_file)

    with open('data/pickles/test.x.pkl', 'wb') as test_x_file:
        pickle.dump(test_x, test_x_file)

    with open('data/pickles/test.y.pkl', 'wb') as test_y_file:
        pickle.dump(test_y, test_y_file)








def windowfy(data_x, window_size):

    new_data_dict = {}

    for index, row in data_x.iterrows():
        if row["user"] in new_data_dict.keys():
            new_data_dict[row["user"]].append(row.to_dict())
        else:
            new_data_dict[row["user"]] = [row.to_dict()]

    data_x = None
    print("Llega hasta aqui")
    window_writings = []

    for user, user_writings in new_data_dict.items():
        print(user)
        print(len(user_writings))
        if(len(user_writings) > 1000):
            continue
        if window_size > 0:
            writings_length = len(user_writings)
            steps = writings_length / window_size
        else:
            steps = 1

        # this is not sliding window lmao
        #flatten = lambda l: [item for sublist in l for item in sublist]
        for step in range(0, int(steps)):
            print(step)

            working_writings = user_writings[window_size * step:window_size * (step + 1)]
            window_writing = join_window_elements(working_writings)
            window_writing["user"] = working_writings[0]["user"]
            window_writing["g_truth"] = working_writings[0]["g_truth"]

            # date = []
            # text = []
            # title = []
            # clean_text = []
            # tokens = []
            # pos_tags = []
            # for writing in working_writings:
            #     date.append(writing["date"])
            #     text.append(writing["text"])
            #     title.append(writing["title"])
            #     clean_text.append(writing["clean_text"])
            #     tokens.append(writing["tokens"])
            #     pos_tags.append(writing["pos_tags"])
            #
            # text = '. '.join(text)
            # title = '. '.join(title)
            # clean_text = '. '.join(clean_text)
            # tokens = flatten(tokens)
            # pos_tags = flatten(pos_tags)
            #
            # window_writing = {"user": user, "g_truth": user_writings[0]["g_truth"], "date": date, "text": text,
            #                   "title": title, "clean_text": clean_text, "tokens": tokens, "pos_tags": pos_tags}
            window_writings.append(window_writing)

    return window_writings



def join_window_elements(window: list) -> dict:
    joint_window = {}
    flatten = lambda l: [item for sublist in l for item in sublist]

    for key in window[0].keys():
        help = [message[key] for message in window]
        if type(help[0]) is list:
            joint_window[key] = flatten(help)
        elif key == 'user':
            pass
        elif key == 'g_truth':
            pass
        elif key == 'date':
            joint_window[key] = help
        else:
            joint_window[key] = ' .'.join(help)

    return joint_window


#
#
#
# def clean_text(text):
#     if len(text) <= 0:
#         return text
#     text = text.strip()
#     text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
#     text = re.sub(r' #[0-9]+;', '', text)
#     text = re.sub(r"[^\w\d'\s]+", ' ', text)
#     text = re.sub('  ', ' ', text)
#     text = contractions.fix(text)  # NEW!!!!
#     return text
#
# def tokenize_text(text):
#     if len(text) <= 0:
#         return text
#     text = text.lower()
#     text = remove_stopwords(text)
#     text = word_tokenize(text)
#     return text
#
# # text tiene que venir en tokens
# def pos_tag_text(text):
#     if len(text) <= 0:
#         return text
#     text = nltk.pos_tag(text)
#     return text
#
# def remove_stopwords(text):
#     if len(text) <= 0:
#         return text
#     pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
#     text = pattern.sub('', text)
#     return text


# # clean data, tokens and pos_tag
# train_x["clean_text"] = train_x["text"].map(lambda x: clean_text(x))
# train_x["tokens"] = train_x["clean_text"].map(lambda x: tokenize_text(x))
# train_x["pos_tags"] = train_x["tokens"].map(lambda x: pos_tag_text(x))
#
# test_x["clean_text"] = test_x["text"].map(lambda x: clean_text(x))
# test_x["tokens"] = test_x["clean_text"].map(lambda x: tokenize_text(x))
# test_x["pos_tags"] = test_x["tokens"].map(lambda x: pos_tag_text(x))
#
# train_x, train_y = train_x[["user", "date", "text", "title", "g_truth"]], train_y[['g_truth']]
# test_x, test_y = test_x[["user", "date", "text", "title", "g_truth"]], test_y[['g_truth']]
#
# # label encode the target variable (I think this is not necessary?)
# encoder = preprocessing.LabelEncoder()
# train_y = encoder.fit_transform(train_y)
# test_y = encoder.fit_transform(test_y)



if __name__ == '__main__':
    main()