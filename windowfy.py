# joins dictionary data into windows of configurable size and converts to pandas dataframe
# after prepare.py
# next: text_featurize.py and/or tfidf_featurize.py

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
from utils import load_pickle
from utils import save_pickle
from utils import load_parameters

train_filename = "clean.train.users.pkl"
test_filename = "clean.test.users.pkl"
train_x_filename = "train.x.pkl"
test_x_filename = "test.x.pkl"
train_y_filename = "train.y.pkl"
test_y_filename = "test.y.pkl"


def main():
    train_users = load_pickle(train_filename)
    test_users = load_pickle(test_filename)

    window_type = "count"  # (count, size or time)
    window_size = load_parameters()["feats_window_size"]

    train_window = windowfy_sliding(train_users, window_size)
    test_window = windowfy_sliding(test_users, window_size)

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

    save_pickle(train_x_filename, train_x)
    save_pickle(test_x_filename, test_x)
    save_pickle(train_y_filename, train_y)
    save_pickle(test_y_filename, test_y)


# def windowfy(data_users, window_size):
#     # new_data_dict = {}
#     #
#     # for index, row in data_users.iterrows():
#     #     if row["user"] in new_data_dict.keys():
#     #         new_data_dict[row["user"]].append(row.to_dict())
#     #     else:
#     #         new_data_dict[row["user"]] = [row.to_dict()]
#     #
#     # data_users = None
#     # print("Llega hasta aqui")
#
#     window_writings = []
#
#     for user, user_writings in data_users.items():
#         print(user)
#         print(len(user_writings))
#         # if(len(user_writings) > 1000):
#         #     continue
#         if window_size > 0:
#             writings_length = len(user_writings)
#             steps = writings_length / window_size
#         else:
#             steps = 1
#
#         # this is not sliding window lmao
#         # flatten = lambda l: [item for sublist in l for item in sublist]
#         for step in range(0, int(steps)):
#             print(step)
#
#             working_writings = user_writings[window_size * step:window_size * (step + 1)]
#             window_writing = join_window_elements(working_writings)
#             window_writing["user"] = working_writings[0]["user"]
#             window_writing["g_truth"] = working_writings[0]["g_truth"]
#
#             # date = []
#             # text = []
#             # title = []
#             # clean_text = []
#             # tokens = []
#             # pos_tags = []
#             # for writing in working_writings:
#             #     date.append(writing["date"])
#             #     text.append(writing["text"])
#             #     title.append(writing["title"])
#             #     clean_text.append(writing["clean_text"])
#             #     tokens.append(writing["tokens"])
#             #     pos_tags.append(writing["pos_tags"])
#             #
#             # text = '. '.join(text)
#             # title = '. '.join(title)
#             # clean_text = '. '.join(clean_text)
#             # tokens = flatten(tokens)
#             # pos_tags = flatten(pos_tags)
#             #
#             # window_writing = {"user": user, "g_truth": user_writings[0]["g_truth"], "date": date, "text": text,
#             #                   "title": title, "clean_text": clean_text, "tokens": tokens, "pos_tags": pos_tags}
#             window_writings.append(window_writing)
#
#     return window_writings


# todo check
def windowfy_sliding(users, window_size):
    users_windows = []
    for user, writings in users.items():
        range_max = len(writings)  # TODO parametrizar esto con la opcion de poner un maximo distinto
        for i in range(0, range_max - 1):  # TODO parametrizar esto
            if len(writings) <= (i + window_size):
                window = writings[i:range_max]  # TODO comprobar este range_max
            else:
                window = writings[i:i + window_size]

            if len(window) == 0:
                print("Window: {}, i: {}, len(writings): {}".format(window, i, len(writings)))
            window = join_window_elements(window)
            users_windows.append(window)

    return users_windows


def join_all_elements(users):
    joined_writings = []
    for user, writings in users.items():
        joined_writings.append(join_window_elements(writings))

    return joined_writings


def join_window_elements(window: list) -> dict:
    joint_window = {}
    flatten = lambda l: [item for sublist in l for item in sublist]

    for key in window[0].keys():
        key_list = [message[key] for message in window]
        if type(key_list[0]) is list:
            joint_window[key] = flatten(key_list)
        elif key == 'user':
            joint_window[key] = key_list[0]
        elif key == 'g_truth':
            joint_window[key] = key_list[0]
        elif key == 'date':
            joint_window[key] = key_list
        else:
            joint_window[key] = ' .'.join(key_list)

    return joint_window



if __name__ == '__main__':
    main()
