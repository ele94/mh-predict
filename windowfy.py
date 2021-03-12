# joins dictionary data into windows of configurable size and converts to pandas dataframe
# after prepare.py
# next: text_featurize.py and/or tfidf_featurize.py

import pandas as pd
from sklearn import preprocessing
from utils import load_pickle
from utils import save_pickle
from utils import remove_pickle
from utils import load_parameters
from utils import logger
import filenames as fp
import numpy as np



def main():

    params = load_parameters()

    window_path = fp.get_window_path()

    remove_pickle(fp.pickles_path, fp.train_x_filename)
    remove_pickle(fp.pickles_path, fp.test_x_filename)
    remove_pickle(fp.pickles_path, fp.train_y_filename)
    remove_pickle(fp.pickles_path, fp.test_y_filename)

    train_users = load_pickle(fp.pickles_path, fp.train_filename)
    test_users = load_pickle(fp.pickles_path, fp.test_filename)


    window_type = "count"  # (count, size or time)
    window_size = params["feats_window_size"]
    weights_window_size = params["weights_window_size"]
    train_range_max = params["train_range_max"]
    test_range_max = params["test_range_max"]

    train_window = windowfy_sliding_training(train_users, window_size, train_range_max)
    test_window = windowfy_sliding_testing(test_users, window_size, test_range_max)

    train_window_frame = pd.DataFrame(train_window)
    test_window_frame = pd.DataFrame(test_window)

    train_x, train_y = train_window_frame[
                           ["user", "date", "text", "title", "g_truth", "clean_text", "tokens", "pos_tags", "stems"]], \
                       train_window_frame[['g_truth']]
    test_x, test_y = test_window_frame[
                         ["user", "date", "text", "title", "g_truth", "clean_text", "tokens", "pos_tags", "stems"]], \
                     test_window_frame[['g_truth']]

    encoder = preprocessing.LabelEncoder()
    train_y = encoder.fit_transform(train_y)
    test_y = encoder.fit_transform(test_y)

    window_sample_weights = get_window_sample_weights(train_users, weights_window_size, window_size, train_range_max)

    save_pickle(window_path, fp.train_weights_file, window_sample_weights)
    save_pickle(window_path, fp.train_x_filename, train_x)
    save_pickle(window_path, fp.test_x_filename, test_x)
    save_pickle(window_path, fp.train_y_filename, train_y)
    save_pickle(window_path, fp.test_y_filename, test_y)


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

def windowfy_sliding_training(users, window_size, param_range_max=-1):
    users_windows = []
    for user, user_writings in users.items():
        count = 0
        if param_range_max < 0 or param_range_max > len(user_writings):
            range_max = len(user_writings)
            writings = user_writings.copy()
        else:
            range_max = param_range_max
            writings = user_writings.copy()[:range_max]
        for i in range(0, range_max):  # TODO parametrizar esto
            #if i < window_size and len(writings) > (i + 1):
            #    window = writings[:i + 1]  # rellenamos mientras "no nos llegan los demas mensajes"
            if len(writings) < (i + window_size):
                window = writings[i:range_max]  # TODO comprobar este range_max
                continue
            else:
                window = writings[i:i + window_size]

            if len(window) == 0:
                pass
                #print("Window: {}, i: {}, len(writings): {}".format(window, i, len(writings)))

            joined_window = join_window_elements(window)
            users_windows.append(joined_window)
            count += 1

    logger("Length of train data after windowfying: {}".format(len(users_windows)))
    return users_windows


def windowfy_sliding_testing(users, window_size, param_range_max=-1):
    users_windows = []
    for user, writings in users.items():
        count = 0
        if param_range_max < 0 or param_range_max > len(writings):
            range_max = len(writings)
        else:
            range_max = param_range_max
        for i in range(0, range_max):  # TODO parametrizar esto
            if i < window_size and len(writings) > (i+1):
                window = writings[:i+1] # rellenamos mientras "no nos llegan los demas mensajes" # todo cambiar esto
            elif len(writings) < (i + window_size):
                #window = writings[i:range_max]  # TODO comprobar este range_max
                window = []
            else:
                window = writings[i:i + window_size]

            if len(window) == 0:
                pass
                #print("Window: {}, i: {}, len(writings): {}".format(window, i, len(writings)))
            else:
                joined_window = join_window_elements(window)
                users_windows.append(joined_window)
                count += 1

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
        elif key == 'sequence':
            joint_window[key] = key_list[-1]
        else:
            joint_window[key] = ' .'.join(key_list)

    return joint_window


def get_window_sample_weights(users, weights_window_size, feats_window_size, param_range_max=-1):

    flatten = lambda t: [item for sublist in t for item in sublist]

    my_range = int(weights_window_size/feats_window_size)
    positive_samples_weights = [x / (1.0*my_range) for x in range((1*my_range), (2*my_range), 1)]
    positive_samples_weights.sort(reverse=True)
    negative_samples_weights = np.ones(1)

    users_sample_weights = []

    for user, user_writings in users.items():

        if param_range_max < 0 or param_range_max > len(user_writings):
            range_max = len(user_writings)
        else:
            range_max = param_range_max

        if range_max < feats_window_size:
            logger("Skipped user {} because writings size {} and window size {}".format(user, len(user_writings), feats_window_size))
            continue
        else:
            if user_writings[0]['g_truth'] == 1:
                user_samples_weights = positive_samples_weights.copy()
            else:
                user_samples_weights = negative_samples_weights.copy()

            new_max_size = (range_max - feats_window_size) +1
            if len(user_samples_weights) < new_max_size:
                end_size = new_max_size - len(user_samples_weights)
                user_samples_weights = list(np.pad(user_samples_weights, (0, end_size), 'minimum'))
            else:
                user_samples_weights = list(np.resize(user_samples_weights, new_max_size))
            users_sample_weights.append(list(user_samples_weights))

    users_sample_weights = flatten(users_sample_weights)
    logger("Final sample weights length: {}".format(len(users_sample_weights)))
    return users_sample_weights

if __name__ == '__main__':
    main()
