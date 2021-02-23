# new prepare file for the new data released for erisk 2021.
# it already came separated in training and test data, so we decided to not merge it
# and use it as so, training for training, and testing for testing.
# g_truths are also separated for training and testing

import pickle, os
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
import contractions
from nltk import word_tokenize
import nltk
import re, itertools
from nltk.corpus import stopwords
import pandas as pd
from sklearn import model_selection, preprocessing

train_data_path = "/home/elena/Documentos/UNED/erisk/2021/data/erisk2021_training_data"
test_data_path = "/home/elena/Documentos/UNED/erisk/2021/data/erisk2021_test_data"

def main():

    # load data (already joined) and golden truth
    g_truth_train = load_golden_truth(train_data_path, "golden_truth.txt")
    g_truth_test = load_golden_truth(test_data_path, "golden_truth.txt")

    train_users = load_user_data(train_data_path, "data", g_truth_train)
    test_users = load_user_data(test_data_path, "data", g_truth_test)

    print(train_users[0:5])

    # convert to dataFrame, and choose which columns to conserve
    train_data_frame = pd.DataFrame(train_users)
    print(train_data_frame.columns)
    test_data_frame = pd.DataFrame(test_users)

    train_x, train_y = train_data_frame[["user", "date", "text", "title", "g_truth"]], train_data_frame[['g_truth']]
    test_x, test_y = test_data_frame[["user", "date", "text", "title", "g_truth"]], test_data_frame[['g_truth']]

    print(train_x["text"][0:50])


    # label encode the target variable (I think this is not necessary?)
    encoder = preprocessing.LabelEncoder()
    train_y = encoder.fit_transform(train_y)
    test_y = encoder.fit_transform(test_y)


    train_x["clean_text"] = train_x["text"].map(lambda x: clean_text(x))
    train_x["tokens"] = train_x["clean_text"].map(lambda x: tokenize_text(x))
    train_x["pos_tags"] = train_x["tokens"].map(lambda x: pos_tag_text(x))

    print(train_x["clean_text"][0:10])
    print(train_x["tokens"][0:10])
    print(train_x["pos_tags"][0:10])

    test_x["clean_text"] = test_x["text"].map(lambda x: clean_text(x))
    test_x["tokens"] = test_x["clean_text"].map(lambda x: tokenize_text(x))
    test_x["pos_tags"] = test_x["tokens"].map(lambda x: pos_tag_text(x))


    with open('data/pickles/train.x.pkl', 'wb') as train_x_file:
        pickle.dump(train_x, train_x_file)

    with open('data/pickles/train.y.pkl', 'wb') as train_y_file:
        pickle.dump(train_y, train_y_file)

    with open('data/pickles/test.x.pkl', 'wb') as test_x_file:
        pickle.dump(test_x, test_x_file)

    with open('data/pickles/test.y.pkl', 'wb') as test_y_file:
        pickle.dump(test_y, test_y_file)






    # print(users[0]["clean_text"], users[0]["tokens"])
    #







def load_golden_truth(path, filename):
    g_path = os.path.join(path, filename)
    g_truth = {line.split()[0]: int(line.split()[1]) for line in open(g_path)}
    return g_truth


# loads data already joined. for other methods, look erisk project
def load_user_data(dir_path, dir_name, g_truth):
    users = []

    path = os.path.join(dir_path, dir_name)

    for filename in os.listdir(path):

        user, file_extension = os.path.splitext(filename)

        tree = ET.parse(os.path.join(path, filename))
        root = tree.getroot()


        for writing in root.findall('WRITING'):
            user_writing = {"user": user, "g_truth": g_truth[user], "date": "", "text": "", "title": ""}

            title, text, date = "", "", ""
            if writing.find('TITLE') is not None:
                title = writing.find('TITLE').text
                if title is None:
                    title = ""
            if writing.find('TEXT') is not None:
                text = writing.find('TEXT').text
                if text is None:
                    text = ""
            if writing.find('DATE') is not None:
                date = writing.find('DATE').text
                if date is None:
                    date = ""

            user_writing['date'] = date
            user_writing['title'] = title
            user_writing['text'] = title + text
            users.append(user_writing)

        #users.append(user_writing)

    return users


# # loads data already joined. for other methods, look erisk project
# def load_user_data(dir_path, dir_name, g_truth):
#     users = []
#     path = os.path.join(dir_path, dir_name)
#
#     for filename in os.listdir(path):
#
#         user, file_extension = os.path.splitext(filename)
#
#         tree = ET.parse(os.path.join(path, filename))
#         root = tree.getroot()
#
#         user_writings = {"user": user, "g_truth": g_truth[user], "date": [], "text": [], "title": []}
#
#         for writing in root.findall('WRITING'):
#             title, text, date = "", "", ""
#             if writing.find('TITLE') is not None:
#                 title = writing.find('TITLE').text
#                 if title is None:
#                     title = ""
#             if writing.find('TEXT') is not None:
#                 text = writing.find('TEXT').text
#                 if text is None:
#                     text = ""
#             if writing.find('DATE') is not None:
#                 date = writing.find('DATE').text
#                 if date is None:
#                     date = ""
#
#             user_writings['date'].append(date)
#             user_writings['text'].append(title)
#             user_writings['title'].append(title)
#             user_writings['text'].append(text)
#
#         user_writings['text'] = ' .'.join(user_writings['text'])
#         user_writings['title'] = ' .'.join(user_writings['title'])
#         users.append(user_writings)
#
#     return users


def clean_text(text):
    if len(text) <= 0:
        return text
    text = text.strip()
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub(r' #[0-9]+;', '', text)
    text = re.sub(r"[^\w\d'\s]+", ' ', text)
    text = re.sub('  ', ' ', text)
    text = contractions.fix(text)  # NEW!!!!
    return text

def tokenize_text(text):
    if len(text) <= 0:
        return text
    text = text.lower()
    text = remove_stopwords(text)
    text = word_tokenize(text)
    return text

# text tiene que venir en tokens
def pos_tag_text(text):
    if len(text) <= 0:
        return text
    text = nltk.pos_tag(text)
    return text

def remove_stopwords(text):
    if len(text) <= 0:
        return text
    pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
    text = pattern.sub('', text)
    return text




if __name__ == '__main__':
    main()