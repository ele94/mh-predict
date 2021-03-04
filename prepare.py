# prepare adapted to erisk 2021, where train and test came separated
# cleaning done to the data in dictionary form, and saved as dictionary

# this is the beginning
# next: windowfy,py

from abc import abstractmethod
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
from utils import load_pickle
from utils import save_pickle
import filenames as fp

train_data_path = "/home/elena/Documentos/UNED/erisk/2021/data/erisk2021_training_data"
test_data_path = "/home/elena/Documentos/UNED/erisk/2021/data/erisk2021_test_data"

def main():

    g_truth_train = load_golden_truth(train_data_path, "golden_truth.txt")
    g_truth_test = load_golden_truth(test_data_path, "golden_truth.txt")

    train_users = load_user_data(train_data_path, "data", g_truth_train)
    test_users = load_user_data(test_data_path, "data", g_truth_test)

    train_users = preprocess_data(train_users)
    test_users = preprocess_data(test_users)

    print(train_users[list(train_users.keys())[0]])

    save_pickle(fp.pickles_path, fp.train_filename, train_users)
    save_pickle(fp.pickles_path, fp.test_filename, test_users)





# cleans text, creates tokens and applies pos_tags
def preprocess_data(users):
    preproc_users = {}
    for user, writings in users.items():
        preproc_writings = []
        for writing in writings:
            writing["clean_text"] = clean_text(writing["text"])
            if len(writing["clean_text"]) == 0:
                print("Text less than 0: ", writing["text"])
            writing["tokens"] = tokenize_text(writing["clean_text"])
            writing["pos_tags"] = pos_tag_text(writing["tokens"])
            preproc_writings.append(writing)

        preproc_users[user] = preproc_writings


    return preproc_users



def load_golden_truth(path, filename):
    g_path = os.path.join(path, filename)
    g_truth = {line.split()[0]: int(line.split()[1]) for line in open(g_path)}
    return g_truth


# loads data already joined. for other methods, look erisk project
def load_user_data(dir_path, dir_name, g_truth):
    users = {}

    path = os.path.join(dir_path, dir_name)

    for filename in os.listdir(path):

        user, file_extension = os.path.splitext(filename)

        tree = ET.parse(os.path.join(path, filename))
        root = tree.getroot()

        user_writings = []

        for writing in root.findall('WRITING'):
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

            if len(title) > 0:
                user_writing = {"user": user, "g_truth": g_truth[user], "date": date, "text": title + ". " + text, "title": title}
            else:
                user_writing = {"user": user, "g_truth": g_truth[user], "date": date, "text": text, "title": title}
            user_writings.append(user_writing)

        users[user] = user_writings

    return users


def clean_text(old_text):

    text = old_text.strip()
    text = re.sub(r'(http|ftp|https):\/\/([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:\/~+#-]*[\w@?^=%&\/~+#-])?', 'URL', text, flags=re.MULTILINE)
    text = re.sub(r' #[0-9]+;', '', text)
    text = re.sub(r"[^\w\d'\s]+", ' ', text)
    text = re.sub('  ', ' ', text)
    text = contractions.fix(text)  # NEW!!!!
    if len(text) <= 0:
        print("Text less than 0", "old text:" , old_text, "new text:",  text)
        text = "0"
    return text

def tokenize_text(text):
    text = text.lower()
    text = remove_stopwords(text)
    text = word_tokenize(text)
    return text

# text tiene que venir en tokens
def pos_tag_text(text):
    text = nltk.pos_tag(text)
    return text

def remove_stopwords(text):
    pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
    text = pattern.sub('', text)
    return text




if __name__ == '__main__':
    main()