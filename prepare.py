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

def main():

    # load data (already joined) and golden truth
    g_truth = load_golden_truth("data", "2019_golden_truth.txt")
    users = load_user_data("data/user_data", g_truth)

    # clean data
    for user in users:
        user["clean_text"] = clean_text(user["text"])

    # tokenize, stem and pos??? de momento solo tokenize
    for user in users:
        user["tokens"] = tokenize_text(user["clean_text"])

    # pos tambien
    for user in users:
        user["pos_tags"] = pos_tag_text(user["tokens"])


    print(users[0])

    # convert to dataFrame, and choose which columns to conserve
    data_frame = pd.DataFrame(users)
    print(data_frame.columns)

    # separate data in train and test
    train_x, test_x, train_y, test_y = train_test_split(data_frame[["user", "date", "text", "title", "clean_text",
                                                                    "tokens", "pos_tags"]], data_frame['g_truth'], test_size=0.33)
    # label encode the target variable (I think this is not necessary?)
    encoder = preprocessing.LabelEncoder()
    train_y = encoder.fit_transform(train_y)
    test_y = encoder.fit_transform(test_y)

    # save train and test set
    with open('data/pickles/users.df.pkl', 'wb') as users_df_file:
        pickle.dump(data_frame, users_df_file)

    with open('data/pickles/train.x.pkl', 'wb') as train_x_file:
        pickle.dump(train_x, train_x_file)

    with open('data/pickles/train.y.pkl', 'wb') as train_y_file:
        pickle.dump(train_y, train_y_file)

    with open('data/pickles/test.x.pkl', 'wb') as test_x_file:
        pickle.dump(test_x, test_x_file)

    with open('data/pickles/test.y.pkl', 'wb') as test_y_file:
        pickle.dump(test_y, test_y_file)

    with open('data/pickles/clean.users.pkl', 'wb') as clean_users_file:
        pickle.dump(users, clean_users_file)

    # with open('data/pickles/train.users.pkl', 'wb') as train_users_file:
    #     pickle.dump(train_users, train_users_file)
    #
    # with open('data/pickles/test.users.pkl', 'wb') as test_users_file:
    #     pickle.dump(test_users, test_users_file)







    # print(users[0]["clean_text"], users[0]["tokens"])
    #









def load_golden_truth(path, filename):
    g_path = os.path.join(path, filename)
    g_truth = {line.split()[0]: int(line.split()[1]) for line in open(g_path)}
    return g_truth


# loads data already joined. for other methods, look erisk project
def load_user_data(path, g_truth):
    users = []

    for filename in os.listdir(path):

        user, file_extension = os.path.splitext(filename)

        tree = ET.parse(os.path.join(path, filename))
        root = tree.getroot()

        user_writings = {"user": user, "g_truth": g_truth[user], "date": [], "text": [], "title": []}

        for writing in root.findall('WRITING'):
            title, text, date = "", "", ""
            if writing.find('TITLE') is not None:
                title = writing.find('TITLE').text
            if writing.find('TEXT') is not None:
                text = writing.find('TEXT').text
            if writing.find('DATE') is not None:
                date = writing.find('DATE').text

            user_writings['date'].append(date)
            user_writings['title'].append(title)
            user_writings['text'].append(title)
            user_writings['text'].append(text)

        user_writings['text'] = '.'.join(user_writings['text'])
        users.append(user_writings)

    return users


def clean_text(text):
    text = text.strip()
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub(r' #[0-9]+;', '', text)
    text = re.sub(r"[^\w\d'\s]+", ' ', text)
    text = re.sub('  ', ' ', text)
    text = contractions.fix(text)  # NEW!!!!
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