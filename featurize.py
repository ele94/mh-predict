# aqui es donde tenemos que determinar el tamaño de las ventanas
from nltk.stem import PorterStemmer
from nltk.sentiment.util import *
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pickle
import re



def main():

    with open('data/pickles/test.users.pkl', 'rb') as test_users_file:
        test_users = pickle.load(test_users_file)
    with open('data/pickles/train.users.pkl', 'rb') as train_users_file:
        train_users = pickle.load(train_users_file)

    feat_extractor = FeaturesExtractor()

    train_feats = []
    test_feats = []
    test_labels = []

    for user in train_users:
        feats = feat_extractor.get_features(user)
        train_feats.append((feats, user["g_truth"]))

    for user in test_users:
        feats = feat_extractor.get_features(user)
        test_feats.append(feats)
        test_labels.append(user["g_truth"])

    with open('data/pickles/test.feats.pkl', 'wb') as test_feats_file:
        pickle.dump(test_feats, test_feats_file)

    with open('data/pickles/train.feats.pkl', 'wb') as train_feats_file:
        pickle.dump(train_feats, train_feats_file)

    with open('data/pickles/test.labels.pkl', 'wb') as test_labels_file:
        pickle.dump(test_labels, test_labels_file)

    print(test_feats[0])




#### FEATURE EXTRACTOR #######
# TODO CORREGIR

#clase og de la que van a derivar todas
# nos vamos a deshacer de normalize porque es obvio que hay que normalizarlos siempre
# TODO añadir codigo para normalizarlos todos!!!
# selected features ahora indica CATEGORIAS
# TODO crear nueva lista de all_feats
class FeaturesExtractor:
    def __init__(self):
        super().__init__()

    # un solo diccionario (sin el usuario como key ni nada!)
    def get_features(self, writing: dict):
        features = {}
        #init extractors
        form_extractor = FormExtractor()
        content_extractor = ContentExtractor()
        pronouns_extractor = PronounsExtractor()
        time_extractor = TimeExtractor()
        sent_extractor = SentimentExtractor()
        freq_extractor = FreqExtractor()
        lex_extractor = LexiconExtractor()

        features.update(form_extractor.get_features(writing))
        features.update(content_extractor.get_features(writing))
        features.update(pronouns_extractor.get_features(writing))
        features.update(time_extractor.get_features(writing))
        features.update(sent_extractor.get_features(writing))
        features.update(freq_extractor.get_features(writing))
        features.update(lex_extractor.get_features(writing))
        return features

    # diccionario de diccionarios! uno para cada usuario (key: user - item: features)
    # devuelve un diccionario con usuario - features (dict tambien)
    def get_group_features(self, users_writings):
        user_features = {}  # usuario - features
        for user, writing in users_writings.items():
            feats = self.get_features(writing)
            user_features[user] = feats

        return user_features

    # helper methods
    # devuelve la longitud del texto incluyendo titulo y texto
    def get_text_length(self, writing):
        text = writing[r'text']
        return len(text)


# Extracts features related to the FORM of the text (num of characters, len, type of characters, etc.)
class FormExtractor(FeaturesExtractor):

    # ''public'' method, gets all features realted to FormExtractor
    def get_features(self, writing):
        feats= {}
        feats.update(self.get_text_features(writing))
        feats.update(self.get_emoji_features(writing))
        return feats

    def get_text_features(self, writing):
        feats = {}
        text_len = self.get_text_length(writing)
        feats['text_len'] = text_len
        feats['text_num_words'] = self.count_words(writing['clean_text'])
        feats['text_num_chars'] = len(writing['clean_text'])
        feats['text_punct'] = len(re.findall(r'\,+|\.+', writing['text']))/text_len
        feats['text_question'] = len(re.findall(r'\?', writing['text']))/text_len
        feats['text_exclamation'] = len(re.findall(r'\!', writing['text']))/text_len
        return feats

    def get_emoji_features(self, writing):
        feats = {}
        text_len = self.get_text_length(writing)
        feats['smilies'] = len(re.findall(r'\:\)+|\(+\:', writing['text']))/text_len
        feats['sad_faces'] = len(re.findall(r'\:\(+|\)+\:', writing['text']))/text_len
        return feats

    # HELPER METHODS
    def count_words(self, text):
        res = len(re.findall(r'\w+', text))
        return res

# Extracts features related to the CONTENT of the text (pos tagging, words, etc.)
class ContentExtractor(FeaturesExtractor):

    def get_features(self, writing):
        features = {}

        return features
# tODO

# Extracts features related to the use of pronouns
class PronounsExtractor(FeaturesExtractor):

    # 'public' method, returns all features for this PronounsExtractor
    def get_features(self, writing):
        features = {}
        text_len = self.get_text_length(writing)
        features['first_prons'] = self.get_first_prons(writing)/text_len
        return features

    ## helper methods ##

    # devuelve el numero de pronombres de primera persona que hay en un writing
    def get_first_prons(self, writing):
        reg = r'\bI\b|\bme\b|\bmine\b|\bmy\b|\bmyself\b'
        text = writing[r'clean_text']
        first_prons = len(re.findall(reg, text))

        return first_prons

# Extracts features related to the time when the texts were produced
# TODO
class TimeExtractor(FeaturesExtractor):

    def get_features(self, writing):
        features = {}
        return features
        # TODO

# Extracts features related to the sentiment analysis
class SentimentExtractor(FeaturesExtractor):

    def __init__(self):
        self.sid = SentimentIntensityAnalyzer()

    # ''public'' method, gets all SentimentExtractor features
    def get_features(self, writing):
        features = {}
        text_len = self.get_text_length(writing)
        features['sentiment'] = self.get_sentiment(writing)/text_len
        return features

    ## feature extractor helper methods ##
    # devuelve el sentiment compound como un numero entre 0 y 1 con dos decimales (simplificar)
    def get_sentiment(self, writing):
        text = writing[r'clean_text']
        polarities = self.sid.polarity_scores(text)
        compound = round(polarities['compound'], 2)
        return compound

# Extracts features related to the frequency of certain words
# TODO
# (if-df o como se llame por ejemplo)
class FreqExtractor(FeaturesExtractor):

    def get_features(self, writing):
        features = {}
        return features
        # TODO

# Extracts features related to the use of words related to certain lexicons
# ex: depression, nssi, anxiety, etc.
# TODO complet with other lexicons
class LexiconExtractor(FeaturesExtractor):
    nssi_corpus_path: str = "data/nssicorpus.txt"

    def __init__(self):
        # leemos corpus
        with open(self.nssi_corpus_path, 'r') as file:
            self.nssi_corpus = file.read().replace('*', '')
        self.nssi_corpus = self.nssi_corpus.split('\n')
        self.nssi_corpus.remove('')

    # ''public'' method that gets all features from LexiconExtractor
    def get_features(self,writing):
        features = {}
        text_len = self.get_text_length(writing)
        features['nssi_words'] = self.get_nssi_words(writing)/text_len
        return features

    ## helper feature extractor methods ##

    # devuelve el numero de palabras nssi que hay en un writing
    def get_nssi_words(self, writing):
        text = ' '.join(writing[r'tokens'])
        return sum((text.count(word) for word in self.nssi_corpus))








if __name__ == '__main__':
    main()