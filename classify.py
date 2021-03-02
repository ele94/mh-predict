# classifies the test data with the trained classifier

# after train.py
# next: evaluate.py and/or evaluate_erisk.py

import pickle
from sklearn.svm import SVC
import pandas as pd

from utils import load_pickle
from utils import save_pickle
from utils import load_parameters

test_x_file = "test.feats.pkl"
test_y_file = "test.y.pkl"
classifier_file = "classifier.pkl"
resul_file = "test.resul.pkl"
score_file = "test.scores.pkl"

def main():

    classifier = load_pickle(classifier_file)
    test_x = load_pickle(test_x_file)

    #test_resul = classifier.classify_many(test_feats)
    if not load_parameters()["feats"] == "text":
        test_x = test_x.tocsc()
    predictions = classifier.predict(test_x)
    scores = predictions #todo fix this!!!

    save_pickle(resul_file, predictions)
    save_pickle(score_file, scores)


if __name__ == '__main__':
    main()