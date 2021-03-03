# classifies the test data with the trained classifier

# after train.py
# next: evaluate.py and/or evaluate_erisk.py

import pickle
from sklearn.svm import SVC
import pandas as pd

from utils import load_pickle
from utils import save_pickle
from utils import load_parameters

test_feats_filename = "test.pkl"
#test_y_filename = "test.y.pkl"
#classifier_file_name = "classifier.pkl"
resul_filename = "test.resul.pkl"
score_filename = "test.scores.pkl"

def main():

    params = load_parameters()
    file_params = str(params["feats_window_size"]) + "." + params["feats"] + "."

    classifier_file = file_params + params["classifier"] + ".pkl"
    test_x_file = file_params + test_feats_filename
    resul_file = file_params + params["classifier"] + "." + resul_filename
    score_file = file_params + params["classifier"] + "." + score_filename

    classifier = load_pickle(classifier_file)
    test_x = load_pickle(test_x_file)

    #test_resul = classifier.classify_many(test_feats)
    if not params["feats"] == "text":
        test_x = test_x.tocsc()
    predictions = classifier.predict(test_x)
    scores = predictions #todo fix this!!!

    save_pickle(resul_file, predictions)
    save_pickle(score_file, scores)


if __name__ == '__main__':
    main()