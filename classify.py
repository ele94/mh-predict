# classifies the test data with the trained classifier

# after train.py
# next: evaluate.py and/or evaluate_erisk.py

from utils import load_pickle
from utils import save_pickle
from utils import load_parameters
import filenames as fp


def main():

    params = load_parameters()
    feats = params["feats"]
    classifier_file = params["classifier"] + ".pkl"
    classifier_path = fp.get_classifier_path()
    resuls_path = fp.get_resuls_path()
    window_path = fp.get_window_path()
    test_feats_file = feats + "." + fp.test_feats_file

    classifier = load_pickle(classifier_path, classifier_file)
    test_feats = load_pickle(window_path, test_feats_file)

    #test_resul = classifier.classify_many(test_feats)
    if feats != "text":
        test_feats = test_feats.tocsc()
    predictions = classifier.predict(test_feats)
    scores = predictions #todo fix this!!!

    save_pickle(resuls_path, fp.resul_file, predictions)
    save_pickle(resuls_path, fp.score_file, scores)


if __name__ == '__main__':
    main()