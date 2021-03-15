from utils import load_parameters
from prepare import main as prepare
from windowfy import main as windowfy
from text_featurize import main as text_featurize
from tfidf_featurize import tfidf as tfidf_featurize
from tfidf_featurize import ngrams as ngram_featurize
from combine_features import main as combine_features
from train import main as train
from classify import main as classify
from evaluate import main as evaluate
from evaluate_erisk import main as eval_erisk
from utils import update_parameters
from utils import logger
import filenames as fp
import numpy as np
import os
import random as rn

last_experiment = {}

def test(params):

    logger("Starting experiment params {}".format(params))

    logger("Windowfying data")
    windowfy()
    if params["feats"] == "text" or params["feats"] == "combined":
        logger("Creating text features")
        text_featurize()
    if params["feats"] == "tfidf" or params["feats"] == "combined":
        logger("Creating tfidf features")
        tfidf_featurize()
    if params["feats"] == "ngram" or params["feats"] == "combined":
        logger("Creating tfidf ngram features")
        ngram_featurize()
    if params["feats"] == "combined":
        logger("Combining features")
        combine_features()

    logger("Training {}".format(params["classifier"]))
    train()

    logger("Classifying")
    classify()

    #print("Evaluating")
    #evaluate()
    logger("Evaluating erisk")
    eval_erisk()

    logger("Fin experiment {}".format(params))

params_history = []

def experiments():

    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(42)
    rn.seed(121412)

    params = load_parameters()

    feats_window_sizes = [3, 5, 10]
    eval_window_sizes = [1]
    feats = ["tfidf", "ngram", "text", "combined"]
    classifiers = ["svm", "xgboost", "linear-svm"]
    strategies = ["weights", "normal"]
    text_features = ["select", "all"]
    ranges_max = [(100, 100)] #[(100, 100), (100, -1), (-1, -1)]
    repeat_experiments = False

    experiments = []
    for train_range_max, test_range_max in ranges_max:
        for feats_window_size in feats_window_sizes:
            for feat in feats:
                for text_feature in text_features:
                    for classifier in classifiers:
                        for strategy in strategies:
                            for eval_window_size in eval_window_sizes:
                                params["text_features"] = text_feature
                                params["strategy"] = strategy
                                params["feats_window_size"] = feats_window_size
                                params["feats"] = feat
                                params["classifier"] = classifier
                                params["eval_window_size"] = eval_window_size
                                params["train_range_max"] = train_range_max
                                params["test_range_max"] = test_range_max
                                experiments.append(params.copy())

    for experiment in experiments:
        do_experiment(experiment.copy(), repeat_experiments)

    print("ENDED EXPERIMENTS")
    # want to test all combinations


def do_experiment(experiment_params, repeat_experiments):
    if experiment_params not in params_history or repeat_experiments:
        params_history.append(experiment_params.copy())
        update_parameters(experiment_params.copy())
        #test(experiment_params.copy())
        try:
            test(experiment_params.copy())
        except Exception as e:
            logger("failed params:{}".format(experiment_params))
            logger("Exception: {}".format(e))
    else:
        logger("Skipping duplicated params {}".format(experiment_params))

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    experiments()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
