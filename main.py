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
from utils import write_experiment
from utils import write_csv
from statistics import mean
from experiment_settings import get_experiment_settings


last_experiment = {}

def test(params, random=False):

    logger("Starting experiment params {}".format(params))
    #
    # if random:


    logger("Windowfying data")
    windowfy()
    if params["feats"] == "text" or params["feats"] == "combined":
        logger("Creating text features")
        text_featurize()
    if params["feats"] == "tfidf" or params["feats"] == "combined":
        logger("Creating tfidf features")
        if params["tfidf_ngrams"]:
            ngram_featurize()
        else:
            tfidf_featurize()
    if params["feats"] == "combined":
        logger("Combining features")
        combine_features()

    logger("Training {}".format(params["classifier"]))
    train()

    # logger("Classifying")
    # classify()
    #
    # #print("Evaluating")
    # #evaluate()
    # logger("Evaluating erisk")
    # eval = eval_erisk()
    #
    # logger("Fin experiment {}".format(params))
    # return eval

params_history = []

def experiments():

    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(42)
    rn.seed(121412)

    cross_validation = False
    cross_validation_n = 1
    repeat_experiments = False

    experiments = get_experiment_settings()

    if cross_validation:
        for experiment in experiments:
            all_eval = {'speed': [], 'latency_weighted_f1': []}
            for validation in range(0, cross_validation_n):
                eval = do_experiment(experiment.copy(), repeat_experiments, random=True)
                all_eval["speed"].append(eval["speed"])
                all_eval["latency_weighted_f1"].append(eval("latency_weighted_f1"))
            mean_eval = get_mean(all_eval)
            write_csv(mean_eval)

    else:
        for experiment in experiments:
            eval = do_experiment(experiment.copy(), repeat_experiments)
            if eval is not None:
                write_csv(eval)

    print("ENDED EXPERIMENTS")
    # want to test all combinations


def do_experiment(experiment_params, repeat_experiments, random=False):
    if experiment_params not in params_history or repeat_experiments:
        params_history.append(experiment_params.copy())
        update_parameters(experiment_params.copy())
        #eval = test(experiment_params.copy(), random)
        try:
            eval = test(experiment_params.copy(), random)
        except Exception as e:
            logger("failed params:{}".format(experiment_params))
            logger("Exception: {}".format(e))
            eval = None
    else:
        logger("Skipping duplicated params {}".format(experiment_params))
        eval = None

    return eval


def get_mean(evals):
    new_evals = {}
    for key, values in evals.items():
        new_evals[key] = mean(values)

    return new_evals



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    experiments()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
