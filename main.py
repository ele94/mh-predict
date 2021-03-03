# This is a sample Python script.

# Press Mayús+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from utils import load_parameters
from prepare import main as prepare
from windowfy import main as windowfy
from text_featurize import main as text_featurize
from tfidf_featurize import main as tfidf_featurize
from combine_features import main as combine_features
from select_feats import main as select_feats
from train import main as train
from classify import main as classify
from evaluate import main as evaluate
from evaluate_erisk import main as eval_erisk
from utils import update_parameters
from utils import check_pickle

last_experiment = {}

def test(params):

    # if last_params["feats_window_size"] != params["feats_window_size"]:
    #     experiment = ["windowfy", "featurize", "select_feats", "train",
    #                   "classify", "evaluate", "eval_erisk"]
    # elif last_params["feats"] != params["feats"]:
    #     experiment = ["select_feats", "train",
    #                   "classify", "evaluate", "eval_erisk"]
    # elif last_params["classifier"] != params["classifier"]:
    #     experiment = ["train",
    #                   "classify", "evaluate", "eval_erisk"]
    # elif last_params["eval_window_size"] != params["eval_window_size"]:
    #     experiment = ["eval_erisk"]
    # else:
    #     experiment = None
    #
    # if experiment is None:
    #     experiment = ["windowfy", "featurize", "select_feats", "train",
    #                   "classify", "evaluate", "eval_erisk"]

    last_params = params.copy()

    print("Starting experiment params {}".format(params))
    #experiment = ["windowfy", "text_featurize", "tfidf_featurize", "combine_features", "select_feats", "train", "classify", "evaluate", "eval_erisk"]


    # from the beginning
    #experiment = ["prepare", "windowfy", "text_featurize", "tfidf_featurize", "combine_features", "select_feats", "train", "classify", "evaluate", "eval_erisk"]

    # combining features from clean data
    #experiment = ["windowfy", "text_featurize", "tfidf_featurize", "combine_features", "select_feats", "train", "classify", "evaluate", "eval_erisk"]

    # text features from clean data
    #experiment = ["windowfy","text_featurize","select_feats", "train", "classify", "evaluate", "eval_erisk"]

    #tfidf features from clean data
    #experiment = ["windowfy","tfidf_featurize","select_feats", "train", "classify", "evaluate", "eval_erisk"]

    # select feats from window already selected
    #experiment = ["select_feats", "train", "classify", "evaluate", "eval_erisk"]

    # experiment with erisk decision params
    #experiment = ["eval_erisk"]


    # if "prepare" in experiment:
    #     print("Preparing data")
    #     prepare()

    window_file = str(params["feats_window_size"]) + ".text.train.pkl"
    if not check_pickle(window_file):
        print("Windowfying data")
        windowfy()
        print("Creating text features")
        text_featurize()
        print("Creating tfidf features")
        tfidf_featurize()
        print("Combining features")
        combine_features()

    classifier_file = str(params["feats_window_size"]) + "." + params["feats"] + "." + params["classifier"] + ".pkl"
    if not check_pickle(classifier_file):
        print("Training")
        train()

    resuls_file = str(params["feats_window_size"]) + "." + params["feats"] + "." + params["classifier"] + ".test.resul.pkl"
    if not check_pickle(resuls_file):
        print("Classifying")
        classify()

    #print("Evaluating")
    #evaluate()
    print("Evaluating erisk")
    eval_erisk()

    # if "windowfy" in experiment:
    #     print("Windowfying data")
    #     windowfy()
    # if "featurize" in experiment:
    #     print("Creating text features")
    #     text_featurize()
    #     print("Creating tfidf features")
    #     tfidf_featurize()
    #     print("Combining features")
    #     combine_features()
    # if "select_feats" in experiment:
    #     print("Selecting features")
    #     select_feats()
    # if "train" in experiment:
    #     print("Training")
    #     train()
    # if "classify" in experiment:
    #     print("Classifying")
    #     classify()
    # if "evaluate" in experiment:
    #     print("Evaluating")
    #     evaluate()
    # if "eval_erisk" in experiment:
    #     print("Evaluating erisk")
    #     eval_erisk()
    print("Fin experiment {}".format(params))
    return last_params

params_history = []

def experiments():

    params = load_parameters()

    feats_window_sizes = [1, 10, 20, 50, 100]
    eval_window_sizes = [1, 3, 5]
    #max_features = [1000, 2000, 5000]
    feats = ["text", "tfidf", "combined"]
    classifiers = ["svm", "linear_svm", "forest", "xgboost"]


    last_params = params.copy()
    last_params["feats_window_size"] = 100
    params["feats_window_size"] = 1
    params["eval_window_size"] = 1
    params["feats"] = "text"
    params["classifier"] = "svm"
    update_parameters(params)

    experiments = []

    for feats_window_size in feats_window_sizes:
        params["feats_window_size"] = feats_window_size
        for feat in feats:
            params["feats"] = feat
            for classifier in classifiers:
                params["classifier"] = classifier
                for eval_window_size in eval_window_sizes:
                    params["eval_window_size"] = eval_window_size
                    experiments.append(params.copy())
                experiments.append(params.copy())
            experiments.append(params.copy())
        experiments.append(params.copy())

    for experiment in experiments:
        do_experiment(experiment.copy())

    print("ENDED EXPERIMENTS")
    # want to test all combinations


def do_experiment(experiment_params):
    if experiment_params not in params_history:
        params_history.append(experiment_params.copy())
        update_parameters(experiment_params.copy())
        try:
            test(experiment_params.copy())
        except Exception as e:
            print("failed params:{}".format(experiment_params))
            print("Exception: {}".format(e))
    else:
        print("Skipping duplicated params {}".format(experiment_params))

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    experiments()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
