from utils import load_parameters
from prepare import main as prepare
from windowfy import main as windowfy
from text_featurize import main as text_featurize
from tfidf_featurize import main as tfidf_featurize
from combine_features import main as combine_features
from train import main as train
from classify import main as classify
from evaluate import main as evaluate
from evaluate_erisk import main as eval_erisk
from utils import update_parameters
from utils import check_pickle
from utils import logger
import filenames as fp

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

    params = load_parameters()

    feats_window_sizes = [10]
    eval_window_sizes = [1]
    feats = ["text", "tfidf", "combined"]
    classifiers = ["svm"]
    strategies = ["weights"]
    ranges_max = [(100, -1)] #[(100, 100), (100, -1), (-1, -1)]

    experiments = []
    for train_range_max, test_range_max in ranges_max:
        for feats_window_size in feats_window_sizes:
            for feat in feats:
                for classifier in classifiers:
                    for strategy in strategies:
                        for eval_window_size in eval_window_sizes:
                            params["strategy"] = strategy
                            params["feats_window_size"] = feats_window_size
                            params["feats"] = feat
                            params["classifier"] = classifier
                            params["eval_window_size"] = eval_window_size
                            params["train_range_max"] = train_range_max
                            params["test_range_max"] = test_range_max
                            experiments.append(params.copy())

    for experiment in experiments:
        do_experiment(experiment.copy())

    print("ENDED EXPERIMENTS")
    # want to test all combinations


def do_experiment(experiment_params):
    if experiment_params not in params_history:
        params_history.append(experiment_params.copy())
        update_parameters(experiment_params.copy())
        test(experiment_params.copy())
        # try:
        #     test(experiment_params.copy())
        # except Exception as e:
        #     logger("failed params:{}".format(experiment_params))
        #     logger("Exception: {}".format(e))
    else:
        logger("Skipping duplicated params {}".format(experiment_params))

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    experiments()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
