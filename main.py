# This is a sample Python script.

# Press Mayús+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from utils import load_parameters
from prepare import main as prepare
from windowfy import main as windowfy
from text_featurize import main as text_featurize
from tfidf_featurize import main as tfidf_featurize
from combine_features import main as combine_features
from sample_weights import main as sample_weights
from select_feats import main as select_feats
from train import main as train
from classify import main as classify
from evaluate import main as evaluate
from evaluate_erisk import main as eval_erisk
from utils import update_parameters
from utils import check_pickle
import filenames as fp

last_experiment = {}

def test(params):

    print("Starting experiment params {}".format(params))

    if not check_pickle(fp.get_feats_path(), fp.train_df_feats_filename):
        print("Windowfying data")
        windowfy()
        print("Creating text features")
        text_featurize()
        print("Creating tfidf features")
        tfidf_featurize()
        print("Combining features")
        combine_features()

    classifier_file = params["classifier"] + ".pkl"
    if not check_pickle(fp.get_classifier_path(), classifier_file):
        print("Training {}".format(params["classifier"]))
        train()

    if not check_pickle(fp.get_resuls_path(), fp.resul_file):
        print("Classifying")
        classify()

    #print("Evaluating")
    #evaluate()
    print("Evaluating erisk")
    eval_erisk()

    print("Fin experiment {}".format(params))

params_history = []

def experiments():

    params = load_parameters()

    feats_window_sizes = [1, 5, 10, 20]
    eval_window_sizes = [1]
    feats = ["text", "tfidf", "combined"]
    classifiers = ["xgboost", "forest", "svm", "linear_svm"]
    strategies = ["balanced", "weights"]

    experiments = []

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
