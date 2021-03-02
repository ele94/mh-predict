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

def test(params, experiment=None):
    # Use a breakpoint in the code line below to debug your script.
    if experiment is None:
        experiment = ["windowfy", "featurize", "select_feats", "train",
                      "classify", "evaluate", "eval_erisk"]

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
    if "windowfy" in experiment:
        print("Windowfying data")
        windowfy()
    if "featurize" in experiment:
        if "text" in params["feats"]:
            print("Creating text features")
            text_featurize()
        elif "tfidf" in params["feats"]:
            print("Creating tfidf features")
            tfidf_featurize()
        elif "combined" in params["feats"]:
            print("Creating text features")
            text_featurize()
            print("Creating tfidf features")
            tfidf_featurize()
            print("Combining features")
            combine_features()
    if "select_feats" in experiment:
        print("Selecting features")
        select_feats()
    if "train" in experiment:
        print("Training")
        train()
    if "classify" in experiment:
        print("Classifying")
        classify()
    if "evaluate" in experiment:
        print("Evaluating")
        evaluate()
    if "eval_erisk" in experiment:
        print("Evaluating erisk")
        eval_erisk()
    print("Fin experiment")



def experiments():

    params = load_parameters()

    feats_window_sizes = [1, 10, 20, 50, 100, 1000]
    eval_window_sizes = [1, 3, 5, 10, 100]
    max_features = [1000, 2000, 5000]
    feats = ["text", "tfidf", "combined"]
    classifiers = ["svm", "linear_svm", "forest", "xgboost", "bayes"]

    params_history = []
    params_history.append({'classifier': 'svm', 'eval_window_size': 100, 'feats': 'text', 'feats_window_size': 1, 'max_features': 1000, 'strategy': 'balanced'}
)
    params_history.append({'classifier': 'svm', 'eval_window_size': 10, 'feats': 'text', 'feats_window_size': 1, 'max_features': 1000, 'strategy': 'balanced'}
)

    params_history.append({'classifier': 'svm', 'eval_window_size': 5, 'feats': 'text', 'feats_window_size': 1, 'max_features': 1000, 'strategy': 'balanced'}
)

    params_history.append({'classifier': 'svm', 'eval_window_size': 3, 'feats': 'text', 'feats_window_size': 1, 'max_features': 1000, 'strategy': 'balanced'}
)

    params_history.append({'classifier': 'svm', 'eval_window_size': 1, 'feats': 'text', 'feats_window_size': 1, 'max_features': 1000, 'strategy': 'balanced'}
)


    def do_experiment(local_experiment):
        if params not in params_history:
            params_history.append(params)
            update_parameters(params)
            print("Experiment {}, params:{}".format(local_experiment, params))
            try:
                test(params, local_experiment)
            except:
                print("failed experiment {}, params:{}".format(local_experiment, params))
        else:
            print("Skipping duplicated experiment {}, params {}".format(local_experiment, params))

    for feats_window_size in feats_window_sizes:
        for feat in feats:
            for classifier in classifiers:
                for eval_window_size in eval_window_sizes:
                    experiment = ["eval_erisk"]
                    params["eval_window_size"] = eval_window_size
                    do_experiment(experiment)
                experiment = ["train", "classify", "evaluate", "eval_erisk"]
                params["classifier"] = classifier
                do_experiment(experiment)
            if not feat == "text":
                for max_feature in max_features:
                    experiment = ["featurize", "select_feats", "train", "classify", "evaluate", "eval_erisk"]
                    params["max_features"] = max_feature
                    do_experiment(experiment)
            experiment = ["select_feats", "train", "classify", "evaluate", "eval_erisk"]
            params["feats"] = feat
            do_experiment(experiment)
        experiment = None
        params["feats_window_size"] = feats_window_size
        do_experiment(experiment)

    print("ENDED EXPERIMENTS")
    # want to test all combinations



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    experiments()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
