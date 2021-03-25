from utils import write_experiment
from utils import load_parameters

def get_experiment_settings():

    params = load_parameters()

    feats_window_sizes = [10]
    eval_window_sizes = [1]
    # feats = ["combined"]
    classifiers = ["svm"]
    # strategies = ["weights"]
    # text_features = ["select"]
    ranges_max = [(-1, 100, -1)]  # [(100, 100), (100, -1), (-1, -1)]
    repeat_experiments = True
    # weights_type = ["all", "window"]
    # weights_window_size = 100

    feats = [("tfidf", "", True, True, "all"), ("tfidf", "", True, True, "positives")]
    train_strategies = [("weights", "all", 0)]

    write_experiment("Testing training tfidf with only positives or all users")

    experiments = []
    for train_pos_range_max, train_neg_range_max, test_range_max in ranges_max:
        for feats_window_size in feats_window_sizes:
            for feat, text_feature, prons, nssi, tfidf_type in feats:
                for classifier in classifiers:
                    for strategy, weights_type, weights_window_size in train_strategies:
                        for eval_window_size in eval_window_sizes:
                            params["classifier"] = classifier
                            params["eval_window_size"] = eval_window_size
                            params["feats_window_size"] = feats_window_size
                            params["feats"] = feat
                            params["tfidf_type"] = tfidf_type
                            params["text_features"] = text_feature
                            params["prons"] = prons
                            params["nssi"] = nssi
                            params["strategy"] = strategy
                            params["weights_type"] = weights_type
                            params["weights_window_size"] = weights_window_size
                            params["train_pos_range_max"] = train_pos_range_max
                            params["train_neg_range_max"] = train_neg_range_max
                            params["test_range_max"] = test_range_max
                            experiments.append(params.copy())

    return experiments