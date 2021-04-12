from utils import write_experiment
from sklearn.model_selection import ParameterGrid
import pandas as pd

def get_experiment_settings():


    new_params = {}

    new_params["feats_window_size"] = [10]
    new_params["eval_window_size"] = [1, 3, 5, 10]
    new_params["classifier"] = ["svm"]
    new_params["strategy"] = ["weights"]
    new_params["train_pos_range_max"] = [100]
    new_params["train_neg_range_max"] = [100]
    new_params["test_range_max"] = [100]
    new_params["max_features"] = [5000]
    new_params["weights_type"] = ["all"]
    new_params["weights_window_size"] = [100]

    new_params["feats"] = ["combined"]
    new_params["text_features"] = ["all", "select"]
    new_params["prons"] = [True]
    new_params["nssi"] = [True]
    new_params["tfidf_type"] = ["positives"]
    new_params["tfidf_ngrams"] = [False]

    new_params["discretize"] = [True]
    new_params["discretize_size"] = [50, 75]
    new_params["discretize_strategy"] = ['quantile']  # uniform, quantile, kmeans
    new_params["discretize_encode"] = ['onehot']   # onehot, onehot-dense, ordinal

    write_experiment("Testing eval window size with sizes 1, 3, 5, and 10")

    experiments = list(ParameterGrid(new_params))

    experiments = pd.DataFrame(experiments).drop_duplicates().to_dict('records')

    print(experiments)
    print(len(experiments))

    return experiments

if __name__ == '__main__':
    get_experiment_settings()