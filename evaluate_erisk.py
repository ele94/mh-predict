# evaluation for the erisk task

# comes after classify.py

from utils import eval_performance
import csv
import pickle
import os
from utils import load_pickle
from utils import save_pickle
import filenames as fp

test_data_path = "/home/elena/Documentos/UNED/erisk/2021/data/erisk2021_test_data"
g_truth_filename = "golden_truth.txt"
erisk_eval_file = "erisk.eval.resuls.csv"
from pprint import pprint
import numpy as np
from utils import load_parameters
from datetime import datetime
import subprocess

def main():

    params = load_parameters()

    feats_window_size = params["feats_window_size"]
    window_size = params["eval_window_size"]

    resuls_path = fp.get_resuls_path()
    window_path = fp.get_window_path()

    g_truth = load_golden_truth(test_data_path, g_truth_filename)
    test_resuls = load_pickle(resuls_path, fp.resul_file)
    test_scores = load_pickle(resuls_path, fp.score_file)
    test_x = load_pickle(window_path, fp.test_x_filename)

    user_resul = prepare_data(test_x, test_resuls)
    user_scores = prepare_data(test_x, test_scores)

    test_resul_proc = process_decisions(user_resul, user_scores, feats_window_size, max_strategy=window_size)
    eval_resuls = eval_performance(test_resul_proc, g_truth)

    write_csv(eval_resuls)


def write_csv(eval_resuls):

    data = {}
    data["commit hash"] = subprocess.check_output(["git", "describe", "--always"]).strip().decode()

    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    data["timestamp"] = dt_string


    params = load_parameters()
    data.update(params)
    data.update(eval_resuls)

    csv_file = erisk_eval_file

    csv_columns = data.keys()
    dict_data = [data]

    try:
        with open(csv_file, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            if os.path.getsize(csv_file) == 0:
                writer.writeheader()
            for data in dict_data:
                writer.writerow(data)
    except IOError:
        print("I/O error")

    csv_columns = eval_resuls.keys()
    dict_data = [eval_resuls]





def prepare_data(test_x, resul_array):

    test_users = np.array(test_x[["user"]]).flatten()
    resul_array = resul_array.tolist()
    test_users = test_users.tolist()

    user_tuples = list(zip(test_users, resul_array))
    user_dict = array_to_dict(user_tuples)

    return user_dict



def array_to_dict(l):
    d = dict()
    [d[t[0]].append(t[1]) if t[0] in list(d.keys())
     else d.update({t[0]: [t[1]]}) for t in l]
    return d

def process_decisions(user_decisions, user_scores, feat_window_size, max_strategy=5):
    decision_list = []
    new_user_decisions = {}
    new_user_sequence = {}
    max = max_strategy

    for user, decisions in user_decisions.items():
        new_user_decisions[user] = []
        new_user_sequence[user] = []

    # politica de decisiones: decidimos que un usuario es positivo a partir del 5 mensaje positivo consecutivo
    # a partir de ahi, todas las decisiones deben ser positivas, y la secuencia mantenerse estable
    for user, decisions in user_decisions.items():
        count = 0
        for i in range(0, len(decisions)):
            if decisions[i] == 0 and count < max:
                count = 0
                new_user_decisions[user].append(0)
                new_user_sequence[user].append(i+feat_window_size)
            elif decisions[i] == 1 and count < max:
                count = count +1
                new_user_decisions[user].append(0)
                new_user_sequence[user].append(i+feat_window_size)
            elif count >= max:
                new_user_decisions[user].append(1)
                new_user_sequence[user].append(new_user_sequence[user][i-1])

    # lo montamos en el formato que acepta el evaluador
    for user, decisions in new_user_decisions.items():
        decision_list.append(
            {"nick": user, "decision": new_user_decisions[user][-1], "sequence": new_user_sequence[user][-1], "score":
                user_scores[user][-1]})

    return decision_list

def load_golden_truth(path, filename):
    g_path = os.path.join(path, filename)
    g_truth = {line.split()[0]: int(line.split()[1]) for line in open(g_path)}
    return g_truth

if __name__ == '__main__':
    main()