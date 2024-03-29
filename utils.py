import os
import pickle
import yaml
import sys
import subprocess
from datetime import datetime
import filenames as fp
import csv

params_file = "params.yaml"
log_file = "log.txt"
report_file = "experiments.txt"

######## logger

def logger(message, log_file=log_file):
    print(message)
    original_stdout = sys.stdout # Save a reference to the original standard output
    with open(log_file, 'a') as f:
        sys.stdout = f # Change the standard output to the file we created.
        print(message)
        sys.stdout = original_stdout # Reset the standard output to its original value


def write_experiment(message, report_file=report_file):
    commit = subprocess.check_output(["git", "describe", "--always"]).strip().decode()
    now = datetime.now()
    timestamp = now.strftime("%d/%m/%Y %H:%M:%S")
    message = " ".join([commit, timestamp, message, "\n"])

    report_file = os.path.join(fp.resuls_path, report_file)

    with open(report_file, 'a') as f:
        f.write(message)

######### pickles

def save_pickle(paths, filename, data):
    filepaths = os.path.join(paths)
    if not os.path.exists(filepaths):
        os.makedirs(filepaths)
    file = os.path.join(filepaths, filename)
    with open(file, 'wb') as data_file:
        pickle.dump(data, data_file)


def load_pickle(paths, filename):
    filepaths = os.path.join(paths)
    file = os.path.join(filepaths, filename)
    with open(file, 'rb') as data_file:
        data = pickle.load(data_file)
    return data


def remove_pickle(paths, filename):
    filepaths = os.path.join(paths)
    file = os.path.join(filepaths, filename)
    if os.path.exists(file):
        os.remove(file)

def check_pickle(paths, filename):
    filepaths = os.path.join(paths)
    if os.path.exists(filepaths):
        file = os.path.join(filepaths, filename)
        return os.path.isfile(file)
    else:
        return False




def write_csv(eval_resuls):

    data = {}
    data["commit hash"] = subprocess.check_output(["git", "describe", "--always"]).strip().decode()

    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    data["timestamp"] = dt_string

    params = load_parameters()
    data.update(params)
    data.update(eval_resuls)

    erisk_eval_file = os.path.join(fp.resuls_path, fp.erisk_eval_filename)
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


######## parameters


def load_parameters():
    with open(params_file) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    return params


def update_parameters(params):
    with open(params_file, 'w') as f:
        yaml.safe_dump(params, f, default_flow_style=False)
    return params






########## erisk evaluation



def penalty(delay):
    import numpy as np
    p = 0.0078
    pen = -1.0 + 2.0 / (1 + np.exp(-p * (delay - 1)))
    return (pen)


def eval_performance(run_results, qrels):
    import numpy as np

    total_pos = n_pos(qrels)

    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0
    erdes5 = np.zeros(len(run_results))
    erdes50 = np.zeros(len(run_results))
    ierdes = 0
    latency_tps = list()
    penalty_tps = list()

    for r in run_results:
        try:
            # print(qrels[ r['nick']   ], r['decision'], r['nick'], qrels[ r['nick']   ] ==  r['decision'] )
            if (qrels[r['nick']] == r['decision']):
                if (r['decision'] == 1):
                    # print('dec = 1')
                    true_pos += 1
                    erdes5[ierdes] = 1.0 - (1.0 / (1.0 + np.exp((r['sequence'] + 1) - 5.0)))
                    erdes50[ierdes] = 1.0 - (1.0 / (1.0 + np.exp((r['sequence'] + 1) - 50.0)))
                    latency_tps.append(r['sequence'] + 1)
                    penalty_tps.append(penalty(r['sequence'] + 1))
                else:
                    # print('dec = 0')
                    true_neg += 1
                    erdes5[ierdes] = 0
                    erdes50[ierdes] = 0
            else:
                if (r['decision'] == 1):
                    # print('++')
                    false_pos += 1
                    erdes5[ierdes] = float(total_pos) / float(len(qrels))
                    erdes50[ierdes] = float(total_pos) / float(len(qrels))
                else:
                    # print('****')
                    false_neg += 1
                    erdes5[ierdes] = 1
                    erdes50[ierdes] = 1

        except KeyError:
            print("User does not appear in the qrels:" + r['nick'])

        ierdes += 1

    if (true_pos == 0):
        precision = 0
        recall = 0
        F1 = 0
    else:
        precision = float(true_pos) / float(true_pos + false_pos)
        recall = float(true_pos) / float(total_pos)
        F1 = 2 * (precision * recall) / (precision + recall)

    speed = 1 - np.median(np.array(penalty_tps))

    eval_results = {}
    eval_results['precision'] = precision
    eval_results['recall'] = recall
    eval_results['F1'] = F1
    eval_results['ERDE_5'] = np.mean(erdes5)
    eval_results['ERDE_50'] = np.mean(erdes50)
    eval_results['median_latency_tps'] = np.median(np.array(latency_tps))
    eval_results['median_penalty_tps'] = np.median(np.array(penalty_tps))
    eval_results['speed'] = speed
    eval_results['latency_weighted_f1'] = F1 * speed

    return eval_results


def n_pos(qrels):
    total_pos = 0
    for key in qrels:
        total_pos += qrels[key]
    return (total_pos)