import os
import pickle
import yaml

params_file = "params.yaml"
pickles_path = "data/pickles"

######### pickles


def save_pickle(filename, data):
    file = os.path.join(pickles_path, filename)
    with open(file, 'wb') as data_file:
        pickle.dump(data, data_file)


def load_pickle(filename):
    file = os.path.join(pickles_path, filename)
    with open(file, 'rb') as data_file:
        data = pickle.load(data_file)
    return data


def remove_pickle(filename):
    file = os.path.join(pickles_path, filename)
    if os.path.exists(file):
        os.remove(file)

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