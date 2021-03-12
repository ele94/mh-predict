# normal classification evaluation

# comes after classify.py

import sklearn.metrics as metrics
from utils import load_pickle
from utils import load_parameters
from utils import logger
import filenames as fp

def main():

    params = load_parameters()

    resuls_path = fp.get_resuls_path()
    window_path = fp.get_window_path()

    labels = load_pickle(window_path, fp.test_y_filename)
    predictions = load_pickle(resuls_path, fp.resul_file)

    classification_report = metrics.classification_report(labels, predictions)
    confusion_matrix = metrics.confusion_matrix(labels, predictions)
    accuracy_score = metrics.accuracy_score(predictions, labels)

    logger(classification_report)
    #numpy.savetxt('data/Metrics.txt', classification_report) #todo fix this
    logger(confusion_matrix)
    #numpy.savetxt('data/Confusion_matrix.txt', confusion_matrix)
    logger(accuracy_score)
    #numpy.savetxt('data/Accuracy_score.txt', accuracy_score)



if __name__ == '__main__':
    main()