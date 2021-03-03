# normal classification evaluation

# comes after classify.py

from sklearn.metrics import precision_recall_curve
import sklearn.metrics as metrics
from utils import load_pickle
from utils import save_pickle
from utils import load_parameters
import numpy

test_y_filename = "test.y.pkl"
resul_filename = "test.resul.pkl"

def main():

    params = load_parameters()
    test_y_file = str(params["feats_window_size"]) + "." + test_y_filename
    resul_file = str(params["feats_window_size"]) + "." + params["feats"] + "." + params[
        "classifier"] + "." + resul_filename

    labels = load_pickle(test_y_file)
    predictions = load_pickle(resul_file)

    classification_report = metrics.classification_report(labels, predictions)
    confusion_matrix = metrics.confusion_matrix(labels, predictions)
    accuracy_score = metrics.accuracy_score(predictions, labels)

    print(classification_report)
    #numpy.savetxt('data/Metrics.txt', classification_report) #todo fix this
    print(confusion_matrix)
    #numpy.savetxt('data/Confusion_matrix.txt', confusion_matrix)
    print(accuracy_score)
    #numpy.savetxt('data/Accuracy_score.txt', accuracy_score)



if __name__ == '__main__':
    main()