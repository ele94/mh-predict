
# todo pasar desde la libreta classify.py!!!
import pickle
from sklearn.metrics import precision_recall_curve
import sklearn.metrics as metrics

#with open('data/pickles/test.labels.mod.pkl', 'rb') as test_labels_file:
#    test_labels = pickle.load(test_labels_file)

with open('data/pickles/test.y.pkl', 'rb') as test_y_file:
    test_labels = pickle.load(test_y_file)

#with open('data/pickles/test.resul.pkl', 'rb') as test_resul_file:
#    test_resul = pickle.load(test_resul_file)

with open('data/pickles/test.resul.mod.pkl', 'rb') as test_resul_mod_file:
    predictions = pickle.load(test_resul_mod_file)



#predictions = test_resul
labels = test_labels

print(metrics.classification_report(labels, predictions))

print(metrics.confusion_matrix(labels, predictions))

print(metrics.accuracy_score(predictions, labels))