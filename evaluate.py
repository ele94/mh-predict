# normal classification evaluation

# comes after classify.py

from sklearn.metrics import precision_recall_curve
import sklearn.metrics as metrics
from utils import load_pickle
from utils import save_pickle
import numpy

test_y_file = "test.y.pkl"
resul_file = "test.resul.pkl"

labels = load_pickle(test_y_file)
predictions = load_pickle(resul_file)

classification_report = metrics.classification_report(labels, predictions)
confusion_matrix = metrics.confusion_matrix(labels, predictions)
accuracy_score = metrics.accuracy_score(predictions, labels)

print(classification_report)
numpy.savetxt('data/Metrics.txt', classification_report)
print(confusion_matrix)
numpy.savetxt('data/Confusion_matrix.txt', confusion_matrix)
print(accuracy_score)
numpy.savetxt('data/Accuracy_score.txt', accuracy_score)