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

def test():
    # Use a breakpoint in the code line below to debug your script.
    params = load_parameters()
    print(params["feats_window_size"])

    #prepare()
    #windowfy()
    #text_featurize()
    #tfidf_featurize()
    #combine_features()
    #print("Selecting features")
    #select_feats()
    #print("Training")
    #train()
    #print("Classifying")
    #classify()
    #print("Evaluating")
    #evaluate()
    print("Evaluating erisk")
    eval_erisk()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    test()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
