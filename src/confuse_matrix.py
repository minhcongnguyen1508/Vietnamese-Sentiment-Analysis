from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from tensorflow import keras
import pandas as pd
import seaborn as sn
from sklearn.metrics import f1_score
import numpy as np

class Confusion:
    def __init__(self, x_valid, y_valid):
        self.x_valid = x_valid
        self.y_valid = y_valid
    def confusion_matrix(self, load_model_from_path= "../models/vi-5sentiment-analysis_cnn.models", save_path='../models/confusion_matrix.png'):
        model = keras.models.load_model(load_model_from_path)
        # reviews_test, y_labels_test = load_data_from_dir('../data/test')
        # y_class_test = y2labels(y_labels_test)
        # label_data_test = y2sentiment(y_labels_test)
        # Data Prepare
        # x_valid, y_valid = prepare_data(reviews_test, y_class_test, './models/word.model')
        # Plot non-normalized confusion matrix
        titles_options = [("Confusion matrix, without normalization", None),
                        ("Normalized confusion matrix", 'true')]
        pred = model.predict(self.x_valid)
        cm=confusion_matrix(self.y_valid.argmax(axis=1), pred.argmax(axis=1))
        print(cm)
        n_class=5
        df_cm = pd.DataFrame(cm, range(n_class),range(n_class))
        #plt.figure(figsize = (10,7))
        sn.set(font_scale=1.4)#for label size
        sn.heatmap(df_cm, annot=True,annot_kws={"size": 10})# font size
        print("F1_Score: ",self.f1_score(pred))
        plt.savefig(save_path)
        plt.show()

    def f1_score(self, pred):
        y_true = self.y_valid.argmax(axis=1)
        y_pred = pred.argmax(axis=1)
        # f1_score(y_true, y_pred, average='macro')
        # f1_score(y_true, y_pred, average='micro')
        f1 = f1_score(y_true, y_pred, average='weighted')
        # f1_score(y_true, y_pred, average=None)
        # y_true = [0, 0, 0, 0, 0, 0]
        # y_pred = [0, 0, 0, 0, 0, 0]
        # f1_score(y_true, y_pred, zero_division=1)
        return f1