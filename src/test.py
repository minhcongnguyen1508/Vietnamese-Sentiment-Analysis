import tensorflow as tf
import gensim.models.keyedvectors as word2vec
import numpy as np
from utils import load_data_from_dir, load_data_from_file, y2labels, y2sentiment, pre_process, prepare_data
from sklearn.metrics import f1_score

model = tf.keras.models.load_model("../models/vi-5sentiment-analysis_cnn.models")
reviews_test, y_labels_test = load_data_from_dir('../data/test')
y_labels_test, dis = y2labels(y_labels_test)
x_test, y_test = prepare_data(reviews_test, y_labels_test, '../models/word.model')

def f1_score_(y_test ,pred):
    y_true = y_test.argmax(axis=1)
    y_pred = pred.argmax(axis=1)
    f1 = f1_score(y_true, y_pred, average='weighted')
    print(y_true[0], y_pred[0])
    return f1

y_pred = model.predict(x_test)
# print(y_pred)
print(f1_score_(y_test, y_pred))

np.savetxt('nhom13.txt', y_pred)
