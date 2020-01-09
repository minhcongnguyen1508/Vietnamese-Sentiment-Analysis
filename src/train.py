from utils import load_data_from_dir, load_data_from_file, y2labels, y2sentiment, pre_process, prepare_data
import numpy as np
from word2vec import comment_embedding, WordModel
from CNN import CNN
from HierarchicalAttentionNetwork import HAN_MODELS
from confuse_matrix import Confusion
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
# Parameter
max_seq=150
embedding_size=128
# Create Word.model
reviews, y_labels = load_data_from_dir('../data/raw')
y_5class_labels, dis = y2labels(y_labels)
# reviews = pre_process(reviews)
# x = WordModel(reviews, max_seq, embedding_size)

# Load data
reviews_train, y_labels_train = load_data_from_dir('../data/train')
reviews_test, y_labels_test = load_data_from_dir('../data/test')

y_5class_train, dis_5 = y2labels(y_labels_train)
y_3class_train, dis_3 = y2sentiment(y_labels_train)
y_5class_test, dis_5t = y2labels(y_labels_test)
y_3class_test, dis_3t = y2sentiment(y_labels_test)
df = pd.DataFrame({'name':["label1", "label2", "label3", "label4", "label5"], 'dis':dis})
df.plot(kind='bar',x='name',y='dis')
plt.savefig('distribution.png')
# Data Prepare
x_train, y_train = prepare_data(reviews, y_5class_labels, '../models/word.model')
x_valid, y_valid = prepare_data(reviews_test, y_5class_test, '../models/word.model')

def run_cnn(x_train, y_train, x_valid, y_valid, save_path='../models/vi-5sentiment-analysis_cnn.models'):
    # Create model
    cnn = CNN(sequence_length=max_seq)
    # print(len(x_train), len(y_train))
    cnn.train(x_train, y_train, x_valid, y_valid, save_path)
    cm = Confusion(x_valid, y_valid)
    cm.confusion_matrix()
def run_han(x_train, y_train, x_valid, y_valid, save_path='../models/vi-5sentiment-analysis_han.models'):
    han = HAN_MODELS()
    han.train(x_train, y_train, x_valid, y_valid, save_path='../models/vi-5sentiment-analysis_han.models')

if __name__ == "__main__":
    # print(x_valid.shape)
    run_cnn(x_train, y_train, x_valid, y_valid)
    # run_han(x_train, y_train, x_valid, y_valid)
    # print(x_valid[0].shape)