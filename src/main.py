from utils import load_data_from_dir, load_data_from_file, y2labels, y2sentiment, pre_process, prepare_data
import numpy as np
from word2vec import comment_embedding, WordModel
from CNN import CNN
from text_feature import LowerCase, RemoveTone, CountEmoticons, RemoveEmoticons, RemoveDuplicate
from gensim.models import Word2Vec
import gensim.models.keyedvectors as word2vec
from copy import deepcopy

# Create model
CNN = CNN()

# Load data
reviews_train, y_labels_train = load_data_from_dir('../data/train')
reviews_test, y_labels_test = load_data_from_dir('../data/test')

y_class_train = y2labels(y_labels_train)
label_data_train = y2sentiment(y_labels_train)
y_class_test = y2labels(y_labels_test)
label_data_test = y2sentiment(y_labels_test)
# Data Prepare
x_train, y_train = prepare_data(reviews_train, y_class_train, '../models/word.model')
x_valid, y_valid = prepare_data(reviews_test, y_class_test, '../models/word.model')

# print(len(x_train), len(y_train))
CNN.train(x_train, y_train, x_valid, y_valid, '../models/vi-sentiment-analysis-5class.models')