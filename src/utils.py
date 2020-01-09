import os
from text_feature import LowerCase, RemoveTone, CountEmoticons, RemoveEmoticons, RemoveDuplicate
import gensim.models.keyedvectors as word2vec
import numpy as np
from word2vec import comment_embedding, WordModel

def load_data_from_file(filepath):
    sentences = []
    labels = []
    with open(filepath, encoding='utf8') as fp:
        line = fp.readline()
        line = line.strip()
        sentences.append(line)
        cnt = 1
        while line:
            line = fp.readline()
            line = line.strip()
            if len(line) == 0:
                continue
            if cnt%2==0:
                sentences.append(line)
            else:
                labels.append(line)
            cnt += 1
    return sentences, labels

def load_data_from_dir(path):
    file_names = os.listdir(path)
    chain_words = None
    labels = None
    for f_name in file_names:
        file_path = os.path.join(path, f_name)
        if f_name.startswith('.') or os.path.isdir(file_path):
            continue
        batch_sentences, batch_labels = load_data_from_file(file_path)
        if chain_words is None:
            chain_words = batch_sentences
            labels = batch_labels
        else:
            chain_words += batch_sentences
            labels += batch_labels
    return chain_words, labels

def y2sentiment(labels):
    sentiments = [] 
    distribution = [0, 0, 0]
    for line in labels:
        y_sentiment = [0, 0, 0] # NEG = -1; POS = 1, NEU = 0
        if line[:3] == 'NEG':
            y_sentiment[0] = 1
            distribution[0] += 1
            sentiments.append(y_sentiment)
        elif line[:3] == 'POS':
            y_sentiment[2] = 1
            distribution[2] += 1
            sentiments.append(y_sentiment)
        else:
            y_sentiment[1] = 1
            distribution[1] += 1
            sentiments.append(y_sentiment)
    return sentiments, distribution

def y2labels(labels):
    y_labels = []
    distribution = [0, 0, 0, 0, 0]
    for line in labels:
        y_class = [0, 0, 0, 0, 0]
        if line[-1] == '1':
            y_class[0] = 1
            distribution[0] += 1
            y_labels.append(y_class)
        elif line[-1] == '2':
            y_class[1] = 1
            distribution[1] += 1
            y_labels.append(y_class)
        elif line[-1] == '3':
            y_class[2] = 1
            distribution[2] += 1
            y_labels.append(y_class)
        elif line[-1] == '4':
            y_class[3] = 1
            distribution[3] += 1
            y_labels.append(y_class)
        elif line[-1] == '5':
            y_class[4] = 1
            distribution[4] += 1
            y_labels.append(y_class)
    return y_labels, distribution

def pre_process(text):
    text = LowerCase(text)
    text = RemoveEmoticons(text)
    text = RemoveTone(text)
    text = RemoveDuplicate(text)
    return text

def prepare_data(x_text, y_labels, path_model='./models/word.model', sequence_length = 150, embedding_size = 128):
    model_embedding = word2vec.KeyedVectors.load(path_model)
    x_text = pre_process(x_text)
    x_text = remove_tone_5class(x_text, y_labels, model_embedding)
    data_x = np.array(x_text)
    data_x = data_x.reshape(data_x.shape[0], sequence_length, embedding_size, 1).astype('float32')
    data_y = np.array(y_labels)
    return data_x, data_y

def remove_tone_3class(reviews, label_data, model_embedding):
    data = []
    for i in range(len(reviews)):
        if reviews[i] == ' ':
            if np.argmax(label_data[i]) == 0:
                reviews[i] += 'negative'
            elif np.argmax(label_data[i]) == 1:
                reviews[i] += 'neutral'
            else:
                reviews[i] += 'positive'
        data.append(comment_embedding(reviews[i], model_embedding))
    return data

def remove_tone_5class(reviews, label_data, model_embedding):
    data = []
    for i in range(len(reviews)):
        if reviews[i] == ' ':
            if np.argmax(label_data[i]) == 0 or np.argmax(label_data[i]) == 1:
                reviews[i] += 'negative'
            elif np.argmax(label_data[i]) == 2:
                reviews[i] += 'neutral'
            else:
                reviews[i] += 'positive'
        data.append(comment_embedding(reviews[i], model_embedding))
    return data