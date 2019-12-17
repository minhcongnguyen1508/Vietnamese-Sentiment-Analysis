import os

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
    for line in labels:
        y_sentiment = [0, 0, 0] # NEG = -1; POS = 1, NEU = 0
        if line[:3] == 'NEG':
            y_sentiment[0] = 1
            sentiments.append(y_sentiment)
        elif line[:3] == 'POS':
            y_sentiment[2] = 1
            sentiments.append(y_sentiment)
        else:
            y_sentiment[1] = 1
            sentiments.append(y_sentiment)
    return sentiments

def y2labels(labels):
    y_labels = []
    for line in labels:
        y_class = [0, 0, 0, 0, 0]
        if line[-1] == '1':
            y_class[0] = 1
            y_labels.append(y_class)
        elif line[-1] == '2':
            y_class[1] = 1
            y_labels.append(y_class)
        elif line[-1] == '3':
            y_class[2] = 1
            y_labels.append(y_class)
        elif line[-1] == '4':
            y_class[3] = 1
            y_labels.append(y_class)
        elif line[-1] == '5':
            y_class[4] = 1
            y_labels.append(y_class)
    return y_labels
