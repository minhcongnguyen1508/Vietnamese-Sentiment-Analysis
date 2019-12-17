from load_data import load_data_from_dir, load_data_from_file, y2labels, y2sentiment
from gensim.models import Word2Vec
import numpy as np

reviews, y_labels = load_data_from_dir('../data/train')
y_label = y2labels(y_labels)
labels = y2sentiment(y_labels)

# input_gensim = []
# for review in reviews:
#     input_gensim.append(review.split())
    
# model = Word2Vec(input_gensim, size=128, window=5, min_count=0, workers=4, sg=1)
# model.wv.save("word.model")

# print(input_gensim)

import gensim.models.keyedvectors as word2vec
model_embedding = word2vec.KeyedVectors.load('./word.model')

word_labels = []
max_seq = 200
embedding_size = 128

for word in model_embedding.vocab.keys():
    word_labels.append(word)
    
def comment_embedding(comment):
    matrix = np.zeros((max_seq, embedding_size))
    words = comment.split()
    lencmt = len(words)

    for i in range(max_seq):
        indexword = i % lencmt
        if (max_seq - i < lencmt):
            break
        if(words[indexword] in word_labels):
            matrix[i] = model_embedding[words[indexword]]
    matrix = np.array(matrix)
    return matrix

train_data = []
label_data = y_label

for i in reviews:
    train_data.append(comment_embedding(i))

train_data = np.array(train_data)

print(train_data[0])
print(label_data[0])