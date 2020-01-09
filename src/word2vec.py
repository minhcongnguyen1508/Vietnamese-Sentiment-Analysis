from gensim.models import Word2Vec
import numpy as np
import gensim.models.keyedvectors as word2vec
from underthesea import word_tokenize

def WordModel(x, max_seq=150, embedding_size=128):
    input_gensim = []
    for review in x:
        input_gensim.append(word_tokenize(review))
    model = Word2Vec(input_gensim, size=embedding_size, window=5, min_count=0, workers=4, sg=1)
    print(model)
    model.wv.save("../models/word.model")
    return input_gensim

def comment_embedding(comment, model_embedding, max_seq=150, embedding_size=128):
    word_labels = []
    for word in model_embedding.vocab.keys():
        word_labels.append(word)
    matrix = np.zeros((max_seq, embedding_size))
    words = word_tokenize(comment)
    lencmt = len(words)
    for i in range(max_seq):
        indexword = i % lencmt
        if (max_seq - i < lencmt):
            break
        if(words[indexword] in word_labels):
            matrix[i] = model_embedding[words[indexword]]
    matrix = np.array(matrix)
    return matrix
