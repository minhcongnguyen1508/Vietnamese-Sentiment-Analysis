import os
import re

import numpy as np
import pandas as pd
# from bs4 import BeautifulSoup
from keras import backend as K
from keras.models import Model
from keras import initializers
from keras.engine.topology import Layer
from keras.layers import Dense, Input
from keras.layers import Embedding, GRU, Bidirectional, TimeDistributed
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.utils.np_utils import to_categorical
from nltk import tokenize
import gensim.models.keyedvectors as word2vec

class HierarchicalAttentionNetwork(Layer):
    def __init__(self, attention_dim):
        self.init = initializers.get('normal')
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(HierarchicalAttentionNetwork, self).__init__()

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)))
        self.b = K.variable(self.init((self.attention_dim,)))
        self.u = K.variable(self.init((self.attention_dim, 1)))
        self.trainable_weights = [self.W, self.b, self.u]
        super(HierarchicalAttentionNetwork, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))

        ait = K.exp(K.squeeze(K.dot(uit, self.u), -1))

        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        weighted_input = x * K.expand_dims(ait)
        output = K.sum(weighted_input, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

class HAN_MODELS:
    def __init__(self, max_len=150, max_sentences=150, embedding_dim=128, epochs=50, batch_size=500):
        self.max_len = max_len
        self.max_sentences = max_sentences
        self.embedding_dim = embedding_dim
        self.epochs = epochs
        self.batch_size = batch_size

    def train(self, x_train, y_train, x_val, y_val, save_path='../models/vi-sentiment-analysis-5class.models'):
        # embedding_matrix = word2vec.KeyedVectors.load(model_embedding_path)
        # print(embedding_matrix)
        # embedding_matrix = np.random.random((len(model_embedding) + 1, self.embedding_dim))
        # for word, i in model_embedding.items():
        #     embedding_vector = embeddings_index.get(word)
        #     if embedding_vector is not None:
        #         # words not found in embedding index will be all-zeros.
        #         embedding_matrix[i] = embedding_vector

        embedding_layer = Embedding(self.max_len + 1, self.embedding_dim,
                            input_length=self.max_len, trainable=True, mask_zero=True)

        sentence_input = Input(shape=(self.max_len,), dtype='int32')
        embedded_sequences = embedding_layer(sentence_input)
        lstm_word = Bidirectional(GRU(100, return_sequences=True))(embedded_sequences)
        attn_word = HierarchicalAttentionNetwork(100)(lstm_word)
        sentenceEncoder = Model(sentence_input, attn_word)

        review_input = Input(shape=(self.max_sentences, self.max_len), dtype='int32')
        review_encoder = TimeDistributed(sentenceEncoder)(review_input)
        lstm_sentence = Bidirectional(GRU(100, return_sequences=True))(review_encoder)
        attn_sentence = HierarchicalAttentionNetwork(100)(lstm_sentence)
        preds = Dense(2, activation='softmax')(attn_sentence)
        model = Model(review_input, preds)

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

        print("model fitting - Hierachical attention network")
        model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=self.epochs, batch_size=self.batch_size)


# if __name__ == "__main__":
#     han = HAN_MODELS()
#     han.train(0, 0, 0, 0)
