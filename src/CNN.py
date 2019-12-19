import numpy as np
import tensorflow as tf
from tensorflow.contrib import keras

class CNN:
    def __init__(self, models=None, sequence_length=200, embedding_size=128, num_classes=3, 
                filter_sizes=3, num_filters=150, epochs=30, batch_size=30, learning_rate=0.1, dropout_rate=0.5):
        self.sequence_length = sequence_length
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.models = models

    def create_models(self):
        # Define model
        model = keras.models.Sequential()
        model.add(keras.layers.Convolution2D(self.num_filters, (self.filter_sizes, self.embedding_size),
                        padding='valid',
                        input_shape=(self.sequence_length, self.embedding_size, 1), activation='relu'))
        model.add(keras.layers.MaxPooling2D(pool_size=(198, 1)))
        model.add(keras.layers.Dropout(self.dropout_rate))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(128, activation='relu'))
        model.add(keras.layers.Dense(self.num_classes, activation='softmax'))
        # Train model
        adam = tf.train.AdamOptimizer()
        model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])
        print(model.summary())
        return model

    def train(self, x_train, y_train, x_valid, y_valid, path_save='../models/vi-sentiment-analysis-5class.models'):
        model = self.create_models()
        model.fit(x = x_train, y = y_train, batch_size = self.batch_size, verbose=1, epochs=self.epochs, validation_data=(x_valid, y_valid))
        model.save(path_save)
    
