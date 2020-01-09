import numpy as np
import tensorflow as tf
from tensorflow.contrib import keras
import matplotlib.pyplot as plt
import keras.backend as K

class CNN:
    def __init__(self, models=None, sequence_length=150, embedding_size=128, num_classes=5, 
                filter_sizes=3, num_filters=150, epochs=50, batch_size=30, learning_rate=0.001, dropout_rate=0.5):
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
    def f1(self, y_true, y_pred): #taken from old keras source code
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        recall = true_positives / (possible_positives + K.epsilon())
        f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
        return f1_val
    def create_models(self):
        # Define model
        model = keras.models.Sequential()
        model.add(keras.layers.Convolution2D(self.num_filters, (self.filter_sizes, self.embedding_size),
                        padding='valid',
                        input_shape=(self.sequence_length, self.embedding_size, 1), activation='relu'))
        model.add(keras.layers.MaxPooling2D(pool_size=(self.sequence_length-2, 1)))
        model.add(keras.layers.Dropout(self.dropout_rate))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(128, activation='relu'))
        model.add(keras.layers.Dense(self.num_classes, activation='softmax'))
        # Train model
        adam = tf.train.AdamOptimizer()
        model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=[self.f1])
        print(model.summary())
        return model

    def train(self, x_train, y_train, x_valid, y_valid, path_save='../models/vi-5sentiment-analysis_cnn.models'):
        model = self.create_models()
        history = model.fit(x = x_train, y = y_train, validation_data=(x_valid, y_valid), batch_size = self.batch_size, verbose=1, epochs=self.epochs, shuffle=True)
        self.acc_visualization(history)
        self.loss_visualization(history)
        model.save(path_save)
    
    def acc_visualization(self, history, path_save='../models/acc_cnn.png'):
        # Plot training & validation accuracy values
        plt.plot(history.history['f1'])
        plt.plot(history.history['val_f1'])
        plt.title('Model F1_Score')
        plt.ylabel('F1_Score')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig(path_save)
        plt.show()
    def loss_visualization(self, history, path_save='../models/loss_cnn.png'):
        # Plot training & validation accuracy values
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig(path_save)
        plt.show()
