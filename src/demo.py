# -*- coding: utf-8 -*-
import tensorflow as tf
from word2vec import comment_embedding, WordModel
import gensim.models.keyedvectors as word2vec
import numpy as np
from utils import pre_process


model = tf.keras.models.load_model("../models/vi-5sentiment-analysis_cnn.models")

text = ["San pham dep, minh rat thich chat vai nay!!!! 👍👍👍", "Làm ăn như ccccc 😡😡😡", "Nghỉ mẹ đi :(", "Cũng tạm", "Shop chán vkl", "Ncl ổn, mk cho ⭐⭐⭐⭐⭐"]
text = pre_process(text)

# print(text)

model_embedding = word2vec.KeyedVectors.load('../models/word.model')

data_prid = []
for i in text:
    data_prid.append(comment_embedding(i, model_embedding))

data_prid = np.array(data_prid)
data_prid = data_prid.reshape(data_prid.shape[0], 150, 128, 1).astype('float32')

result = model.predict(data_prid)
for i in result:
   if np.argmax(i) == 0 or np.argmax(i)==1:
       print("Label predict: NEGATIVE", np.argmax(i))
   elif np.argmax(i) == 2:
       print("Label predict: NEUTRAL", np.argmax(i))
   else:
       print("Label predict: POSITIVE", np.argmax(i))
