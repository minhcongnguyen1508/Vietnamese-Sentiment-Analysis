import tensorflow as tf
from word2vec import comment_embedding, WordModel
import gensim.models.keyedvectors as word2vec
import numpy as np
from utils import pre_process
from underthesea import word_tokenize

text = ["San pham dep, minh rat thich chat vai nay!!!! 👍👍👍", "Làm ăn như ccccc 😡😡😡", "<3", "💥💥💥Giao hàng nhanh.👏 💥💥💥", "🔥 <3"]
text = pre_process(text)
print(text)
for i in range(len(text)):
    text[i] = word_tokenize(text[i])

print(text)
