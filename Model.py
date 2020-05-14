#!/usr/bin/env python
# coding: utf-8

# In[4]:


import sys, os, re, csv, codecs, numpy as np, pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D,Bidirectional
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import gensim.models.keyedvectors as word2vec
import gc


# In[5]:


train = pd.read_csv('train.csv')


# In[6]:


list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values
list_sentences_train = train["comment_text"]
  


# In[7]:


max_features = 210338
tokenizer = Tokenizer(num_words=max_features,  filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' ')
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
word_index = tokenizer.word_index


# In[78]:


maxlen = 200
X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)


# In[79]:


embeddings_index={}
f = open('glove.6B.100d.txt','r',encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs 
f.close()


# In[80]:


embedding_matrix = np.zeros((len(word_index) , 100))
for word, i in word_index.items():
    i-=1
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


# In[81]:


embedding_matrix.shape


# In[83]:




import tensorflow as tf
model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(len(tokenizer.word_index),maxlen, weights=[embedding_matrix],trainable=False, input_shape=(maxlen,)))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)))
model.add(tf.keras.layers.Dense(32, activation="relu"))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.Dense(6, activation="sigmoid"))
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
model.summary()


# In[84]:


batch_size = 32
epochs = 1
model.fit(X_t,y, batch_size=batch_size, epochs=epochs, validation_split=0.2)


# In[91]:





# In[ ]:


model.save('glove_model.h5')


# embedding_index['Rfc']

# In[86]:





# In[36]:


metadata = {
  'word_index': tokenizer.word_index,
  'max_len': maxlen,
  'vocabulary_size': max_features,
}


# In[37]:


import json

with open('metadata.json', 'w') as fp:
    json.dump(metadata, fp)


# In[70]:





# In[1]:





# In[2]:





# In[12]:





# In[ ]:




