#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np, pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional
from keras.models import Model
from keras import optimizers, layers



# In[5]:


train = pd.read_csv('train.csv')


# In[6]:


list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values
list_sentences_train = train["comment_text"]
  


# In[7]:


max_features = 210340
tokenizer = Tokenizer(num_words=max_features,  filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',oov_token='<OOV>' ,lower=True, split=' ')
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
word_index = tokenizer.word_index


# In[78]:


maxlen = 80
X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)


# In[79]:




import tensorflow as tf
model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(len(tokenizer.word_index)+1,32, input_shape=(maxlen,)))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(32, activation="relu"))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(6, activation="sigmoid"))
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
model.summary()
# In[84]:


batch_size = 128
epochs = 2
model.fit(X_t,y, batch_size=batch_size, epochs=epochs, validation_split=0.2)


# In[91]:





# In[ ]:


model.save('learned_embedding32.h5')



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





