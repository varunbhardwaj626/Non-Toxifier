#!/usr/bin/env python
# coding: utf-8

# In[ ]:



# In[ ]:



import numpy as np
import pandas as pd
import json
import threading
import PySimpleGUI as sg
from keras.preprocessing import text, sequence

def load_emojis():
    rows = []
    with open('emojis.json') as f:
        for emoji in json.loads(f.read()):
            rows.append([emoji['name'], emoji['unicode'], ' '.join(emoji['keywords'])])    
    return np.array(rows)
    
emojis = load_emojis()

with open('metadata.json', 'r') as fp:
    data = json.load(fp)

max_len=200
num_words=data['vocabulary_size']
from keras.preprocessing.text import Tokenizer

t = Tokenizer(num_words=num_words,  filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' ')

t.word_index=data['word_index']
from tensorflow.keras.preprocessing.sequence import pad_sequences

import tensorflow as tf 
m = tf.keras.models.load_model('glove_model.h5')


# In[ ]:



import json
import spacy


def load_emojis():
    rows = []
    with open('emojis.json') as f:
        for emoji in json.loads(f.read()):
            rows.append([emoji['name'], emoji['unicode'], ' '.join(emoji['keywords'])])    
    return np.array(rows)
    
emojis = load_emojis()


# In[ ]:



nlp = spacy.load('en_core_web_sm')
from tqdm import tqdm

with open('glove.6B.100d.txt', 'r',encoding='utf-8') as f:
    for line in tqdm(f, total=400000):
        parts = line.split()
        word = parts[0]
        vec = np.array([float(v) for v in parts[1:]], dtype='f')
        nlp.vocab.set_vector(word, vec)


# In[ ]:


docs = [nlp(str(keywords)) for _, _, keywords in tqdm(emojis)]
doc_vectors = np.array([doc.vector for doc in docs])


# In[ ]:


from numpy import dot
from numpy.linalg import norm

def most_similar(vectors, vec):
    cosine = lambda v1, v2: dot(v1, v2) / (norm(v1) * norm(v2))
    dst = np.dot(vectors, vec) / (norm(vectors) * norm(vec))
    return np.argsort(-dst)

def query(v):
    ids = most_similar(doc_vectors, v)[:5]
    for name, unicode, keywords in emojis[ids]:
        values = unicode.split(' ')
        
        for v in values:
            if int(v[2::],16) in range(0,int('FFFF',16)):
                    c = chr(int(v.replace('U+', ''), 16))
                    return c
    return ''
    


# In[ ]:


def check(s):
    test_example = s
    x_test_ex = t.texts_to_sequences([test_example])
    x_test_ex =sequence.pad_sequences(x_test_ex, maxlen=200, padding='post')
    pred=m.predict(x_test_ex)
    if np.amax(pred)>0.4:
        return True
    else:
        return False



# In[ ]:



sg.ChangeLookAndFeel('darkamber') 
progress_bar1=[[sg.ProgressBar(1000, orientation='h',style='clam',bar_color=('red','white'),size=(30, 4), key='progressbar1')]]
progress_bar2=[[sg.ProgressBar(1000, orientation='h',style='clam',bar_color=('red','white'),size=(30, 4), key='progressbar2')]]
progress_bar3=[[sg.ProgressBar(1000, orientation='h',style='clam',bar_color=('red','white'),size=(30, 4), key='progressbar3')]]
progress_bar4=[[sg.ProgressBar(1000, orientation='h',style='clam',bar_color=('red','white'),size=(30, 4), key='progressbar4')]]
progress_bar5=[[sg.ProgressBar(1000, orientation='h',style='clam',bar_color=('red','white'),size=(30, 4), key='progressbar5')]]
progress_bar6=[[sg.ProgressBar(1000, orientation='h',style='clam',bar_color=('red','white'),size=(30, 4), key='progressbar6')]]

column1=[[sg.Frame('Toxicity',progress_bar1)],
         [sg.Frame('Severe-Toxic',progress_bar2)],
         [sg.Frame('Obscene',progress_bar3)],
         [sg.Frame('Threat',progress_bar4)],
         [sg.Frame('Insult',progress_bar5)],
         [sg.Frame('Identity-Hate',progress_bar6)]]

column2=[[sg.Text('Enter Your Text :',font="Helvetica " + str(12),size=(40,0))],
        [sg.Multiline(key='input',background_color='white',text_color='black',focus=True,size=(50,4))],
        [sg.Button('', button_color=sg.TRANSPARENT_BUTTON,pad=(165,3),key='convert',image_filename='convert.png', image_subsample=2, border_width=0)],
        [sg.Text('Output Text :',font="Helvetica " + str(12),size=(40,0),key='out_text')],
        [sg.Multiline(text_color='black',background_color='white',size=(50,4),key='output')]]

layout = [
          [sg.Column(column2), sg.Column(column1,pad=(5,21))],
          [sg.Button('Exit',size=(15,1),pad=(300,5))]
          
          
         ]

window = sg.Window('Toxic Comment Detector', layout,margins=(10,10),return_keyboard_events=True,use_default_focus=True,background_color='#282923')
progress_bar1 = window['progressbar1']
progress_bar2 = window['progressbar2']
progress_bar3 = window['progressbar3']
progress_bar4 = window['progressbar4']
progress_bar5 = window['progressbar5']
progress_bar6 = window['progressbar6']

while True:  
    event, values = window.read()
    
    if event in  (None, 'Exit'):
        break
    else:
        if event=='convert':
            
            test=values['input'].lower().split()
            toxic=[]
            for i in range(len(test)):
                res=check(test[i])
                if res==True:
                    toxic.append(i)
            
            for i in toxic:
                v = nlp(test[i].lower()).vector
                c=query(v)
                test[i]=c
                
            
            window['output'].update(' '.join(test))
            
                
        if values['input'] in('','\n'):
            progress_bar1.UpdateBar(0)
            progress_bar2.UpdateBar(0)
            progress_bar3.UpdateBar(0)
            progress_bar4.UpdateBar(0)
            progress_bar5.UpdateBar(0)
            progress_bar6.UpdateBar(0)
            continue
     
        x_test_ex = t.texts_to_sequences([values['input']])
        test = pad_sequences(x_test_ex, maxlen=max_len, padding='post')
        pred=m.predict(test)   
        progress_bar1.UpdateBar((pred[0][0]*1000))
        progress_bar2.UpdateBar((pred[0][1]*1000))
        progress_bar3.UpdateBar((pred[0][2]*1000))
        progress_bar4.UpdateBar((pred[0][3]*1000))
        progress_bar5.UpdateBar((pred[0][4]*1000))
        progress_bar6.UpdateBar((pred[0][5]*1000))
        
        

window.close()


# In[ ]:




