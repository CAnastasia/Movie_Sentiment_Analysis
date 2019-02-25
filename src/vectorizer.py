import pandas as pd
import numpy as np

data = pd.read_csv("../datas/train.tsv", delimiter='\t')


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data.drop("Sentiment", axis=1), data["Sentiment"])
#Divise les datas en deux catégories, celles qui serviront à l'entrainaiment et celles qui permettront de tester

from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

X_train=X_train['Phrase']
X_test=X_test['Phrase']

import nltk
from nltk.tokenize import word_tokenize
from nltk import FreqDist
from nltk.stem import SnowballStemmer,WordNetLemmatizer
#Téléchargement des packages nltk
#ntlk.downlowd() 
lemma=WordNetLemmatizer()
from string import punctuation
import re

def clean_review(review_col):
    review_corpus=[]
    for i in range(0,len(review_col)):
        review=str(review_col[i])
        review=re.sub('[^a-zA-Z]',' ',review)
        review=[lemma.lemmatize(w) for w in word_tokenize(str(review).lower())]
        review=' '.join(review)
        review_corpus.append(review)
    return review_corpus

#creation de la colonne clean_review
data['clean_review']=clean_review(data.Phrase.values)
print(data['clean_review'])

tokenizer= Tokenizer()
total_sentences =  data['Phrase']
tokenizer.fit_on_texts(total_sentences)

#Padding des sentences en vecteurs de même tailles 
#On recupère la phrase la plus longue 
max_length = max([len(s.split(' ')) for s in total_sentences])
#On récupère le nombre total de mots
total_size = len(tokenizer.word_index) + 1

#Vectorisation des sentences
X_train_tok = tokenizer.texts_to_sequences(X_train)
X_test_tok = tokenizer.texts_to_sequences(X_test)

#Normalisation des tokens
X_train_norm = pad_sequences(X_train_tok, maxlen= max_length, padding = 'post')
X_test_norm = pad_sequences(X_test_tok, maxlen= max_length, padding = 'post')

#Implémentation du modèle LSTM 

from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, GRU, Dropout
from keras.layers.embeddings import Embedding

Embedding_Dim = 100
print('Building model...')

model = Sequential()
model.add(Embedding(total_size,Embedding_Dim,input_length= max_length))
model.add(Dropout(0.3))
model.add(GRU(units=32,dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation = 'sigmoid'))
model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print("train ... ")
model.fit(X_train_norm, y_train,batch_size=128,epochs=8,validation_data=(X_test_norm,y_test), verbose=2)
#Ce type de model me retourne une accuracy d'environ 0.17, ce qui est insuffisant
# Nous allons donc essayer 

