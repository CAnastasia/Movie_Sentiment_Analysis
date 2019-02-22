import pandas as pd
import numpy as np

data = pd.read_csv("../datas/train.tsv", delimiter='\t')


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data.drop("Sentiment", axis=1), data["Sentiment"])
#Divise les datas en deux catégories, celles qui serviront à l'entrainaiment et celles qui permettront de tester

from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

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
from keras.layers import Dense, Embedding, LSTM, GRU
from keras.layers.embeddings import Embedding

Embedding_Dim = 100
print('Building model...')

model = Sequential()
model.add(Embedding(total_size,Embedding_Dim,input_length= max_length))
model.add(GRU(units=32,dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(X_train.shape)
print("train ... ")
#model.fit(X_train_norm, y_train,batch_size=128,epochs=25,validation_data=(X_test_norm,y_test), verbose=2)