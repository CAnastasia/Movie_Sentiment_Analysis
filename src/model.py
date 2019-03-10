from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, GRU, Dropout
from keras.layers.embeddings import Embedding
from keras.models import load_model

Embedding_Dim = 100
print('Building model...')
#Implémentation du modèle LSTM 

model = Sequential()
model.add(Embedding(total_size,32,input_length = max_length))
model.add(LSTM(100))
#model.add(Dropout(0.3))
#model.add(GRU(units=32,dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation = 'sigmoid'))
model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print("train ... ")
model.fit(X_train_norm, y_train,batch_size=64,epochs=3,validation_data=(X_test_norm,y_test), verbose=2)

scores = model.evaluate(X_test_norm, y_test, verbose=0)
print("Accuracy:",scores[1])