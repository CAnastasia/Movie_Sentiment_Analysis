from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, GRU, Dropout
from keras.layers.embeddings import Embedding
from keras.models import load_model

def model(total_size, max_length, X_train_norm, y_train, X_test_norm, y_test):
    Embedding_Dim = 100
    class_dim = 5
    print('Building model...')
    model = Sequential()
    model.add(Embedding(total_size,Embedding_Dim,input_length = max_length))
    model.add(Dropout(0.3))
    model.add(GRU(units=32,dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(5, activation = 'sigmoid'))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print("train ... ")
    model.fit(X_train_norm, y_train,batch_size=128,epochs=3,validation_data=(X_test_norm,y_test), verbose=2)
