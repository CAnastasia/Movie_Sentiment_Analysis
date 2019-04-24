from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, GRU, Dropout, Flatten, Activation
from keras.layers.embeddings import Embedding
from keras.models import load_model
from keras.models import Sequential,model_from_json

def model(total_size, max_length, X_train_norm, y_train, X_test_norm, y_test):
    Embedding_Dim = 100
    class_dim = 5
    print('Building model...')
    model = Sequential()
    model.add(Embedding(total_size,Embedding_Dim,input_length = max_length))
    model.add(Dropout(0.3))
    model.add(GRU(units=32,dropout=0.2, recurrent_dropout=0.2))
    #model.add(Flatten())
    model.add(Dense(8, input_dim=max_length, activation='relu'))
    model.add(Dense(5, activation = 'sigmoid'))
    #model.add(Dense(1, activation='linear')) 
   
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print("train ... ")
    model.fit(X_train_norm, y_train,batch_size=32,epochs=3,validation_data=(X_test_norm,y_test), verbose=2)
    #tester le model    
     #Prediction 
    return model

def save_to_disk(model, filejs, fileh5):
    model_json = model.to_json()
    with open(filejs, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(fileh5)
    print("Saved model to disk")

def load_from_disk(filejs, fileh5):
    json_file = open(filejs, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(fileh5)
    print("Loaded model from disk")
    loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return loaded_model
    