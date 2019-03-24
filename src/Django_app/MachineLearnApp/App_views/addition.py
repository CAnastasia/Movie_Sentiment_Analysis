import sys
import os
from datetime import datetime
from django.shortcuts import render
from django.http import HttpResponse
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras.models import Sequential,model_from_json
import pickle
import os

path = "../"
print(os.listdir("./"))

def date_actuelle(request):
    return render(request, 'MachineLearn/date.html', {'date': datetime.now()})

def addition(request, nombre1, nombre2):    
    total = nombre1 + nombre2
    return HttpResponse(total)
    #return render(request, 'MachineLearn/addition.html', locals())

def state(request):
    return render(request, 'MachineLearn/state.html')

def feature(request):
    return render(request, 'MachineLearn/feature.html')

def models(request):
    return render(request, 'MachineLearn/models.html')

def submission(request):
    return render(request, 'MachineLearn/submission.html')

def description(request):
    return render(request, 'MachineLearn/description.html')

def MachineLean(request):
    return render(request, 'MachineLearn/MachineLearn.html')

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

def model(request, sentence):
    mode = load_from_disk(path+"model.json", path+"model.h5")
    with open(path+'tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    query = tokenizer.texts_to_sequences(sentence)
    query = pad_sequences(query, maxlen=48)
    prediction = mode.predict_classes(x=query)
    return return render(request, 'MachineLearn/models.html', locals())