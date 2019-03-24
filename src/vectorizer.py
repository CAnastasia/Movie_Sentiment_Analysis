import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from nlp_filter import stop_word_phrase, clean_review
from model import *
from keras.utils import to_categorical
import tensorflow as tf


def main():

    if os.path.exists("./model.h5") == False or os.path.exists("./model.h5") == False or os.path.exists("./tokenizer.pickle")==False:
        data = pd.read_csv("../datas/train.tsv", delimiter='\t')
        #Divise les datas en deux catégories, celles qui serviront à l'entrainaiment et celles qui permettront de tester
        #----------------------------------------------------------------------------------------------
        #creation de la colonne clean_review
        #code before saving filtered data in data_treated
        data['clean_review'] = clean_review(data.Phrase.values)
        total_sentences =  data['clean_review']
        # import csv
        # with open("data_treated.csv", "w") as outfile:
        #     outfile.write("clean_review\n")
        #     for entries in total_sentences:
        #       s1 = entries.strip('[]')
        #         s2 = s1.replace(",","")
        #         s3 = s2.replace("'", "")
        #         if s3:
        #             outfile.write(s3)
        #             outfile.write("\n")
        #----------------------------------------------------------------------------------------------
        target_sentiment = data.Sentiment.values
        y = to_categorical(target_sentiment)
        X_train, X_test,y_train, y_test = train_test_split(total_sentences, y, stratify=y)
        tokenizer= Tokenizer()
        
        print(y_train.shape)
        print(X_train.shape)
        tokenizer.fit_on_texts(list(total_sentences))
        with open('tokenizer.pickle', 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

                #Padding des sentences en vecteurs de même tailles 
            #On recupère la phrase la plus longue 
        max_length = max([len(s.split(' ')) for s in total_sentences])
            #On récupère le nombre total de mots
        total_size = len(tokenizer.word_index) + 1
                
        

            #Vectorisation des sentences
        X_train_tok = tokenizer.texts_to_sequences(X_train)
        X_test_tok = tokenizer.texts_to_sequences(X_test)
        
        X_train_norm = pad_sequences(X_train_tok, maxlen= max_length, padding = 'post')
        X_test_norm = pad_sequences(X_test_tok, maxlen= max_length, padding = 'post')
        mode = model(total_size, max_length, X_train_norm, y_train, X_test_norm, y_test)
        print("Test phrase :")
        test = pd.read_csv("test.tcv", delimiter='\t')
        print(test)
        query = tokenizer.texts_to_sequences(test['test'])
        query = pad_sequences(query, maxlen=48)
        pred = mode.predict(query, verbose=1)
        predictions = np.round(np.argmax(pred, axis=1)).astype(int)

        print("Prediction : ", predictions)
        save_to_disk(mode, "model.json", "model.h5")
    else:
        mode = load_from_disk("model.json", "model.h5")
        print(mode)
        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
    print("Test phrase :")
    test = pd.read_csv("test.tcv", delimiter='\t')
    print(test)
    query = tokenizer.texts_to_sequences(test['test'])
    query = pad_sequences(query, maxlen=48)
    pred = mode.predict(query, verbose=1)
    predictions = np.round(np.argmax(pred, axis=1)).astype(int)

    print("Prediction : ", predictions)
    return None

if __name__ == '__main__':
    main()


