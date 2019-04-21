import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from nlp_filter import stop_word_phrase, clean_review, ngram_func, score, add_sentiment_to_dictionary
from model import *
from keras.utils import to_categorical
import tensorflow as tf
from nltk.corpus import sentiwordnet as swn
from model import model , model_lstm, model_lstm2, model_GRU
from pyensae.sql import import_flatfile_into_database

def main():

    if os.path.exists("./model.h5") == False or os.path.exists("./model.h5") == False or os.path.exists("./tokenizer.pickle")==False:
        data = pd.read_csv("../datas/train.tsv", delimiter='\t', index_col=False, header=0)
        dictionary = pd.read_csv("../datas/datafile_fixed.tsv", delimiter='\t', index_col=False, header=0)
        #fix sentiment dictionary -> delete first col from file 
        # f = open("../datas/sentiwordnet.tsv", "r")
        # g = open("../datas/datafile_fixed.tsv", "w")

        # for line in f:
        #     if line.strip():
        #         g.write("\t".join(line.split()[1:]) + "\n")

        # f.close()
        # g.close()
                #Divise les datas en deux catégories, celles qui serviront à l'entrainaiment et celles qui permettront de tester
        #----------------------------------------------------------------------------------------------
        #creation de la colonne clean_review
        #code before saving filtered data in data_treated
        data['clean_review'] = clean_review(data.Phrase.values)
        dictionary['id_sentiment'] = add_sentiment_to_dictionary(dictionary.Pos.values, dictionary.Neg.values)
        total_sentences =  data['clean_review']
       
        #----------------------------------------------------------------------------------------------
        target_sentiment = data.Sentiment.values
        id_sentiments = dictionary['id_sentiment']
        dictionary['clean_review'] = clean_review(dictionary.Sentiment.values)
        dictionary_words = dictionary['clean_review']
        y_dic = to_categorical(id_sentiments)
        y = to_categorical(target_sentiment)
        X_train, X_test,y_train, y_test = train_test_split(total_sentences, y, test_size=0.2, stratify=y, random_state=123)
        X_dictionary,X_dictionary_test, y_train_dictionary, y_test_dictionary = train_test_split(dictionary_words, y_dic, test_size=0.2, stratify=y_dic, random_state=123)
        X_train = X_train.append(X_dictionary, ignore_index=True)
        X_test = X_test.append(X_dictionary_test, ignore_index=True)

        y_train= np.concatenate((y_train,y_train_dictionary), axis=0)
        y_test = np.concatenate((y_test,y_test_dictionary), axis=0)
        tokenizer= Tokenizer()
        tokenizer.fit_on_texts(list(total_sentences) + list(dictionary_words))
        with open('tokenizer.pickle', 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        #Padding des sentences en vecteurs de même tailles 
        #On recupère la phrase la plus longue 
        len_dictionary = max([len(s.split(' ')) for s in dictionary_words])
        max_length = max([len(s.split(' ')) for s in total_sentences])
        max_length += len_dictionary
        #On récupère le nombre total de mots
        total_size = len(tokenizer.word_index) + 1
        #Vectorisation des sentences
        X_train_tok = tokenizer.texts_to_sequences(X_train)
        X_test_tok = tokenizer.texts_to_sequences(X_test)
        X_train_norm = pad_sequences(X_train_tok, maxlen= max_length, padding = 'post')
        X_test_norm = pad_sequences(X_test_tok, maxlen= max_length, padding = 'post')
        mode = model(total_size, max_length, X_train_norm, y_train, X_test_norm, y_test)
        save_to_disk(mode, "model.json", "model.h5")
    else:
        mode = load_from_disk("model.json", "model.h5")
        print(mode)
        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
            
    #want, enthusiastic, happiest
    test1 = ['I was enthusiastic after this scene ']
    #test1 = ngram_func(test1)
    #################################################################################

    query = tokenizer.texts_to_sequences(test1)
    query = pad_sequences(query, maxlen=57)
    pred = mode.predict(query, verbose=1)
    predictions = np.round(np.argmax(pred, axis = 1)).astype(int)
    #prediction_n = score(test1)
    ##################################################################################
    # print("Prediction1 : ", predictions)
    # label = ['0', '1', '2', '3', '4']
    # if prediction_n:
    #     new_pred = label[len(label) - 1 - predictions[0]]
    #     print(new_pred)
    #     predictions = new_pred
    print("Prediction : ", predictions)
    return None

if __name__ == '__main__':
    main()


