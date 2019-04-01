import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from nlp_filter import stop_word_phrase, clean_review, ngram_func, score
from model import *
from keras.utils import to_categorical
import tensorflow as tf
from nltk.corpus import sentiwordnet as swn
from model import model , model_lstm

def main():

    if os.path.exists("./model.h5") == False or os.path.exists("./model.h5") == False or os.path.exists("./tokenizer.pickle")==False:
        data = pd.read_csv("../datas/train.tsv", delimiter='\t', index_col=False, header=0)
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
        X_train, X_test,y_train, y_test = train_test_split(total_sentences, y, test_size=0.2, stratify=y, random_state=123)
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
        mode = model_lstm(total_size, max_length, X_train_norm, y_train, X_test_norm, y_test)
        save_to_disk(mode, "model.json", "model.h5")
    else:
        mode = load_from_disk("model.json", "model.h5")
        print(mode)
        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)

    #test = pd.read_csv("test.tcv", delimiter='\t')
    #query = tokenizer.texts_to_sequences(test['test'])
    #want, enthusiastic, happiest
    test1 = ['As I said, there are things to admire and enjoy about “The Dark Knight,” but they ultimately get swept aside by the film’s pretentious ambitions. The human scenes – between Bale and Hathaway; Bruce Wayne and Michael Caine’s Alfred; or between Gary Oldman’s Commissioner Gordon and Joseph Gordon Levitt as a young cop who becomes his protégé – demonstrate what this movie could have been, if Nolan had made it as a drama instead of a dirigible. But hot air rules.']
    #test1 = ngram_func(test1)
    query = tokenizer.texts_to_sequences(test1)
    query = pad_sequences(query, maxlen=48)
    pred = mode.predict(query, verbose=1)
    predictions = np.round(np.argmax(pred, axis = 1)).astype(int)
    prediction_n = score(test1)
    print(prediction_n)
    print("Prediction1 : ", predictions)
    label = ['0', '1', '2', '3', '4']
    if prediction_n:
        new_pred = label[len(label) - 1 - predictions[0]]
        print(new_pred)
        predictions = new_pred
    print("Prediction : ", predictions)
    return None

if __name__ == '__main__':
    main()


