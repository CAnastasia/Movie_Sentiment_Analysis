import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from nlp_filter.py import clean_review
def main():
    data = pd.read_csv("../datas/train.tsv", delimiter='\t')
    X_train, X_test, y_train, y_test = train_test_split(data.drop("Sentiment", axis=1), data["Sentiment"])
    #Divise les datas en deux catégories, celles qui serviront à l'entrainaiment et celles qui permettront de tester
    #print(X_train.shape)
    #print(X_test.shape)
    #print(data.shape)
    X_train=X_train['Phrase']
    X_test=X_test['Phrase']
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

    X_train_norm = pad_sequences(X_train_tok, maxlen= max_length, padding = 'post')
    X_test_norm = pad_sequences(X_test_tok, maxlen= max_length, padding = 'post')

    #print(X_test_norm)
    return None

if __name__ == '__main__':
    main()


