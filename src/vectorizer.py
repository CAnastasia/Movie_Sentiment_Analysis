import pandas as pd 

train = pd.DataFrame()
train = pd.read_csv('../datas/train.tsv',delimiter='\t',encoding='utf-8')
test = pd.read_csv('../datas/test.tsv', delimiter='\t', encoding='utf-8') 

all_corpus   = list(train['Phrase'].values) + list(test['Phrase'].values)
train_phrases  = list(train['Phrase'].values) 
test_phrases   = list(test['Phrase'].values)
X_train_target_binary = pd.get_dummies(train['Sentiment'])

from keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


vocab_size = 20000  # based on words in the entire corpus
max_len = 60
tokenizer = Tokenizer(num_words=vocab_size, lower=True, filters='\n\t')
tokenizer.fit_on_texts(all_corpus)
encoded_train_phrases = tokenizer.texts_to_sequences(train_phrases)
encoded_test_phrases = tokenizer.texts_to_sequences(test_phrases)

X_train_words = sequence.pad_sequences(encoded_train_phrases, maxlen=max_len,  padding='post')
X_test_words = sequence.pad_sequences(encoded_test_phrases, maxlen=max_len,  padding='post')

import io

def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return data

#tab = load_vectors("../datas/wiki-news-300d-1M.vec")