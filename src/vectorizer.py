from sklearn.feature_extraction.text import CountVectorizer

import csv

with open("../datas/train.tsv", 'r') as G:
    print(G.head(10))

#with open("../datas/test.tsv", 'r') as D: 
#    print(D.read)
