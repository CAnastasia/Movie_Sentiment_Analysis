from sklearn.feature_extraction.text import CountVectorizer

import csv

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #for vizualization of data
from tqdm import tqdm #for progress information
tqdm.pandas()

#Packages for Preprocessing Data and building the embedding layer
from sklearn.model_selection import train_test_split #for splitting the data into train and into test set
from keras.preprocessing.text import Tokenizer #for preprocessing the data
from keras.preprocessing.sequence import pad_sequences #for preprocessing the data

#Packages for building the Neural Network: Keras Packages.
import keras.backend as K #to use math functions like "keras.backend.sum"
from keras.models import Model #to build the Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation,Bidirectional, CuDNNGRU # the layers we will use
from keras.layers.embeddings import Embedding #to create an Embedding, more in chapter 3
from keras.engine.topology import Layer #to create the attention layer
from keras import initializers, regularizers, constraints #used in attention layer

from keras.preprocessing import sequence #scheint bei mir dafür zu sorgen, dass Layers richtig geordnet sind. Keine Ahnung warum.

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

print(os.listdir("../datas"))
#else print(os.listdir("./datas"))
import gc
pd.options.display.max_colwidth=100
pd.set_option('display.max_colwidth',100)

train=pd.read_csv('../datas/train.tsv',sep='\t')
print(train.shape)
print(train.head(100))