from sklearn.feature_extraction.text import CountVectorizer

import csv

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../datas"))
import gc
pd.options.display.max_colwidth=100
pd.set_option('display.max_colwidth',100)

train=pd.read_csv('../datas/train.tsv',sep='\t')
print(train.shape)
print(train.head(100))