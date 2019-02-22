import pandas as pd
import numpy as np

df = pd.DataFrame()
df = pd.read_csv('train.tsv', encoding ='utf-8')

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop("Sentiment", axis=1), df["sentiment"])