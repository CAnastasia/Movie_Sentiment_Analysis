import pandas as pd
from random import randint

def get_random_sentence():
    train_path = '../datas/train.tsv'
    train_data = pd.read_csv(train_path, sep="\t")
    rnd_id = randint(1, train_data['SentenceId'].max())
    random_sample = train_data.loc[train_data['SentenceId'] == rnd_id].iloc[0].loc['Phrase']
    return random_sample
