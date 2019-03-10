import nltk
nltk.download()
from nltk.tokenize import word_tokenize
from nltk import FreqDist
#import for lemmatization
from nltk.stem import SnowballStemmer,WordNetLemmatizer
#import for stop wods
from nltk.corpus import stopwords 
lemma=WordNetLemmatizer()
from string import punctuation
import re
#lemmatization function
def clean_review(review_col):
    review_corpus=[]
    for i in range(0,len(review_col)):
        review = str(review_col[i])
        review = re.sub('[^a-zA-Z]',' ',review)
        review = [lemma.lemmatize(w) for w in word_tokenize(str(review).lower())]
        review=' '.join(review)
        review_corpus.append(review)
    return review_corpus

#stop words function

#def stop_word_phrase(phrase):
 #   stop_words = set(stopwords.words('english'))
  #  word_tokens = word_tokenize(phrase) 