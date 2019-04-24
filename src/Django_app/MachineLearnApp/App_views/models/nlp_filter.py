import nltk
#nltk.download()
#nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk import FreqDist
#nltk.download('wordnet')
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
        #review = str(stop_word_phrase(review))
        review_corpus.append(review)
    return review_corpus

#stop words function
#function incomplete
def stop_word_phrase(review_col):
    filtered_sentence = []
    stop_words = set(stopwords.words('english'))
    #filtered_sentence = [w for w in word_tokens if not w in stop_words]
    filtered_sentence = [w for w in word_tokenize(review_col) if not w in stop_words]
    for w in  word_tokenize(review_col): 
        if w not in stop_words: 
            filtered_sentence.append(w)
    return filtered_sentence
