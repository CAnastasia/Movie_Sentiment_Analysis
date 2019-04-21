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
from nltk import ngrams

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
def score(sentence):
    """Sentiment score a sentence."""
    # init sentiwordnet lookup/scoring tools
    negations = set(['not', 'n\'t', 'less', 'no', 'never',
                        'nothing', 'nowhere', 'hardly', 'barely',
                        'scarcely', 'nobody', 'none'])
    s = ' '.join(sentence)
    s = s.split()
    for p in s:
        for n in negations:
            if p == n:
                return True
    return False

def ngram_func(list_review):
    n = 3
    review_corpus=[]
    for i in range(0,len(list_review)):
        review = str(list_review[i])
        bigrams = ngrams(review.split(), n)
        for grams in bigrams:
            review = ' '.join(grams)
            review_corpus.append(review)
    print(review_corpus)
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

def add_sentiment_to_dictionary(positive, negative):
    review_corpus=[]
    for i in range(0,len(positive)):
        # if (positive[i] >= 0.5):
        #     print ((positive[i] >= 0.5).all())
        if (positive[i] == negative[i]):
            review = str(2)
        elif (negative[i] >= 0.5).any() and (positive[i] < negative[i]).any():
            review = str(0)
        elif (negative[i] < 0.5).any() and (positive[i] < negative[i]).any():
            review = str(1)
        elif (positive[i] >= 0.5).any() and (positive[i] > negative[i]).any():
            review = str(4)
        elif (positive[i] < 0.5).any() and (positive[i] > negative[i]).any():
            review = str(3)
        else:
            review = str(2)
        review_corpus.append(review)
    return review_corpus
