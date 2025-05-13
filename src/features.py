from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import string
import re

nltk.download('stopwords')
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words('english'))

def preprocess_text(text: str) -> str:
    """
    Basic text cleaning: lowercase, remove punctuation, stopword cleaning.
    """
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS]
    return " ".join(tokens)

def vectorize_text(corpus, max_features=5000):
    """
    Vectorizes texts with TF-IDF.
    """
    tfidf = TfidfVectorizer(max_features=max_features)
    X = tfidf.fit_transform(corpus)
    return X, tfidf