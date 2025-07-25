import streamlit as st
import pickle
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from PIL import Image
import pytesseract

# --- Pre-loading and Function Definitions ---

# This function will run once and cache the resources
@st.cache_resource
def load_resources():
    """Load the saved model, vectorizer, and NLTK data."""
    # Load model and vectorizer
    try:
        with open('phishing_detector_model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
            vectorizer = pickle.load(vectorizer_file)
    except FileNotFoundError:
        return None, None # Return None if files are not found

    # Download NLTK data if not already present
    try:
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('corpora/stopwords')
    except nltk.downloader.DownloadError:
        nltk.download('stopwords', quiet=True)
    try:
        nltk.data.find('corpora/wordnet')
    except nltk.downloader.DownloadError:
        nltk.download('wordnet', quiet=True)
    
    return model, vectorizer

# Call the function to load everything
loaded_model, loaded_vectorizer = load_resources()

# Define the preprocessing function (must be the same as in training)
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower() # Lowercase
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) # Remove URLs
    text = re.sub(r'<.*?>', '', text) # Remove HTML tags
    text = text.translate(str.maketrans('', '', string.punctuation)) # Remove punctuation
    text = re.sub(r'\d+', '', text) # Remove numbers
    tokens = nltk.word_tokenize(text) # Tokenize
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in
