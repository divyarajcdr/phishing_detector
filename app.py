import streamlit as st
import pickle
import os
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import pytesseract
from PIL import Image
import io
import scipy.sparse

# --- Set Streamlit page configuration FIRST ---
# This must be the very first Streamlit command in your script.
st.set_page_config(page_title="Intelligent Phishing Detector", page_icon="🛡️", layout="wide")

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
        # If model files are not found, return None to indicate an error
        return None, None 

    # Download NLTK data (only if not already downloaded)
    # These downloads are quiet to avoid verbose output in the Streamlit app
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    
    return model, vectorizer

# Call the function to load everything when the app starts
loaded_model, loaded_vectorizer = load_resources()

# Initialize NLTK components for preprocessing
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """
    Cleans and preprocesses the input text.
    Steps include:
    - Removing URLs
    - Removing HTML tags
    - Removing non-alphabetic characters
    - Converting text to lowercase
    - Tokenization
    - Lemmatization
    - Stop word removal
    """
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove non-alphabetic characters (keep spaces)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert text to lowercase
    text = text.lower()
    # Tokenize the text into individual words
    tokens = nltk.word_tokenize(text)
    # Lemmatize tokens and remove stop words
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    # Join the processed tokens back into a single string
    return " ".join(lemmatized_tokens)

# --- Main Classification Function ---
def classify_content(text_content):
    """
    Preprocesses the given text content, vectorizes it, and uses the loaded model
    to predict if it's legitimate or phishing. Displays the result and confidence.
    """
    if not text_content.strip():
        st.warning("⚠️ Please provide some text to analyze.")
        return

    # 1. Preprocess the text content using the defined function
    processed_text = preprocess_text(text_content)
    
    # 2. Vectorize the processed text using the loaded TF-IDF vectorizer
    # The vectorizer expects a list of strings, so we pass [processed_text]
    vectorized_text = loaded_vectorizer.transform([processed_text])

    # 3. Manually add the 'urls' feature. 
    # In this simplified app, we assume 0 URLs for any new input.
    # This must match the feature engineering done during model training.
    urls_feature = scipy.sparse.csr_matrix([[0]])

    # 4. Combine the vectorized text features with the 'urls' feature.
    # This creates a sparse matrix that matches the input shape the model expects.
    combined_features = scipy.sparse.hstack([vectorized_text, urls_feature])

    # 5. Predict using the loaded model
    prediction = loaded_model.predict(combined_features)[0]
    # Get prediction probabilities for confidence score
    prediction_proba = loaded_model.predict_proba(combined_features)[0]

    # Display results to the user
    st.write("---") # Separator line
    st.subheader("Analysis Result")
    if prediction == 1: # Assuming 1 means phishing
        confidence = prediction_proba[1] # Confidence for the phishing class
        st.error(f"Result: PHISHING 🎣 (Confidence: {confidence:.2%})")
        st.info("This content exhibits characteristics commonly found in phishing attempts. Exercise extreme caution.")
    else: # Assuming 0 means legitimate
        confidence = prediction_proba[0] # Confidence for the legitimate class
        st.success(f"Result: LEGITIMATE ✅ (Confidence: {confidence:.2%})")
        st.info("This content appears legitimate. However, always remain vigilant with suspicious emails.")

# --- Page Navigation and UI ---

# Initialize session state for page navigation. 'home' is the default page.
if 'page' not in st.session_state:
    st.session_state.page = 'home'

# Functions to change the page state
def go_to_analyzer():
    """Sets the session state to 'analyzer' page."""
    st.session_state.page = 'analyzer'

def go_to_home():
    """Sets the session state to 'home' page."""
    st.session_state.page = 'home'

# Display content based on the current page
if st.session_state.page == 'home':
    # Home Page content
    st.title("🛡️ Welcome to the Intelligent Phishing Detector")
    st.markdown("---")
    
    # --- IMPORTANT: FILL IN YOUR DETAILS HERE ---
    st.header("Team Name: PyRates")
    st.subheader("Team Members: ")
    st.markdown("""
    * DIVYARAJ RAJPUROHIT - 24BCY10311
    * HARDIK JAIN - 24BAI10355
    * MISHITA TIWARI - 24BAI10204
    * VINITA SHARMA - 24BSA10281
    * PRIYAMVADA TIWARI - 24BCE11480
    """)
    st.markdown("---")
    
    # Button to navigate to the analyzer page
    st.button("Proceed to Analyzer", on_click=go_to_analyzer, type="primary")

elif st.session_state.page == 'analyzer':
    # Analyzer Page content
    # Check if model files were loaded successfully
    if loaded_model is None or loaded_vectorizer is None:
        st.error("❌ Error: Model files not found. Please ensure 'phishing_detector_model.pkl' and 'tfidf_vectorizer.pkl' are in the same folder as this app.")
        st.button("Go Back to Home", on_click=go_to_home)
    else:
        st.title("📧 Phishing Email Analyzer")
        # Button to navigate back to the home page
        st.button("← Back to Home", on_click=go_to_home)
        
        # Create tabs for different analysis methods
        tab1, tab2 = st.tabs(["Paste Email Text", "Upload Image"])

        with tab1:
            st.header("Analyze Pasted Text")
            # Text area for user to paste email content
            text_area = st.text_area("Paste the full email text here...", height=250, placeholder="e.g., 'Dear customer, your account has been suspended. Click here to verify.'")
            if st.button("Classify Text"):
                classify_content(text_area)

        with tab2:
            st.header("Analyze from an Image")
            # File uploader for image analysis
            uploader = st.file_uploader("Upload a screenshot of the email (.png, .jpg, .jpeg)", type=['png', 'jpg', 'jpeg'])
            if uploader is not None:
                try:
                    # Attempt to extract text from the uploaded image using Tesseract OCR
                    image = Image.open(uploader)
                    # Display the uploaded image (optional, for user confirmation)
                    st.image(image, caption='Uploaded Image', use_column_width=True)
                    
                    # Perform OCR
                    extracted_text = pytesseract.image_to_string(image)
                    
                    st.write("### Extracted Text:")
                    # Display the extracted text (disabled for editing, just for review)
                    st.text_area("Text found in image:", value=extracted_text, height=150, disabled=True)
                    
                    # Classify the extracted text
                    if st.button("Classify Image Text"): # New button for clarity
                        classify_content(extracted_text)
                except pytesseract.TesseractNotFoundError:
                    st.error("Tesseract is not installed or not in your PATH. Please install it to use image analysis. Refer to the Streamlit documentation or Tesseract's GitHub for installation instructions.")
                except Exception as e:
                    st.error(f"An error occurred during image processing: {e}. Please try another image.")
