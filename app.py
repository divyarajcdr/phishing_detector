import streamlit as st
from PIL import Image

# --- Page Navigation and UI ---

st.set_page_config(page_title="Phishing Detector - TEST", page_icon="🧪", layout="wide")

# Initialize session state for page navigation
if 'page' not in st.session_state:
    st.session_state.page = 'home'

# Functions to change the page
def go_to_analyzer():
    st.session_state.page = 'analyzer'
def go_to_home():
    st.session_state.page = 'home'

# Display content based on the current page
if st.session_state.page == 'home':
    st.title("🛡️ Welcome to the Intelligent Phishing Detector")
    st.markdown("---")
    st.header("Team Name: [Your Team Name]")
    st.subheader("Team Members:")
    st.markdown("""
    * [Member 1 Name]
    * [Member 2 Name]
    * [Member 3 Name]
    """)
    st.markdown("---")
    st.button("Proceed to Analyzer", on_click=go_to_analyzer, type="primary")

elif st.session_state.page == 'analyzer':
    st.title("📧 Phishing Email Analyzer - TEST MODE")
    st.info("Model loading is currently disabled for this test.")
    st.button("← Back to Home", on_click=go_to_home)
    
    tab1, tab2 = st.tabs(["Paste Email Text", "Upload Image"])

    with tab1:
        st.header("Analyze Pasted Text")
        text_area = st.text_area("Paste the full email text here...", height=250)
        if st.button("Classify Text"):
            st.success("Test successful! The UI is working.")

    with tab2:
        st.header("Analyze from an Image")
        uploader = st.file_uploader("Upload a screenshot of the email (.png, .jpg, .jpeg)", type=['png', 'jpg', 'jpeg'])
        if uploader is not None:
            st.success("Test successful! Image upload is working.")
