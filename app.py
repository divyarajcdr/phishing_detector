# --- Main Classification Function --- CORRECTED ---
def classify_content(text_content):
    """Preprocesses and classifies the given text, showing results."""
    if not text_content.strip():
        st.warning("⚠️ Please provide some text to analyze.")
        return

    # 1. Preprocess the input text
    processed_text = preprocess_text(text_content)

    # 2. Vectorize the processed text using the loaded vectorizer
    vectorized_text = loaded_vectorizer.transform([processed_text])

    # 3. Predict using the model with the correct features
    prediction = loaded_model.predict(vectorized_text)[0]
    prediction_proba = loaded_model.predict_proba(vectorized_text)[0]

    # Display results
    st.write("---")
    st.subheader("Analysis Result")
    if prediction == 1:
        confidence = prediction_proba[1]
        st.error(f"Result: PHISHING 🎣 (Confidence: {confidence:.2%})")
    else:
        confidence = prediction_proba[0]
        st.success(f"Result: LEGITIMATE ✅ (Confidence: {confidence:.2%})")
