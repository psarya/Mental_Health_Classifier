
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load NLTK data if not already downloaded (for deployment)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# --- Text Cleaning Function (from notebook) ---
STOP_WORDS  = set(stopwords.words('english'))
lemmatizer  = WordNetLemmatizer()

MENTAL_HEALTH_STOPWORDS = {
    'feel', 'like', 'im', 'ive', 'dont', 'cant', 'just',
    'get', 'really', 'know', 'think', 'time', 'one', 'day',
    'want', 'make', 'go', 'going', 'got', 'even', 'way',
}

def clean_text(text):
    """Full NLP cleaning pipeline matching production preprocessing."""
    if not isinstance(text, str) or not text.strip():
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)          # remove URLs
    text = re.sub(r'@\w+|#\w+', '', text)               # mentions/hashtags
    text = re.sub(r'\d+', '', text)                     # remove numbers
    text = re.sub(r'[^a-z\s]', ' ', text)               # keep letters only
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens
              if t not in STOP_WORDS
              and t not in MENTAL_HEALTH_STOPWORDS
              and len(t) > 2]
    return ' '.join(tokens)

# --- Load Models and Artifacts ---
@st.cache_resource
def load_models():
    with open("models/svm_model.pkl", "rb") as f:
        svm_model = pickle.load(f)
    with open("models/lr_model.pkl", "rb") as f:
        lr_model = pickle.load(f)
    with open("models/tfidf.pkl", "rb") as f:
        tfidf = pickle.load(f)
    with open("models/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    with open("models/meta.json", "r") as f:
        meta = json.load(f)
    return svm_model, lr_model, tfidf, label_encoder, meta

svm_model, lr_model, tfidf, label_encoder, meta = load_models()
CLASSES = meta['classes']

# --- Streamlit App ---
st.set_page_config(
    page_title="Mental Health Classifier",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.title("🧠 Mental Health Status Classifier")
st.markdown("Classify social media text into 'Anxiety', 'Depression', 'Suicidal', or 'Normal'.")

user_input = st.text_area("Enter your text here:", height=150)

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter some text to classify.")
    else:
        st.info("Classifying...")

        # Preprocess text
        clean_text_input = clean_text(user_input)
        if clean_text_input.strip() == "":
            st.warning("The input text became empty after cleaning. Please try different text.")
        else:
            # TF-IDF transform
            text_tfidf = tfidf.transform([clean_text_input])

            # Predict with SVM
            svm_pred_encoded = svm_model.predict(text_tfidf)[0]
            svm_pred_proba   = svm_model.predict_proba(text_tfidf)[0]
            svm_pred_class   = label_encoder.inverse_transform([svm_pred_encoded])[0]

            # Predict with Logistic Regression
            lr_pred_encoded  = lr_model.predict(text_tfidf)[0]
            lr_pred_proba    = lr_model.predict_proba(text_tfidf)[0]
            lr_pred_class    = label_encoder.inverse_transform([lr_pred_encoded])[0]

            st.subheader("Classification Results")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### SVM Prediction")
                st.success(f"Predicted Class: **{svm_pred_class}**")
                st.write("Probabilities:")
                svm_proba_df = pd.DataFrame({
                    'Class': CLASSES,
                    'Probability': svm_pred_proba
                }).set_index('Class')
                st.bar_chart(svm_proba_df)

            with col2:
                st.markdown("#### Logistic Regression Prediction")
                st.success(f"Predicted Class: **{lr_pred_class}**")
                st.write("Probabilities:")
                lr_proba_df = pd.DataFrame({
                    'Class': CLASSES,
                    'Probability': lr_pred_proba
                }).set_index('Class')
                st.bar_chart(lr_proba_df)

            st.markdown("**Note:** SVM and Logistic Regression models might give slightly different results. For critical applications, consider ensemble methods or expert review.")

st.markdown("""
---
This app uses TF-IDF features with SVM and Logistic Regression models to classify text.
Models were trained on a dataset of social media posts.
""")
