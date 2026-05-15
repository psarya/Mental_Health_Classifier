
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="Mental Health Classifier",
    page_icon="🧠",
    layout="wide",
)

# ==============================
# CUSTOM CSS
# ==============================
st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

h1, h2, h3 {
    color: white;
}

.stButton button {
    width: 100%;
    border-radius: 12px;
    height: 50px;
    font-size: 18px;
    font-weight: bold;
    background-color: #6C63FF;
    color: white;
}

.stTextArea textarea {
    border-radius: 12px;
    font-size: 16px;
}

.result-box {
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 10px;
}

</style>
""", unsafe_allow_html=True)

# ==============================
# NLTK DOWNLOADS
# ==============================
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# ==============================
# TEXT CLEANING
# ==============================
STOP_WORDS = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

MENTAL_HEALTH_STOPWORDS = {
    'feel', 'like', 'im', 'ive', 'dont', 'cant', 'just',
    'get', 'really', 'know', 'think', 'time', 'one', 'day',
    'want', 'make', 'go', 'going', 'got', 'even', 'way',
}

def clean_text(text):
    if not isinstance(text, str) or not text.strip():
        return ""

    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)

    tokens = text.split()

    tokens = [
        lemmatizer.lemmatize(t)
        for t in tokens
        if t not in STOP_WORDS
        and t not in MENTAL_HEALTH_STOPWORDS
        and len(t) > 2
    ]

    return ' '.join(tokens)

# ==============================
# LOAD MODELS
# ==============================
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

# ==============================
# SIDEBAR
# ==============================
with st.sidebar:

    st.title("🧠 About")

    st.info("""
    Mental Health Text Classification System

    Models Used:
    - SVM
    - Logistic Regression

    NLP Pipeline:
    TF-IDF + Text Preprocessing
    """)

    st.warning("""
    ⚠️ This tool is for educational purposes only.

    It is NOT a clinical diagnostic system.
    """)

    st.markdown("---")

    st.markdown("### Example Inputs")

    st.code("I feel hopeless and tired")
    st.code("I am anxious all the time")
    st.code("I want to end my life")
    st.code("Today was a beautiful day")

# ==============================
# HERO SECTION
# ==============================
st.markdown("""
<h1 style='text-align:center;'>
🧠 Mental Health Status Classifier
</h1>

<p style='text-align:center; font-size:20px; color:lightgray;'>
AI-powered NLP system for social media mental health classification
</p>
""", unsafe_allow_html=True)

st.markdown("---")

# ==============================
# INPUT
# ==============================
user_input = st.text_area(
    "Enter social media text:",
    height=180,
    placeholder="Type a sentence here..."
)

# ==============================
# PREDICTION
# ==============================
if st.button("🔍 Analyze Text"):

    if user_input.strip() == "":
        st.warning("Please enter text.")
    else:

        with st.spinner("Analyzing mental health patterns..."):

            clean_text_input = clean_text(user_input)

            if clean_text_input.strip() == "":
                st.warning("Input became empty after preprocessing.")
            else:

                text_tfidf = tfidf.transform([clean_text_input])

                # ======================
                # SVM
                # ======================
                svm_pred_encoded = svm_model.predict(text_tfidf)[0]
                svm_pred_proba = svm_model.predict_proba(text_tfidf)[0]
                svm_pred_class = label_encoder.inverse_transform(
                    [svm_pred_encoded]
                )[0]

                svm_conf = np.max(svm_pred_proba) * 100

                # ======================
                # LR
                # ======================
                lr_pred_encoded = lr_model.predict(text_tfidf)[0]
                lr_pred_proba = lr_model.predict_proba(text_tfidf)[0]
                lr_pred_class = label_encoder.inverse_transform(
                    [lr_pred_encoded]
                )[0]

                lr_conf = np.max(lr_pred_proba) * 100

                st.markdown("## 📊 Classification Results")

                col1, col2 = st.columns(2)

                # ======================
                # SVM COLUMN
                # ======================
                with col1:

                    st.markdown("### 🧠 SVM Prediction")

                    if svm_pred_class == "Suicidal":
                        st.error(f"Prediction: {svm_pred_class}")

                    elif svm_pred_class == "Depression":
                        st.warning(f"Prediction: {svm_pred_class}")

                    elif svm_pred_class == "Anxiety":
                        st.info(f"Prediction: {svm_pred_class}")

                    else:
                        st.success(f"Prediction: {svm_pred_class}")

                    st.metric(
                        label="Confidence",
                        value=f"{svm_conf:.2f}%"
                    )

                    svm_df = pd.DataFrame({
                        "Class": CLASSES,
                        "Probability": svm_pred_proba
                    }).set_index("Class")

                    st.bar_chart(svm_df)

                # ======================
                # LR COLUMN
                # ======================
                with col2:

                    st.markdown("### 🤖 Logistic Regression")

                    if lr_pred_class == "Suicidal":
                        st.error(f"Prediction: {lr_pred_class}")

                    elif lr_pred_class == "Depression":
                        st.warning(f"Prediction: {lr_pred_class}")

                    elif lr_pred_class == "Anxiety":
                        st.info(f"Prediction: {lr_pred_class}")

                    else:
                        st.success(f"Prediction: {lr_pred_class}")

                    st.metric(
                        label="Confidence",
                        value=f"{lr_conf:.2f}%"
                    )

                    lr_df = pd.DataFrame({
                        "Class": CLASSES,
                        "Probability": lr_pred_proba
                    }).set_index("Class")

                    st.bar_chart(lr_df)

                st.markdown("---")

                st.warning("""
                ⚠️ Predictions may vary between models.

                This system should not replace professional mental health support.
                """)

# ==============================
# FOOTER
# ==============================
st.markdown("""
---
<div style='text-align:center; color:gray;'>

Built using Streamlit, NLP, TF-IDF, SVM, and Logistic Regression

</div>
""", unsafe_allow_html=True)
