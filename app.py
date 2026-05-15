
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
    layout="centered"
)

# ==============================
# SOFT DARK THEME CSS
# ==============================
st.markdown("""
<style>

/* Remove Top White Header */
header {
    background: transparent !important;
}

[data-testid="stHeader"] {
    background: transparent;
}

/* Main App Background */
.stApp {
    background-color: #0F172A;
}

/* Main Container */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Titles */
h1, h2, h3 {
    color: #F8FAFC;
}

/* Normal Text */
p, label, div {
    color: #CBD5E1;
    font-size: 16px;
}

/* Text Area */
.stTextArea textarea {
    background-color: #1E293B !important;
    color: #F8FAFC !important;
    border-radius: 12px;
    border: 1px solid #334155;
    font-size: 16px;
    padding: 12px;
}

/* Button */
.stButton button {
    background-color: #7C6CF2;
    color: white;
    border-radius: 10px;
    height: 50px;
    border: none;
    font-size: 16px;
    font-weight: 600;
    width: 100%;
}

.stButton button:hover {
    background-color: #6B5AE0;
    color: white;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #1E293B;
}

/* Sidebar Text */
[data-testid="stSidebar"] * {
    color: #F8FAFC !important;
}

/* Info Boxes */
.stAlert {
    border-radius: 12px;
}

/* Metric Cards */
.stMetric {
    background-color: #1E293B;
    padding: 10px;
    border-radius: 10px;
}

hr {
    border-color: #334155;
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
• SVM
• Logistic Regression

NLP Pipeline:
TF-IDF + Text Preprocessing
    """)

    st.warning("""
⚠️ This tool is for educational purposes only.

It is NOT a clinical diagnostic system.
    """)

    st.markdown("---")

    st.markdown("### Example Inputs")

    st.markdown("""
<div style="
background-color:#334155;
padding:12px;
border-radius:10px;
margin-bottom:10px;
color:white;
">
I feel hopeless and tired
</div>

<div style="
background-color:#334155;
padding:12px;
border-radius:10px;
margin-bottom:10px;
color:white;
">
I am anxious all the time
</div>

<div style="
background-color:#334155;
padding:12px;
border-radius:10px;
margin-bottom:10px;
color:white;
">
I want to end my life
</div>

<div style="
background-color:#334155;
padding:12px;
border-radius:10px;
margin-bottom:10px;
color:white;
">
Today was a beautiful day
</div>
""", unsafe_allow_html=True)

# ==============================
# TITLE
# ==============================
st.title("🧠 Mental Health Status Classifier")

st.markdown(
    "Classify social media text into 'Anxiety', 'Depression', 'Suicidal', or 'Normal'."
)

st.markdown("---")

# ==============================
# INPUT
# ==============================
user_input = st.text_area(
    "Enter your text here:",
    height=180,
    placeholder="Type a sentence here..."
)

# ==============================
# BUTTON
# ==============================
if st.button("Analyze Text"):

    if user_input.strip() == "":
        st.warning("Please enter some text to classify.")

    else:

        with st.spinner("Analyzing text..."):

            clean_text_input = clean_text(user_input)

            if clean_text_input.strip() == "":
                st.warning(
                    "The input became empty after preprocessing."
                )

            else:

                text_tfidf = tfidf.transform([clean_text_input])

                # ======================
                # SVM PREDICTION
                # ======================
                svm_pred_encoded = svm_model.predict(text_tfidf)[0]

                svm_pred_proba = svm_model.predict_proba(
                    text_tfidf
                )[0]

                svm_pred_class = label_encoder.inverse_transform(
                    [svm_pred_encoded]
                )[0]

                svm_confidence = np.max(svm_pred_proba) * 100

                # ======================
                # LR PREDICTION
                # ======================
                lr_pred_encoded = lr_model.predict(text_tfidf)[0]

                lr_pred_proba = lr_model.predict_proba(
                    text_tfidf
                )[0]

                lr_pred_class = label_encoder.inverse_transform(
                    [lr_pred_encoded]
                )[0]

                lr_confidence = np.max(lr_pred_proba) * 100

                # ======================
                # RESULTS
                # ======================
                st.subheader("📊 Classification Results")

                col1, col2 = st.columns(2)

                # ======================
                # SVM COLUMN
                # ======================
                with col1:

                    st.markdown("### 🧠 SVM Prediction")

                    if svm_pred_class == "Suicidal":
                        st.error(f"Predicted Class: {svm_pred_class}")

                    elif svm_pred_class == "Depression":
                        st.warning(f"Predicted Class: {svm_pred_class}")

                    elif svm_pred_class == "Anxiety":
                        st.info(f"Predicted Class: {svm_pred_class}")

                    else:
                        st.success(f"Predicted Class: {svm_pred_class}")

                    st.metric(
                        label="Confidence",
                        value=f"{svm_confidence:.2f}%"
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

                    st.markdown("### 🤖 Logistic Regression Prediction")

                    if lr_pred_class == "Suicidal":
                        st.error(f"Predicted Class: {lr_pred_class}")

                    elif lr_pred_class == "Depression":
                        st.warning(f"Predicted Class: {lr_pred_class}")

                    elif lr_pred_class == "Anxiety":
                        st.info(f"Predicted Class: {lr_pred_class}")

                    else:
                        st.success(f"Predicted Class: {lr_pred_class}")

                    st.metric(
                        label="Confidence",
                        value=f"{lr_confidence:.2f}%"
                    )

                    lr_df = pd.DataFrame({
                        "Class": CLASSES,
                        "Probability": lr_pred_proba
                    }).set_index("Class")

                    st.bar_chart(lr_df)

                st.markdown("---")

                st.info("""
Predictions from SVM and Logistic Regression may vary slightly.

This system is intended for educational and research purposes only.
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
