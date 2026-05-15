# 🧠 Mental Health Status Classification from Social Media Text

> Predictive Analytics Course Project | Academic Year 2025–26

---

# 📋 Project Details

| Field | Details |
|---|---|
| **Course** | Predictive Analytics |
| **Institution** | Kerala Digital University |
| **Project Type** | NLP & Machine Learning |
| **Deployment** | Streamlit Web Application |
| **Models Used** | TF-IDF + SVM, Logistic Regression, BERT/DistilBERT |

---

# 👥 Team Members

| Name | Contribution |
|---|---|
| Sree | EDA, Evaluation, README |
| Member 2 | Preprocessing, Feature Engineering |
| Member 3 | Model Building, Deployment |
| Member 4 | PPT, Documentation, Ethics |

> Replace placeholder names and contributions with actual team details.

---

# 📌 Project Overview

Mental health conditions such as depression, anxiety, PTSD, and suicidal ideation are increasingly discussed on social media platforms like Reddit and Twitter.

This project focuses on building an NLP-based machine learning system capable of classifying social media posts into mental health categories:

- Anxiety
- Depression
- Suicidal
- Normal

The project compares traditional machine learning techniques using TF-IDF + SVM/Logistic Regression with transformer-based deep learning models such as BERT/DistilBERT.

---

# ❗ Problem Statement

Social media users often express emotional distress online through text posts. Manual analysis of large-scale online content is difficult and time-consuming.

This project aims to develop an NLP-based classification system capable of identifying mental health-related linguistic patterns from social media text.

> ⚠️ This system is intended for educational and research purposes only and is not a clinical diagnostic tool.

---

# 🎯 Objectives

- Perform text preprocessing and exploratory data analysis (EDA)
- Extract meaningful linguistic features using TF-IDF
- Train and evaluate machine learning models such as Logistic Regression and SVM
- Fine-tune BERT/DistilBERT models
- Compare traditional ML with transformer-based NLP
- Evaluate models using:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Sensitivity
  - Specificity
- Discuss ethical considerations and responsible AI usage
- Deploy the model using Streamlit

---

# 📂 Dataset Description

## Dataset Source

The dataset was collected from publicly available Reddit/Twitter mental health datasets.

## Dataset Files

```bash
mental_health_unbalanced.csv
mental_health_feature_engineered.csv
mental_health_combined_test.csv
```

## Classes

| Label | Description |
|---|---|
| Anxiety | Panic, stress, and worry-related posts |
| Depression | Sadness, hopelessness, low-energy posts |
| Suicidal | Self-harm and suicidal ideation posts |
| Normal | General non-mental-health-related posts |

---

# ⚙️ Data Preprocessing

The following preprocessing steps were applied:

- Lowercasing
- URL removal
- Punctuation removal
- Stopword removal
- Tokenization
- Lemmatization

For BERT-based models, minimal preprocessing was used because transformer tokenizers already handle much of the linguistic processing.

---

# 📊 Exploratory Data Analysis (EDA)

EDA techniques used:

- Class distribution analysis
- Word clouds
- Text length distribution
- Frequent word analysis
- N-gram analysis
- Sentiment analysis
- Correlation heatmaps

---

# 🧩 Feature Engineering

## TF-IDF Vectorization

TF-IDF converts text into numerical vectors based on word importance.

### Features Used

- Unigrams
- Bigrams

### Additional Engineered Features

- Text length
- Word count
- Polarity
- Subjectivity
- Keyword presence
- POS tag ratios

---

# 🤖 Models Used

## Traditional Machine Learning Models

### Logistic Regression

Used as a baseline classifier for NLP text classification.

### Support Vector Machine (SVM)

A powerful classifier effective for high-dimensional sparse TF-IDF vectors.

---

## Deep Learning Model

### BERT / DistilBERT

Transformer-based contextual embedding model capable of understanding semantic relationships within text.

### Advantages

- Contextual understanding
- Better semantic representation
- Improved classification performance

---

# 📈 Evaluation Metrics

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Sensitivity
- Specificity
- Confusion Matrix

---

# 📊 Results Summary

| Model | Accuracy | F1-Score | Recall |
|---|---|---|---|
| Logistic Regression | XX% | XX | XX |
| SVM | XX% | XX | XX |
| BERT | XX% | XX | XX |

> Replace placeholder values with actual notebook results.

---

# 🔍 Explainability & Interpretation

The project includes:

- Important keyword analysis
- Feature importance visualization
- SHAP/LIME explainability techniques

These methods help understand which words strongly influence predictions.

---

# 🖥️ Streamlit Deployment

The project includes deployment preparation using Streamlit for real-time text prediction.

## Features

- User text input
- Real-time prediction
- Prediction confidence scores
- Probability visualization
- User-friendly dashboard

---

# 🖼️ Application Screenshots

## Home Screen

![Home Screen](screenshots/app_home.png)

---

## Prediction Output

![Prediction Result](screenshots/app_result.png)

---

## Probability Distribution

![Probability Chart](screenshots/app_probabilities.png)

---

# ⚖️ Ethical Considerations

Mental health NLP systems involve highly sensitive information.

## Ethical Concerns Addressed

- Data privacy and confidentiality
- Risk of false positives and false negatives
- Dataset bias and demographic imbalance
- Responsible AI usage
- Non-clinical usage disclaimer

> This system should assist professionals, not replace them.

---

# 🛠️ Technologies Used

| Category | Tools |
|---|---|
| Programming Language | Python |
| NLP | NLTK, spaCy |
| Machine Learning | Scikit-learn |
| Deep Learning | Transformers, PyTorch |
| Visualization | Matplotlib, Seaborn |
| Deployment | Streamlit |

---

# 📁 Project Structure

```bash
mental-health-classifier/
│
├── Mental_Health_Classifier.ipynb
├── app.py
├── requirements.txt
├── README.md
│
├── models/
│   ├── svm_model.pkl
│   ├── lr_model.pkl
│   ├── tfidf.pkl
│   └── label_encoder.pkl
│
├── screenshots/
│   ├── app_home.png
│   ├── app_result.png
│   └── app_probabilities.png
│
└── datasets/
```

---

# 🚀 Installation & Setup

## Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/mental-health-classifier.git
cd mental-health-classifier
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run Notebook

```bash
jupyter notebook
```

---

## Run Streamlit App

```bash
streamlit run app.py
```

---

# 🌐 Deployment Link

```text
https://your-streamlit-app-link.streamlit.app/
```

---

# ⚠️ Limitations

- Performance depends heavily on dataset quality
- Informal slang and sarcasm may affect predictions
- Model is not suitable for clinical diagnosis
- Possible demographic and linguistic bias

---

# 🔮 Future Work

Possible future improvements:

- Multilingual mental health detection
- Real-time monitoring systems
- Better explainability techniques
- Emotion-aware transformers
- Multi-modal analysis using text + audio + images

---

# ✅ Conclusion

This project demonstrates how NLP and machine learning techniques can be applied to identify mental health-related linguistic patterns from social media text.

The comparison between TF-IDF + SVM and BERT highlights the effectiveness of transformer-based contextual understanding for mental health classification tasks.

The project also emphasizes ethical AI, explainability, and responsible deployment in healthcare-related NLP systems.

---

# 📚 References

- Hugging Face Transformers Documentation
- Scikit-learn Documentation
- Streamlit Documentation
- Research papers on Mental Health NLP
- Public Reddit/Twitter mental health datasets

---
