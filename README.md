# Mental Health Status Classification from Social Media Text

> **Project #26 — Predictive Analytics (Group project-2)**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://mentalhealthclassifier-predictiveproject.streamlit.app/)

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

| Name |  |
|---|---|
| Project 2 | Group32 |
| Arya Sukku |  |
| Sreekutty Santhosh  |  |
| Aparna V |  |



---

---

## 📌 Problem Statement

Mental health conditions such as depression, anxiety, PTSD, and suicidal ideation are increasingly expressed on social media platforms like Reddit and Twitter. Detecting such patterns manually from large-scale textual data is difficult, time-consuming, and often impractical.

This project focuses on automatically classifying social media posts into mental health categories using Natural Language Processing (NLP) and Machine Learning techniques.

The classification categories include:

- Anxiety
- Depression
- Suicidal
- Normal

The project compares traditional machine learning approaches using TF-IDF + SVM/Logistic Regression with transformer-based deep learning models such as BERT/DistilBERT.

---

## 🎯 Success Criterion

| Metric | Target | Rationale |
|---|---|---|
| Macro F1-score | ≥ 0.85 | Ensures balanced performance across all mental health classes |

---

## 📊 Dataset

The dataset was collected from publicly available Reddit/Twitter mental health datasets containing user-generated textual posts related to emotional and psychological conditions.

### Dataset Files

```bash
mental_health_unbalanced.csv
mental_health_feature_engineered.csv
mental_health_combined_test.csv
```

### Classes

| Label | Description |
|---|---|
| Anxiety | Panic, stress, and worry-related posts |
| Depression | Sadness, hopelessness, and low-energy posts |
| Suicidal | Self-harm and suicidal ideation posts |
| Normal | General non-mental-health-related posts |

### Dataset Characteristics

- Multi-class NLP classification problem
- Short-form and long-form social media text
- Imbalanced emotional vocabulary distribution
- Informal language, slang, abbreviations, and noisy text patterns

Mental health text classification is particularly challenging because multiple categories often share overlapping emotional vocabulary and contextual patterns.

---

## 🔬 Methodology — Full Data Science Lifecycle

All **10 stages** of the Data Science Project Lifecycle were implemented end-to-end:

| Stage | Description |
|---|---|
| 1. Problem Definition & Literature Review | Understanding mental health NLP systems and prior research |
| 2. Data Collection & Understanding | Collecting and understanding social media mental health datasets |
| 3. Data Preprocessing & Cleaning | Lowercasing, stopword removal, tokenization, lemmatization |
| 4. Exploratory Data Analysis | Word clouds, class distribution, sentiment analysis, text statistics |
| 5. Feature Engineering & Selection | TF-IDF vectorization, text statistics, engineered linguistic features |
| 6. Model Building & Training | Logistic Regression, SVM, BERT/DistilBERT |
| 7. Model Evaluation & Comparison | Accuracy, Precision, Recall, F1-score, Sensitivity, Specificity |
| 8. Model Interpretation & Explainability | Feature importance, keyword analysis, SHAP/LIME |
| 9. Deployment | Streamlit web application for real-time prediction |
| 10. Documentation | README, PPT presentation, GitHub repository |

---

## ⚙️ Data Preprocessing

The following preprocessing techniques were applied:

- Lowercasing
- URL removal
- Punctuation removal
- Stopword removal
- Tokenization
- Lemmatization

For transformer-based models such as BERT, minimal preprocessing was used because transformer tokenizers already handle much of the linguistic processing internally.

---

## 📊 Exploratory Data Analysis (EDA)

EDA techniques used in the project:

- Class distribution analysis
- Word clouds
- Text length distribution
- Frequent word analysis
- N-gram analysis
- Sentiment analysis
- Correlation heatmaps

These analyses helped identify emotional vocabulary patterns associated with different mental health categories.

---

## 🧩 Feature Engineering

### TF-IDF Vectorization

TF-IDF converts text into numerical vectors based on word importance across the corpus.

### Features Used

- Unigrams
- Bigrams

### Additional Engineered Features

- Text length
- Word count
- Sentiment polarity
- Subjectivity
- Keyword presence
- POS tag ratios

---

## 🤖 Models Used

### Traditional Machine Learning Models

| Model | Purpose |
|---|---|
| Logistic Regression | Baseline NLP classifier |
| Support Vector Machine (SVM) | High-dimensional sparse text classification |

### Deep Learning Model

| Model | Purpose |
|---|---|
| BERT / DistilBERT | Context-aware transformer-based NLP classification |

### Why BERT?

Transformer models provide contextual understanding of language and semantic relationships, making them highly effective for complex NLP tasks involving emotional and psychological text patterns.

---

## 📈 Results

### Model Comparison Summary

| Model | Accuracy | F1-Score | Recall | Notes |
|---|---|---|---|---|
| Logistic Regression | 75.81% | 0.7579 | 0.76 | Best overall performance among classical ML models |
| SVM | 74.80% | 0.7456 | 0.75 | Strong TF-IDF based classification performance |
| Ensemble Model | 75.30% | 0.7517 | 0.75 | Balanced performance using combined predictions |

> Logistic Regression achieved the highest overall Macro F1-score of **0.7579**, while SVM demonstrated strong performance for sparse high-dimensional TF-IDF features.

---

### Per-Class Performance — Logistic Regression

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Anxiety | 0.85 | 0.76 | 0.80 |
| Depression | 0.63 | 0.62 | 0.62 |
| Normal | 0.82 | 0.85 | 0.83 |
| Suicidal | 0.74 | 0.81 | 0.77 |

---

### Per-Class Performance — SVM

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Anxiety | 0.93 | 0.59 | 0.72 |
| Depression | 0.60 | 0.66 | 0.63 |
| Normal | 0.77 | 0.92 | 0.84 |
| Suicidal | 0.76 | 0.83 | 0.80 |

---

## 📌 Key Observations

- Logistic Regression achieved the best overall Macro F1-score among the implemented models.
- SVM achieved very high precision for Anxiety classification but lower recall.
- The "Normal" class achieved the highest classification performance across both models.
- Depression classification remained the most challenging category because of overlapping emotional vocabulary patterns.
- Ensemble learning provided balanced performance but did not significantly outperform Logistic Regression.

---

## 🔍 Model Explainability

The project includes explainability-focused analysis using:

- Important keyword analysis
- Feature importance visualization
- SHAP/LIME explainability methods

These approaches help interpret which textual patterns strongly influence predictions.

---

## 🌐 Live App

🔗 **[Try the deployed Streamlit app →](https://mentalhealthclassifier-predictiveproject.streamlit.app/)**

### Features

- User text input
- Real-time prediction
- Multi-model comparison
- Prediction confidence display
- Interactive web interface

---

## 🖼️ Application Screenshots

### Home Screen

![Home Screen] <img width="1920" height="1197" alt="app_home" src="https://github.com/user-attachments/assets/61db7ad8-8f90-41a5-a060-51f715aa602d" />


---

### Prediction Output

![Prediction Result]<img width="1920" height="1888" alt="app_result" src="https://github.com/user-attachments/assets/d561dbe5-db6c-42f3-a74b-eceb0fb129eb" />


---

### Example Prediction

![Prediction Example]<img width="1920" height="1888" alt="app_probabilities" src="https://github.com/user-attachments/assets/fec200b1-b09a-4699-b699-168192f41b83" />

---

## ⚖️ Ethical Considerations

Mental health NLP systems involve highly sensitive personal information and require responsible deployment practices.

### Ethical Concerns Addressed

- Data privacy and confidentiality
- Risk of false positives and false negatives
- Dataset bias and demographic imbalance
- Responsible AI usage
- Non-clinical usage disclaimer

> This system is intended to assist awareness and research efforts and should not replace professional mental health diagnosis.

---

## ⚠️ Limitations

- Performance depends heavily on dataset quality and diversity.
- Informal slang, sarcasm, and ambiguous emotional language may affect predictions.
- Mental health categories can share overlapping vocabulary patterns.
- The system is not suitable for clinical diagnosis or emergency intervention.

---

## 🔮 Future Work

Possible future improvements include:

- Multilingual mental health detection
- Emotion-aware transformer architectures
- Real-time social media monitoring systems
- Better explainability techniques
- Multi-modal analysis using text + audio + image data

---

## 🛠️ Technologies Used

| Category | Tools |
|---|---|
| Programming Language | Python |
| NLP | NLTK, spaCy |
| Machine Learning | Scikit-learn |
| Deep Learning | Transformers, PyTorch |
| Visualization | Matplotlib, Seaborn |
| Deployment | Streamlit |

---

## 📁 Project Structure

```bash
Mental_Health_Classifier/
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
├── app_image/
│   ├── app_home.png
│   ├── app_result.png
│   └── app_probabilities.png
│
└── Dataset/
```

---

## 🚀 How to Run Locally

```bash
# 1. Clone repository
git clone https://github.com/YOUR_USERNAME/Mental_Health_Classifier.git

# 2. Navigate into project directory
cd Mental_Health_Classifier

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run Streamlit app
streamlit run app.py
```

---

## 📚 References

- Hugging Face Transformers Documentation
- Scikit-learn Documentation
- Streamlit Documentation
- Public Reddit/Twitter mental health datasets
- Research papers on Mental Health NLP

---

## 📜 License

This project was developed as part of  capstone project of the Predictive Analytics course
