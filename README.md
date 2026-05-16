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
| Arya Sukku |CS-DA(253007)  |
| Sreekutty Santhosh  | Bio-AI(253213) |
| Aparna V |CS-DA(253015)  |



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
---
## 📚 Literature Review 
### 1.1 Historical Evolution (1999-2024)

Mental health detection from social media has evolved through three distinct phases over the past quarter-century. Owen et al. (2024) documented this progression from early psycholinguistic analysis using LIWC dictionaries (1999-2012), through machine learning applications on platforms like Twitter and Reddit (2013-2018), to the current deep learning and transformer-based era (2019-present) [1].

### 1.2 Current State of Research

A comprehensive review by Rohei et al. (2026) analyzing 229 studies (2017-2024) revealed that Twitter (42%) and Reddit (12%) remain the most common data sources for mental health detection research. Textual and linguistic features dominate the field, appearing in 67% of studies, while emotional features account for 17%. The review also found that LLM-based models (GPT-4, Llama) achieve up to 85% accuracy in mental health classification tasks [2].

### 1.3 Key Foundational Studies

**Depression Detection from Social Media**  
De Choudhury et al. (2013) pioneered the use of Twitter data to predict depression before clinical diagnosis. Their research identified key linguistic markers including increased first-person pronoun usage, higher frequency of negative emotion words, and reduced social engagement indicators [4].

**Assessment of Thought Disorders**  
Argolo et al. (2024) applied natural language processing techniques to assess at-risk mental states, demonstrating that semantic analysis and graph theory approaches can effectively detect formal thought disorders with 86% balanced accuracy [3].

**Methodological Framework**  
Chancellor and De Choudhury (2020) provided a comprehensive framework for mental health detection methods on social media, categorizing approaches into dictionary-based, supervised machine learning, and deep learning paradigms [5].

### 1.4 Key Challenges in the Literature

| Challenge | Description | Impact |
|-----------|-------------|--------|
| Data Quality | Limited high-quality public datasets with clinical ground truth | Use self-disclosed Reddit labels |
| Class Imbalance | Depression and anxiety overrepresented vs. other disorders | Apply weighted loss functions |
| Language Overlap | Mental health categories share emotional vocabulary | Implement SHAP/LIME explainability |
| Non-clinical Labels | Most datasets rely on self-disclosure, not clinical diagnosis | Include disclaimer in deployment |

### 1.5 Research Gap & Project Contribution

**Limitations of Existing Research:**
- Most studies focus on binary classification (depressed vs. non-depressed)
- Limited multi-class models for diverse mental health conditions
- Few academic projects provide deployed, usable applications

**This Project Addresses:**
- ✅ Multi-class classification (Anxiety, Depression, Suicidal, Normal)
- ✅ Comparative analysis: TF-IDF+SVM/LR vs. BERT/DistilBERT
- ✅ Production deployment via Streamlit with real-time prediction
- ✅ Model explainability using SHAP/LIME for interpretability

---



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

 <img width="1920" height="1197" alt="app_home" src="https://github.com/user-attachments/assets/61db7ad8-8f90-41a5-a060-51f715aa602d" />


---

### Prediction Output

<img width="1920" height="1888" alt="app_result" src="https://github.com/user-attachments/assets/d561dbe5-db6c-42f3-a74b-eceb0fb129eb" />


---

### Example Prediction

<img width="1920" height="1888" alt="app_probabilities" src="https://github.com/user-attachments/assets/fec200b1-b09a-4699-b699-168192f41b83" />

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
├── Dataset/
│
├── EDA_images/
│
├── Model_results/
│
├── app_image/
│   ├── app_home.png
│   ├── app_result.png
│   └── app_probabilities.png
│
├── individual_profile/
│
├── models/
│   ├── svm_model.pkl
│   ├── lr_model.pkl
│   ├── tfidf.pkl
│   └── label_encoder.pkl
│
├── .devcontainer/
│
├── Mental_Health_Classifier.ipynb
├── Mental_Health_Status_Classification.pdf
├── README.md
├── app.py
└── requirements.txt
```

---

## 🚀 How to Run Locally

```bash
# 1. Clone repository
git clone https://github.com/psarya/Mental_Health_Classifier.git

# 2. Navigate into project directory
cd Mental_Health_Classifier

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run Streamlit app
streamlit run app.py
```

---

## 📖 References

### Peer-Reviewed Journal Articles

[1] Owen, D., et al. (2024). *AI for Analyzing Mental Health Disorders Among Social Media Users: Quarter-Century Narrative Review*. Journal of Medical Internet Research, 26, e59225.  
🔗 https://doi.org/10.2196/59225

[2] Rohei, M. S., et al. (2026). *Review of predictive techniques for detecting mental disorders from user-generated content on social media*. PeerJ Computer Science, 12, e3559.  
🔗 https://doi.org/10.7717/peerj-cs.3559

[3] Argolo, F., et al. (2024). *Natural language processing in at-risk mental states: enhancing the assessment of thought disorders*. Brazilian Journal of Psychiatry.  
🔗 https://doi.org/10.47626/1516-4446-2024-1234

### Conference Papers

[4] De Choudhury, M., Counts, S., & Horvitz, E. (2013). *Predicting Depression via Social Media*. In Proceedings of the International AAAI Conference on Web and Social Media (ICWSM), 7(1), 128-137.  
🔗 https://www.aaai.org/ocs/index.php/ICWSM/ICWSM13/paper/view/6124

### Book Chapters

[5] Chancellor, S., & De Choudhury, M. (2020). *Methods for detecting mental health status on social media*. In Social Media and Mental Health (pp. 35-54). Cambridge University Press.  
🔗 https://doi.org/10.1017/9781108981400.004

### Technical Documentation

[6] Hugging Face. (2024). *Transformers Documentation: BERT Fine-tuning for Classification*.  
🔗 https://huggingface.co/docs/transformers

[7] Scikit-learn Developers. (2024). *Scikit-learn: Text Feature Extraction with TF-IDF*.  
🔗 https://scikit-learn.org/stable/modules/feature_extraction.html

[8] Streamlit. (2024). *Streamlit Documentation: Building ML Apps*.  
🔗 https://docs.streamlit.io

### Datasets

[9] *Reddit Mental Health Dataset* (r/depression, r/anxiety, r/SuicideWatch).  
🔗 https://www.kaggle.com/datasets/mental-health-reddit

[10] *Twitter Mental Health Corpus*.  
🔗 https://www.kaggle.com/datasets/twitter-mental-health

# Team Members Contributions
| Members | Contributions |
| --------- | --------- |
| Arya Suku |
| Aparna V | 
| Sreekutty |

---

## 📜 License

This project was developed as part of  capstone project of the Predictive Analytics course
