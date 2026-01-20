# 🚨 Hate Speech Detection System 

A Machine Learning + NLP based **Hate Speech Detection System** that classifies text into:
- **Hate Speech**
- **Offensive Language**
- **Normal / Neither**

This project was developed and tested in **Jupyter Notebook**, focusing on building a complete NLP pipeline, evaluating performance with proper metrics, and preparing it for deployment.

---

## ✅ Features

- Load and preprocess hate speech dataset
- Clean raw text (tweets) using NLP techniques
- Convert text to numerical features (TF-IDF)
- Train and evaluate classification models
- Performance visualization (Loss/Accuracy curves + Confusion Matrix)
- Predict custom user input text in real time

---

## 🧠 Dataset

The dataset contains tweets labeled into 3 classes:

| Label | Meaning |
|------:|---------|
| 0 | Hate Speech |
| 1 | Offensive Language |
| 2 | Normal / Neither |

**Columns used:**
- `tweet` → Input text
- `class` → Target label

---

## 🛠️ Tech Stack

- **Python**
- **Jupyter Notebook**
- **Pandas / NumPy**
- **NLTK** (text preprocessing)
- **Scikit-learn**
- **Matplotlib**
- *(Optional)* TensorFlow/Keras for Deep Learning experiments

---

## 🔄 Workflow

### 1) Data Cleaning (Text Preprocessing)
The raw tweets are cleaned using:
- Lowercasing
- URL removal
- Mention removal (`@user`)
- Removing numbers and punctuation
- Removing stopwords
- Stemming

### 2) Feature Engineering
- **TF-IDF Vectorization**
- N-grams (unigram + bigram) for better phrase understanding

### 3) Model Training
Multiple models can be used (example):
- Logistic Regression
- SVM (LinearSVC)
- Naive Bayes

### 4) Model Evaluation
Model performance is measured using:
- Accuracy
- Precision / Recall / F1-score (Class-wise)
- Confusion Matrix
- Training vs Validation Graphs

---

## 📊 Results Summary

✅ Overall accuracy achieved: **~92%**  
✅ Macro Average F1-score: **~0.77**

📌 Class-wise performance (example):
- **Offensive** class performs very strongly
- **Normal** class is predicted reliably
- **Hate speech** class is harder due to dataset imbalance and similarity with offensive language

---

## 📷 Visualizations Included

- Training vs Validation Loss Curve  
- Training vs Validation Accuracy Curve  
- Confusion Matrix Heatmap  
- Classification Report (Precision, Recall, F1-score)

---

## 👤 Author

Arkaprava Roy
