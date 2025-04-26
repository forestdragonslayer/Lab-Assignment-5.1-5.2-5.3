# Lab-Assignment-5.1-5.2-5.3

# 📊 Deep Learning Experiments with LSTM

This repository contains three LSTM-based deep learning experiments focused on different types of sequence modeling problems using Python and TensorFlow/Keras.

---

## 📈 Experiment 1: Univariate Time Series Forecasting (Stock Prices)

### Objective
To forecast future values of a univariate time series using LSTM, using stock price data (e.g., Tesla).

### Dataset
- **Stock Price Dataset** (e.g., Tesla from Yahoo Finance)
- Format: Date, Open, High, Low, Close, Volume

### Steps
- Data normalization
- Sequence generation using sliding window
- LSTM model building and training
- Prediction and evaluation

### Expected Outcomes
- **Prediction vs Actual plot**
- **Evaluation Metrics:**  
  - RMSE (Root Mean Squared Error)  
  - MAE (Mean Absolute Error)

---

## 📖 Experiment 2: Sequence Text Prediction (Character-Level Text Generation)

### Objective
To generate the next characters based on an input sequence using an LSTM model trained on Harry Potter books.

### Dataset
- **Harry Potter Books** (plain text format)

### Steps
- Text preprocessing and cleaning
- Character-level tokenization
- LSTM sequence generation and training
- Text generation using the trained model

### Expected Outcomes
- **Auto-generated Text Samples**
- **Training Accuracy and Loss Plots**

---

## ✉️ Experiment 3: Sequence Text Classification (Spam Detection)

### Objective
To classify SMS messages as spam or ham using an LSTM-based text classification model.

### Dataset
- **SMS Spam Collection Dataset** from UCI
- Format: Label (ham/spam), Message

### Steps
- Text preprocessing and tokenization
- Sequence padding
- LSTM model training
- Model evaluation

### Expected Outcomes
- **Classification Metrics:**
  - Accuracy
  - Precision
  - Recall
  - F1-Score
- **Confusion Matrix Visualization**

---

## 🧠 Evaluation Metrics Explained (for Text Classification)

- **Accuracy:** Overall correctness of predictions.
- **Precision:** Proportion of predicted spam that is actually spam.
- **Recall:** Proportion of actual spam correctly identified.
- **F1-Score:** Harmonic mean of precision and recall.
- **Confusion Matrix:** Shows true/false positives and negatives to visualize model performance.

> In spam detection, **high recall** helps avoid missing spam, while **high precision** ensures real messages aren’t falsely flagged. The **F1-score** balances both.

---

## 🛠 Technologies Used

- Python, TensorFlow, Keras
- NumPy, Pandas, Matplotlib, Seaborn
- Scikit-learn
- Jupyter/Colab Notebooks

---

## 📌 How to Run
1. Clone this repo
2. Install required libraries
3. Run each experiment notebook/script in sequence

---

## 📬 License
This project is for educational and research purposes.
