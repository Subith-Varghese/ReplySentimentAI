# SvaraAI Reply Classifier

A modular Python project for **text classification** of replies into **positive, negative, and neutral** categories. It supports two models:

1. **Baseline Model** – TF-IDF features + Logistic Regression  
2. **Transformer Model** – DistilBERT for sequence classification  

The project is structured for **reproducibility, modularity, and easy deployment**.

---

## Project Structure
```
project/
│
├─ data/
│ ├─ reply_classification_dataset.csv # Original raw dataset
│ ├─ train_replies.csv # Preprocessed training dataset
│ └─ test_replies.csv # Preprocessed test dataset
│
├─ models/
│ ├─ baseline_model.joblib # Saved TF-IDF + Logistic Regression pipeline
│ └─ bert_reply_classifier/ # Saved DistilBERT model + tokenizer
│
├─ src/
│ ├─ preprocessing.py # Text cleaning & train/test split functions
│ ├─ train_baseline.py # Build, train, evaluate, and save baseline model
│ ├─ train_transformer.py # Prepare datasets, train DistilBERT, save model/tokenizer
│ ├─ predict.py # Functions to predict on new sentences
│ └─ logger.py # Logger configuration for consistent logging
│
├─ notebook.ipynb
├─ main.py # Orchestrates preprocessing, baseline, and transformer training
├─ app.py # Flask app for web-based predictions
└─ README.md # Project overview and workflow
```

---

## Features

- **Preprocessing**
  - Cleans text by:
    - Expanding contractions (e.g., `can't` → `cannot`, `won't` → `will not`)
    - Correcting slang (e.g., `"plz"` → `"please"`, `"u"` → `"you"`, `"thx"` → `"thanks"`, `"gr8"` → `"great"`)
    - Correcting typos/variants (e.g., `"schdule"` → `"schedule"`, `"intrested"` / `"intrsted"` → `"interested"`, `"alredy"` → `"already"`)
    - Removing punctuation and special characters
    - Normalizing whitespace
  - Normalizes labels to lowercase (`positive`, `negative`, `neutral`)
  - Saves train/test splits as CSV (`train_replies.csv` and `test_replies.csv`)


- **Baseline Model**
  - TF-IDF vectorization (unigrams)
  - Logistic Regression with class balancing
  - Evaluation metrics: Accuracy, Weighted F1, Confusion Matrix
  - Saved as `baseline_model.joblib`

- **Transformer Model**
  - DistilBERT for sequence classification
  - Max sequence length optimized for dataset (~12 tokens)
  - Early stopping during training
  - Saved model and tokenizer for inference

- **Prediction Module**
  - Predict using baseline or DistilBERT models
  - Returns predicted label and confidence/probabilities

- **Web App**
  - Simple Flask app to input text
  - Select model (baseline / BERT)
  - Returns label and confidence

---

## Performance Comparison
```
| Model                   | Accuracy | Weighted F1 | Notes                                      |
|-------------------------|----------|------------|--------------------------------------------|
| Baseline (TF-IDF + LR)  | 99.77%   | 99.77%     | Lightweight, fast inference                |
| DistilBERT (Transformer)| 100%     | 100%       | Excellent accuracy, heavier, slower       |
```
### Baseline Model: TF-IDF + Logistic Regression

**Metrics:**
- Accuracy: 0.9977  
- Weighted F1: 0.9977  


**Confusion Matrix:**
![confusion_metrix](https://github.com/Subith-Varghese/ReplySentimentAI/blob/8e678a48c3e10abd83609115e3fc7650a118add8/confusion_metrix.png)

---

### Transformer Model: DistilBERT

**Training/Validation Performance:**
| Epoch | Training Loss | Validation Loss | Accuracy | F1    |
|-------|---------------|----------------|----------|-------|
| 1     | No log        | 0.000573       | 1.0      | 1.0   |
| 2     | No log        | 0.000181       | 1.0      | 1.0   |
| 3     | 0.040000      | 0.000088       | 1.0      | 1.0   |
| 4     | 0.040000      | 0.000054       | 1.0      | 1.0   |
| 5     | 0.000100      | 0.000036       | 1.0      | 1.0   |
| 6     | 0.000100      | 0.000026       | 1.0      | 1.0   |

**Evaluation on Test Set:**
```json
{
  "eval_loss": 0.000573,
  "eval_accuracy": 1.0,
  "eval_f1": 1.0,
  "eval_runtime": 0.3976,
  "eval_samples_per_second": 1071.317,
  "eval_steps_per_second": 135.801,
  "epoch": 6.0
}
```
---


## Workflow

1. **Data Preprocessing**
    ```python
    from src.preprocessing import preprocess_and_save
    import pandas as pd

    dataset = pd.read_csv("data/reply_classification_dataset.csv")
    train_df, test_df = preprocess_and_save(dataset)
    ```

2. **Train Baseline Model**
    ```python
    from src.train_baseline import train_and_evaluate

    X_train = train_df['reply']
    y_train = train_df['label']
    X_test = test_df['reply']
    y_test = test_df['label']

    pipeline = train_and_evaluate(X_train, y_train, X_test, y_test)
    ```

3. **Train DistilBERT Model**
    ```python
    from src.train_transformer import prepare_datasets, tokenize_dataset, build_model, train_model

    train_dataset, test_dataset = prepare_datasets(train_df, test_df)
    train_dataset, tokenizer = tokenize_dataset(train_dataset)
    model = build_model()
    train_model(model, tokenizer, train_dataset, test_dataset)
    ```

4. **Prediction on New Text**
    ```python
    from src.predict import predict_baseline, predict_bert

    text = "I love this product!"

    baseline_label = predict_baseline(text)
    bert_label = predict_bert(text)
    ```

5. **Run Web Application**
    ```bash
    python app.py
    ```
    - Visit `http://127.0.0.1:8000/` in a browser
    - Enter text and choose model
    - View predicted label and confidence

![website](https://github.com/Subith-Varghese/ReplySentimentAI/blob/7dc8d61ac8eb82aae31d9c81e2db8f9a3395f1d2/website.png)
---

## Dependencies

- Python 3.8+
- Pandas, NumPy
- scikit-learn
- matplotlib, seaborn
- transformers, datasets
- torch
- flask
- contractions

Install dependencies:
```bash
pip install -r requirements.txt



















