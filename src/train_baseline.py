import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from src.logger import logger

MAX_FEATURES = 215  # Slightly higher than vocab size

def build_pipeline():
    # Build TF-IDF + Logistic Regression pipeline
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,1), max_features=MAX_FEATURES)),
        ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
    ])
    return pipeline

def train_and_evaluate(X_train, y_train, X_test, y_test, save_path="models/baseline_model.joblib"):
    try: 
        # Train the baseline model, evaluate, plot confusion matrix, and save the model
        pipeline = build_pipeline()
        pipeline.fit(X_train, y_train)
        logger.info("Baseline model trained successfully.")

        # Predict
        y_pred = pipeline.predict(X_test)

        # Evaluation
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        logger.info(f"Baseline Model Accuracy: {acc:.4f}")
        logger.info(f"Baseline Model Weighted F1: {f1:.4f}")
        logger.info("\nClassification Report:\n", classification_report(y_test, y_pred))

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred, labels=['positive', 'negative', 'neutral'])
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=['positive', 'negative', 'neutral'],
                    yticklabels=['positive', 'negative', 'neutral'], cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix - Baseline Model')
        plt.show()

        # Save model
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(pipeline, save_path)
        logger.info(f"Baseline model saved as '{save_path}'")
        return pipeline
    except Exception as e:
        logger.exception("Error during model training:")
        raise e
