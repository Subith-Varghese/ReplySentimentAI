import joblib
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
import torch
from src.preprocessing import clean_text  # import from preprocessing
from src.logger import logger

baseline_pipeline = joblib.load("models/baseline_model.joblib")
bert_model = DistilBertForSequenceClassification.from_pretrained("models/bert_reply_classifier")
bert_tokenizer = DistilBertTokenizerFast.from_pretrained("models/bert_reply_classifier")

# -------- Baseline Model Prediction --------
def predict_baseline(text: str):
    try:
        logger.info("Loading baseline model...")
        cleaned_text = clean_text(text)
        pred_label = baseline_pipeline.predict([cleaned_text])[0]
        pred_prob = baseline_pipeline.predict_proba([cleaned_text])[0]
        return pred_label
    except Exception as e:
        logger.exception("Error predicting with baseline model:")
        raise e

# -------- DistilBERT Prediction --------
def predict_bert(text:str):
    try:
        logger.info("Loading DistilBERT model...")
        cleaned_text = clean_text(text)
        inputs = bert_tokenizer(cleaned_text, return_tensors="pt", padding=True, truncation=True)
        outputs = bert_model(**inputs)
        pred_idx = outputs.logits.argmax(dim=-1).item()
        pred_label = bert_model.config.id2label[pred_idx]
        return pred_label
    except Exception as e:
        logger.exception("Error predicting with DistilBERT model:")
        raise e

# -------- Example Usage --------
if __name__ == "__main__":
    print("=== Text Classification ===")
    text = input("Enter your text: ").strip()
    model_choice = input("Enter the model you want to use for prediction (baseline / bert):").strip().lower()

    if model_choice == "baseline":
        label, prob = predict_baseline(text)
        print(f"\n[Baseline Model] Predicted label: {label}\nProbabilities: {prob}")
    elif model_choice == "bert":
        label = predict_bert(text)
        print(f"\n[DistilBERT Model] Predicted label: {label}")
    else:
        print("Invalid model choice! Please choose 'baseline' or 'bert'.")

