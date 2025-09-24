import os
import logging
from datasets import Dataset
from transformers import (
    DistilBertTokenizerFast, 
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, f1_score
from src.logger import logger  # logger from logger.py

# Label mappings
label2id = {"positive": 0, "negative": 1, "neutral": 2}
id2label = {v: k for k, v in label2id.items()}

def prepare_datasets(train_df, test_df):
    try:
        # Map labels to integers
        train_df["labels"] = train_df["label"].map(label2id)
        test_df["labels"] = test_df["label"].map(label2id)

        train_dataset = Dataset.from_pandas(train_df.drop(columns=["label"]))
        test_dataset = Dataset.from_pandas(test_df.drop(columns=["label"]))

        logger.info("Datasets converted to Hugging Face Dataset format.")
        return train_dataset, test_dataset

    except Exception as e:
        logger.exception("Error preparing datasets:")
        raise e

def tokenize_dataset(dataset, max_length=12):
    try:
        tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
        def tokenize(batch):
            return tokenizer(batch["reply"], padding="max_length", truncation=True, max_length=max_length)
        tokenized_dataset = dataset.map(tokenize)
        logger.info("Dataset tokenized successfully.")
        return tokenized_dataset, tokenizer
    except Exception as e:
        logger.exception("Error during tokenization:")
        raise e

def build_model():
    try:
        model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels=3,
            id2label=id2label,
            label2id=label2id
        )
        logger.info("DistilBERT model loaded successfully.")
        return model
    except Exception as e:
        logger.exception("Error loading DistilBERT model:")
        raise e

def train_model(model, tokenizer, train_dataset, test_dataset, output_dir="models/bert_reply_classifier"):
    try:
        training_args = TrainingArguments(
            output_dir="./results",
            save_strategy="epoch",
            eval_strategy="epoch",
            per_device_train_batch_size=8,
            num_train_epochs=15,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            report_to="none"
        )

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = logits.argmax(axis=-1)
            acc = accuracy_score(labels, preds)
            f1 = f1_score(labels, preds, average="weighted")
            return {"accuracy": acc, "f1": f1}

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
        )

        trainer.train()
        logger.info("DistilBERT training completed successfully.")

        # Save model + tokenizer
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info(f"Model and tokenizer saved in '{output_dir}'.")

    except Exception as e:
        logger.exception("Error during DistilBERT training:")
        raise e
