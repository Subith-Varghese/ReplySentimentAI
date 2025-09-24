import pandas as pd
from src.preprocessing import preprocess_and_save
from src.train_baseline import train_and_evaluate
from src.train_transformer import prepare_datasets, tokenize_dataset, build_model, train_model


# Step 1: Preprocessing (if needed)
dataset = pd.read_csv("data/reply_classification_dataset.csv")
train_df, test_df = preprocess_and_save(dataset)

X_train = train_df['reply']
y_train = train_df['label']

X_test = test_df['reply']
y_test = test_df['label']

# Step 2: Train, evaluate and save baseline model
pipeline = train_and_evaluate(X_train, y_train, X_test, y_test)

# Step 3: Train DistilBERT model
train_dataset, test_dataset = prepare_datasets(train_df, test_df)
train_dataset, tokenizer = tokenize_dataset(train_dataset)
model = build_model()
train_model(model, tokenizer, train_dataset, test_dataset)
