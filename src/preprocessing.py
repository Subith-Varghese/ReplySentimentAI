import pandas as pd
import re
import contractions
import os
from sklearn.model_selection import train_test_split
from src.logger import logger

# Slang replacements
slang_dict = {
    "plz": "please",
    "u": "you",
    "thx": "thanks",
    "gr8": "great"
}

# Common typo/variant corrections
typo_dict = {
    "intrest": "interest",
    "intrested": "interested",
    "intrsted": "interested",
    "alredy": "already",
    "schdule": "schedule",
    "oppurtunity": "opportunity"
}

# Basic text cleaning function
def clean_text(text):
    # Expand contractions (e.g., "can't" -> "cannot")
    text = contractions.fix(str(text))
    # Lowercase and strip spaces
    text = str(text).lower().strip()
    # Replace slang
    text = " ".join([slang_dict.get(word, word) for word in text.split()])
    # Replace typos/variants
    text = " ".join([typo_dict.get(word, word) for word in text.split()])
    # Remove punctuation/special characters
    text = re.sub(r'[^a-z0-9\s]', '', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    return text

def preprocess_and_save(df, output_dir = 'data'):
    try: 
        # Preprocess the dataset, split into train/test, and save as CSV.
        # Normalize labels
        df['label'] = df['label'].str.lower().str.strip()

        # Clean text
        df['clean_reply'] = df['reply'].apply(clean_text)
        logger.info("Dataset cleaned successfully.")

        # Train/test split
        X = df['clean_reply']
        y = df['label']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        # Create directories if not exists
        os.makedirs(output_dir, exist_ok=True)

        # Save to CSV
        train_df = pd.DataFrame({'reply': X_train, 'label': y_train})
        test_df = pd.DataFrame({'reply': X_test, 'label': y_test})

        train_df.to_csv(os.path.join(output_dir, 'train_replies.csv'), index=False)
        test_df.to_csv(os.path.join(output_dir, 'test_replies.csv'), index=False)

        logger.info(f"Train and Test datasets saved in '{output_dir}/'")
        return train_df, test_df
    
    except Exception as e:
        logger.exception("Error during preprocessing:")
        raise e
