1. If you only had 200 labeled replies, how would you improve the model without collecting thousands more?

With only 200 labeled replies, Logistic Regression or DistilBERT might overfit due to limited data.

- Data Augmentation:
    - Paraphrasing: Rewrite replies differently but keep the same label.
        - Example: "I am interested in a demo" → "I would like to schedule a demo" (positive)
    - Synonym Replacement: Swap words with similar meaning.
        - Example: "Looking forward to meeting" → "Excited about meeting"
    - Back-Translation: Translate to another language and back to English.
        - Example: "I can’t attend the demo" → French → back → "I won’t be able to attend the demo" (negative)
- Pretrained Models / Transfer Learning:
- Cross-Validation & Regularization:

    - Use k-fold cross-validation to train multiple models and check stability.
    - Apply regularization (class_weight='balanced') in Logistic Regression to prevent overfitting.


2. How would you ensure your reply classifier doesn’t produce biased or unsafe outputs in production?

- Ensure balanced & representative dataset
- Monitor prediction like track model ourputs for unexpected patterns
- Better prprocessing and filtering like normalize text consistently (lowercase, remove special characters) to avoid bias from text formatting.

3. Suppose you want to generate personalized cold email openers using an LLM. What prompt design strategies would you use to keep outputs relevant and non-generic?