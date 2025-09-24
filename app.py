from flask import Flask, request, jsonify, render_template_string
from src.predict import predict_baseline, predict_bert, baseline_pipeline, bert_model, bert_tokenizer
import torch
import torch.nn.functional as F
from src.preprocessing import clean_text

app = Flask(__name__)

# Simple HTML page template
HTML_TEMPLATE = """
<!doctype html>
<html>
<head>
    <title>SvaraAI Reply Classifier</title>
</head>
<body>
    <h1>SvaraAI Reply Classifier</h1>
    <form method="POST" action="/predict_web">
        <label for="text">Enter your text:</label><br>
        <textarea name="text" rows="4" cols="50" required></textarea><br><br>
        <label for="model">Choose model:</label>
        <select name="model">
            <option value="baseline">Baseline (Logistic Regression)</option>
            <option value="bert">DistilBERT</option>
        </select><br><br>
        <input type="submit" value="Predict">
    </form>
    {% if result %}
        <h2>Prediction Result:</h2>
        <p><b>Model:</b> {{ result.model }}</p>
        <p><b>Label:</b> {{ result.label }}</p>
        <p><b>Confidence:</b> {{ result.confidence }}</p>
    {% endif %}
</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route("/predict_web", methods=["POST"])
def predict_web():
    try:
        text = request.form["text"].strip()
        model_choice = request.form.get("model", "baseline").lower()

        if model_choice == "baseline":
            cleaned_text = clean_text(text)
            label = baseline_pipeline.predict([cleaned_text])[0]
            confidence = max(baseline_pipeline.predict_proba([cleaned_text])[0])
        elif model_choice == "bert":
            cleaned_text = clean_text(text)
            inputs = bert_tokenizer(cleaned_text, return_tensors="pt", padding=True, truncation=True)
            outputs = bert_model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1).detach().numpy()[0]
            pred_idx = outputs.logits.argmax(dim=-1).item()
            label = bert_model.config.id2label[pred_idx]
            confidence = float(probs[pred_idx])
        else:
            return "Invalid model choice", 400

        result = {
            "model": model_choice,
            "label": label,
            "confidence": round(float(confidence), 4)
        }

        return render_template_string(HTML_TEMPLATE, result=result)

    except Exception as e:
        return f"Error: {str(e)}", 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
