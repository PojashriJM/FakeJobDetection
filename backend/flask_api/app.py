from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ================= BASIC SETUP =================

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MAX_LEN = 300

# ================= GLOBAL MODEL VARIABLES =================

logistic_model = None
rf_model = None
vectorizer = None
cnn_model = None
lstm_model = None
tokenizer = None

# ================= LOAD MODELS (LAZY LOAD) =================

def load_models():
    global logistic_model, rf_model, vectorizer
    global cnn_model, lstm_model, tokenizer

    if logistic_model is None:
        with open(os.path.join(BASE_DIR, "models", "logistic_model.pkl"), "rb") as f:
            logistic_model = pickle.load(f)

    if rf_model is None:
        with open(os.path.join(BASE_DIR, "models", "random_forest_model.pkl"), "rb") as f:
            rf_model = pickle.load(f)

    if vectorizer is None:
        with open(os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl"), "rb") as f:
            vectorizer = pickle.load(f)

    if cnn_model is None:
        cnn_model = load_model(os.path.join(BASE_DIR, "models", "cnn.keras"))

    if lstm_model is None:
        lstm_model = load_model(os.path.join(BASE_DIR, "models", "lstm.keras"))

    if tokenizer is None:
        with open(os.path.join(BASE_DIR, "models", "tokenizer.pkl"), "rb") as f:
            tokenizer = pickle.load(f)

# ================= TEXT PREPROCESS =================

def clean_text(text):
    return str(text).lower().strip()

# ================= PAGE ROUTES =================

@app.route("/")
def dashboard():
    return render_template("dashboard.html")

@app.route("/dataset")
def dataset():
    return render_template("dataset.html")

@app.route("/algorithms")
def algorithms():
    splits = ["80_20", "70_30", "60_40"]
    models = {
        "logistic": "Logistic Regression",
        "random_forest": "Random Forest",
        "cnn": "CNN",
        "lstm": "LSTM"
    }
    images = [
        "data_distribution.png",
        "split_ratio.png",
        "confusion_matrix.png",
        "roc_curve.png",
        "auc_comparison.png"
    ]

    return render_template(
        "algorithms.html",
        splits=splits,
        models=models,
        images=images
    )

@app.route("/comparison")
def comparison():
    return render_template("comparison.html")

@app.route("/prediction")
def prediction_page():
    return render_template("prediction.html")

# ================= PREDICTION API (ENSEMBLE) =================

@app.route("/predict", methods=["POST"])
def predict():
    try:
        load_models()
        data = request.json

        text = clean_text(
            data.get("job_title", "") + " " +
            data.get("job_description", "") + " " +
            data.get("requirements", "") + " " +
            data.get("salary", "")
        )

        if len(text) < 10:
            return jsonify({"error": "Please enter more job details"}), 400

        # ---------- TF-IDF MODELS ----------
        text_vec = vectorizer.transform([text])

        prob_logistic = logistic_model.predict_proba(text_vec)[0][1]
        prob_rf = rf_model.predict_proba(text_vec)[0][1]

        # ---------- DL MODELS ----------
        seq = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post")

        prob_cnn = float(cnn_model.predict(padded, verbose=0).flatten()[0])
        prob_lstm = float(lstm_model.predict(padded, verbose=0).flatten()[0])

        # ---------- WEIGHTED ENSEMBLE ----------
        final_prob = (
            0.2 * prob_logistic +
            0.2 * prob_rf +
            0.3 * prob_cnn +
            0.3 * prob_lstm
        )

        if final_prob >= 0.5:
            prediction = "Fake Job"
            confidence = final_prob * 100
        else:
            prediction = "Real Job"
            confidence = (1 - final_prob) * 100

        return jsonify({
            "prediction": prediction,
            "confidence": round(confidence, 2)
        })

    except Exception as e:
        print("ENSEMBLE ERROR:", e)
        return jsonify({"error": "Prediction failed"}), 500

# ================= RUN SERVER =================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
