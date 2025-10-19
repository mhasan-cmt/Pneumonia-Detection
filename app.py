from flask import Flask, request, render_template, jsonify
import os
from PIL import Image
import numpy as np
import tensorflow as tf

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# [Unverified] I do not have access to your model; I'm assuming a Keras .h5 binary classification model saved at 'model.h5'.
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.h5")

# Load model lazily to avoid errors if file missing at import time.
_model = None
def load_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Place your .h5 there.")
        _model = tf.keras.models.load_model(MODEL_PATH)
    return _model

# [Unverified] Assumptions about the model's expected input:
# - Input image size: 224x224
# - Color channels: 3 (RGB)
# - Pixel scaling: rescaled to [0,1]
IMG_SIZE = (224, 224)

def preprocess_image(image_path):
    """
    Loads an image from disk and preprocesses it for the model.
    """
    img = Image.open(image_path)
    # Convert grayscale to RGB if necessary
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)  # batch dim
    return arr

# [Unverified] Prediction decoding:
# This scaffold assumes the model outputs a single sigmoid probability for 'pneumonia'
# where values close to 1 mean pneumonia and close to 0 mean normal.
def decode_prediction(pred):
    # pred is a numpy array like [[0.87]] or [[0.12]]
    prob = float(pred.ravel()[0])
    label = "Pneumonia" if prob >= 0.5 else "Normal"
    return {"label": label, "probability": prob}

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Accept file from form-data 'file' or JSON base64 is not handled here.
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    filename = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filename)
    try:
        model = load_model()
    except Exception as e:
        return jsonify({"error": f"Failed to load model: {e}"}), 500
    try:
        x = preprocess_image(filename)
        pred = model.predict(x)
        result = decode_prediction(pred)
        return jsonify({"result": result})
    except Exception as e:
        return jsonify({"error": f"Prediction error: {e}"}), 500

if __name__ == "__main__":
    # for local debug only
    app.run(host="0.0.0.0", port=8080, debug=True)