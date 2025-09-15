import os
import pickle
import uuid
import csv
from datetime import datetime
from flask import Flask, request, render_template, send_file
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
from tensorflow.keras.models import model_from_json

# ========== SETTINGS ==========
UPLOAD_FOLDER = "static/uploads"
RESULTS_FILE = "results.csv"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
TARGET_SIZE = (128, 128)  # Image resize dimensions

# Initialize Flask
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Create upload folder if not exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model from pickle containing 'architecture' and 'weights'
with open("F:\\Self learning\\garbage-classification\\garbageModel.pkl", "rb") as f:
    model_data = pickle.load(f)

if not isinstance(model_data, dict) or 'architecture' not in model_data or 'weights' not in model_data:
    raise ValueError("Pickle must contain 'architecture' and 'weights' keys.")

# Rebuild Keras model
architecture = model_data['architecture']  # Should be JSON string
weights = model_data['weights']            # Should be list or array

model = model_from_json(architecture)
model.set_weights(weights)

# Check file type
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# Preprocess image (resize + flatten)
def preprocess_image(image_path, target_size=TARGET_SIZE):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = img_array.flatten().reshape(1, -1)
    return img_array

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    filename = None
    error = None

    if request.method == "POST":
        if "file" not in request.files:
            error = "No file part"
            return render_template("index.html", error=error)

        file = request.files["file"]
        if file.filename == "":
            error = "No selected file"
            return render_template("index.html", error=error)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            unique_name = str(uuid.uuid4()) + "_" + filename
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
            file.save(filepath)

            try:
                # Preprocess image
                img_array = preprocess_image(filepath)

                # Predict using Keras model
                pred_probs = model.predict(img_array)
                pred_class = np.argmax(pred_probs, axis=1)[0]  # Convert probabilities to class index
                prediction = str(pred_class)

                # Save result in CSV
                row = [datetime.now().strftime("%Y-%m-%d %H:%M:%S"), filename, prediction]
                file_exists = os.path.isfile(RESULTS_FILE)
                with open(RESULTS_FILE, "a", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    if not file_exists:
                        writer.writerow(["Timestamp", "Image", "Prediction"])
                    writer.writerow(row)
            except Exception as e:
                error = f"Prediction failed: {str(e)}"

    return render_template("index.html", prediction=prediction, filename=filename, error=error)

@app.route("/download_report")
def download_report():
    if os.path.exists(RESULTS_FILE):
        return send_file(RESULTS_FILE, as_attachment=True)
    return "No report found!"

if __name__ == "__main__":
    app.run(debug=True)
