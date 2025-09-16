import os
import pickle
import uuid
import csv
from datetime import datetime
from flask import Flask, request, render_template, send_file, redirect, url_for, session
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
from tensorflow.keras.models import model_from_json

# ================= SETTINGS =================
UPLOAD_FOLDER = "static/uploads"
RESULTS_FILE = "results.csv"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
TARGET_SIZE = (128, 128)
class_names = ["paper", "plastic", "trash", "cardboard", "glass", "metal"]

# Initialize Flask
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.secret_key = "supersecretkey123"  # Needed for session
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load Keras model
with open("garbageModel.pkl", "rb") as f:
    model_data = pickle.load(f)

if not isinstance(model_data, dict) or 'architecture' not in model_data or 'weights' not in model_data:
    raise ValueError("Pickle must contain 'architecture' and 'weights' keys.")

model = model_from_json(model_data['architecture'])
model.set_weights(model_data['weights'])

# ================= HELPERS =================
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path, target_size=TARGET_SIZE):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ================= ROUTES =================
@app.route("/", methods=["GET", "POST"])
def upload_page():
    error = None

    if request.method == "POST":
        if "file" not in request.files:
            error = "No file part in request."
        else:
            file = request.files["file"]
            if file.filename == "":
                error = "No file selected."
            elif allowed_file(file.filename):
                filename = secure_filename(file.filename)
                unique_name = str(uuid.uuid4()) + "_" + filename
                filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
                file.save(filepath)

                try:
                    # Preprocess and predict
                    img_array = preprocess_image(filepath)
                    pred_probs = model.predict(img_array)
                    pred_class_index = np.argmax(pred_probs, axis=1)[0]
                    prediction = class_names[pred_class_index]

                    # Save to CSV
                    row = [datetime.now().strftime("%Y-%m-%d %H:%M:%S"), filename, prediction]
                    file_exists = os.path.isfile(RESULTS_FILE)
                    with open(RESULTS_FILE, "a", newline="") as csvfile:
                        writer = csv.writer(csvfile)
                        if not file_exists:
                            writer.writerow(["Timestamp", "Image", "Prediction"])
                        writer.writerow(row)

                    # Store info in session and redirect to result page
                    session["filename"] = unique_name
                    session["prediction"] = prediction
                    return redirect(url_for("result_page"))

                except Exception as e:
                    error = f"Prediction failed: {str(e)}"

    return render_template("upload.html", error=error)

@app.route("/result")
def result_page():
    filename = session.get("filename")
    prediction = session.get("prediction")

    if not filename or not prediction:
        return redirect(url_for("upload_page"))

    return render_template("result.html", filename=filename, prediction=prediction)

@app.route("/download_report")
def download_report():
    if os.path.exists(RESULTS_FILE):
        return send_file(RESULTS_FILE, as_attachment=True)
    return "No report found!"

# ================= RUN =================
if __name__ == "__main__":
    app.run(debug=True)
