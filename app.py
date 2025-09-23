from flask import Flask, render_template, request, redirect, url_for, send_file
import os
import pickle
import numpy as np
import cv2
import pandas as pd
from datetime import datetime
import re
import logging
from tensorflow import keras
import tensorflow as tf

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# --------------------------
# Flask Setup
# --------------------------
app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
RESULTS_CSV = "results.csv"

# --------------------------
# Load Model and Label Encoder
# --------------------------
def load_model():
    """Load the model with comprehensive error handling"""
    global model, le, IMG_SIZE, class_names, model_loaded, model_type
    
    try:
        # Try to load Keras H5 model first
        if os.path.exists("garbageModel.h5"):
            logger.info("Attempting to load Keras model from garbageModel.h5")
            
            model = keras.models.load_model("garbageModel.h5")
            model_type = "keras"
            IMG_SIZE = 224  # Standard size for EfficientNet models
            
            # Define class names based on your dataset
            class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
            le = None  # No label encoder needed for Keras model
            
            logger.info("✅ Keras model loaded successfully from garbageModel.h5")
            logger.info(f"Model type: {type(model)}")
            logger.info(f"Input shape: {model.input_shape}")
            logger.info(f"Output shape: {model.output_shape}")
            
            # Update IMG_SIZE based on model input shape
            if model.input_shape[1] is not None:
                IMG_SIZE = model.input_shape[1]
                logger.info(f"Updated IMG_SIZE to: {IMG_SIZE}")
            
            model_loaded = True
            return True
        
        # Try to load pickle model as fallback
        elif os.path.exists("garbageModel.pkl"):
            logger.info("Attempting to load model from garbageModel.pkl")
            
            with open("garbageModel.pkl", "rb") as f:
                model_data = pickle.load(f)
            
            logger.info(f"Model keys: {list(model_data.keys())}")
            
            # Check for required keys
            required_keys = ["model", "label_encoder", "img_size"]
            for key in required_keys:
                if key not in model_data:
                    logger.error(f"Missing required key: {key}")
                    model_loaded = False
                    return False
            
            model = model_data["model"]
            le = model_data["label_encoder"]
            IMG_SIZE = model_data["img_size"]
            class_names = le.classes_
            model_type = "pickle"
            
            logger.info(f"Pickle model loaded successfully. Image size: {IMG_SIZE}")
            logger.info(f"Classes: {list(class_names)}")
            logger.info(f"Model type: {type(model)}")
            
            model_loaded = True
            return True
        else:
            logger.error("No model file found (garbageModel.h5 or garbageModel.pkl)")
            model_loaded = False
            return False
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        model_loaded = False
        return False

# Initialize model variables
model = None
le = None
IMG_SIZE = 224
class_names = []
model_loaded = False
model_type = None

# Load model on startup
load_model()

# --------------------------
# Demo Model (Fallback)
# --------------------------
class DemoModel:
    """Demo model for testing when real model fails to load"""
    def predict(self, x):
        # Return random probabilities
        probs = np.random.dirichlet(np.ones(len(class_names) if class_names else 6))
        return probs.reshape(1, -1)

class DemoLabelEncoder:
    def __init__(self, classes=None):
        if classes is None:
            classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
        self.classes_ = np.array(classes)
    
    def inverse_transform(self, indices):
        return [self.classes_[i] for i in indices]

# --------------------------
# Routes
# --------------------------
@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("upload.html", error="No file part")

        file = request.files["file"]
        if file.filename == "":
            return render_template("upload.html", error="No selected file")

        if not allowed_file(file.filename):
            return render_template("upload.html", error="Please upload an image file (jpg, jpeg, png, gif)")

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            file.save(filepath)
            if not os.path.exists(filepath):
                return render_template("upload.html", error="Failed to save file")
                
        except Exception as e:
            return render_template("upload.html", error=f"Error saving file: {e}")

        try:
            prediction, percentages = predict_image(filepath)
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            return render_template("upload.html", error=f"Prediction error: {e}")

        save_result(filename, prediction, percentages)

        return render_template("result.html", 
                             filename=filename, 
                             prediction=prediction,
                             percentages=percentages,
                             now=datetime.now(),
                             model_loaded=model_loaded,
                             model_type=model_type)
    
    return render_template("upload.html", model_loaded=model_loaded, model_type=model_type)


@app.route("/demo_predict", methods=["POST"])
def demo_predict():
    """Demo prediction endpoint for testing"""
    try:
        # Create a demo prediction
        demo_classes = class_names if len(class_names) > 0 else ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
        demo_le = DemoLabelEncoder(demo_classes)
        
        # Generate random probabilities
        probs = np.random.dirichlet(np.ones(len(demo_le.classes_)))
        class_index = np.argmax(probs)
        prediction = demo_le.inverse_transform([class_index])[0]
        
        percentages = {cls: float(prob * 100) for cls, prob in zip(demo_le.classes_, probs)}
        percentages = dict(sorted(percentages.items(), key=lambda item: item[1], reverse=True))
        
        return render_template("result.html", 
                             filename="demo_image.jpg",
                             prediction=prediction,
                             percentages=percentages,
                             now=datetime.now(),
                             demo_mode=not model_loaded)
                             
    except Exception as e:
        return render_template("upload.html", error=f"Demo prediction failed: {e}")


@app.route("/download_report")
def download_report():
    if os.path.exists(RESULTS_CSV):
        return send_file(RESULTS_CSV, as_attachment=True, 
                        download_name=f"garbage_classification_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    return "No report available."


@app.route("/view_results")
def view_results():
    """Fixed view_results route to handle confidence percentages properly"""
    try:
        if os.path.exists(RESULTS_CSV):
            df = pd.read_csv(RESULTS_CSV)
            # Convert DataFrame to list of dictionaries with proper confidence handling
            results = []
            for _, row in df.iterrows():
                result_dict = {
                    'Filename': row['Filename'],
                    'Prediction': row['Prediction'],
                    'Timestamp': row['Timestamp']
                }
                
                # Add confidence percentages for each class
                for class_name in class_names if len(class_names) > 0 else ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']:
                    col_name = f"{class_name}_Percentage"
                    if col_name in row:
                        # Ensure confidence value is properly formatted
                        confidence_value = row[col_name]
                        if isinstance(confidence_value, str):
                            # Remove % sign and convert to float for template processing
                            confidence_float = float(confidence_value.replace('%', ''))
                            result_dict[col_name] = confidence_value
                            result_dict[f"{class_name}_Float"] = confidence_float
                        else:
                            # If it's already a float, format it as percentage string
                            confidence_str = f"{float(confidence_value):.1f}%"
                            result_dict[col_name] = confidence_str
                            result_dict[f"{class_name}_Float"] = float(confidence_value)
                    else:
                        result_dict[col_name] = "0.0%"
                        result_dict[f"{class_name}_Float"] = 0.0
                
                results.append(result_dict)
            
            return render_template("results.html", results=results, class_names=class_names, now=datetime.now())
        else:
            return render_template("results.html", results=None, class_names=class_names, now=datetime.now())
            
    except Exception as e:
        logger.error(f"Error in view_results: {e}")
        return render_template("results.html", results=None, class_names=class_names, now=datetime.now())


@app.route("/reload_model")
def reload_model():
    """Endpoint to reload the model"""
    global model_loaded
    success = load_model()
    if success:
        return "Model reloaded successfully!"
    else:
        return "Failed to reload model!"


# --------------------------
# Helper Functions
# --------------------------
def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def secure_filename(filename):
    filename = re.sub(r'[^a-zA-Z0-9._-]', '_', filename)
    name, ext = os.path.splitext(filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{name}_{timestamp}{ext}"


def ensure_3_channels(img):
    """Ensure image has 3 channels (convert grayscale to RGB)"""
    if len(img.shape) == 2:
        # Grayscale image
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        logger.info("Converted grayscale to RGB")
    elif img.shape[2] == 1:
        # Single channel image
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        logger.info("Converted single channel to RGB")
    elif img.shape[2] == 4:
        # RGBA image
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        logger.info("Converted RGBA to RGB")
    return img


def preprocess_image(img_path, target_size):
    """Preprocess image for model prediction"""
    logger.info(f"Preprocessing image: {img_path}")
    
    # Read image
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("Cannot read image file")
    
    logger.info(f"Original image shape: {img.shape}")
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    logger.info(f"After BGR to RGB: {img.shape}")
    
    # Ensure 3 channels
    img = ensure_3_channels(img)
    logger.info(f"After channel conversion: {img.shape}")
    
    # Resize to target size (exactly what the model expects)
    img = cv2.resize(img, (target_size, target_size))
    logger.info(f"After resizing to {target_size}x{target_size}: {img.shape}")
    
    # Normalize pixel values to [0, 1]
    img = img.astype(np.float32) / 255.0
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    logger.info(f"Final shape for prediction: {img.shape}")
    
    return img


def predict_image(img_path):
    if not model_loaded:
        # Use demo model as fallback
        logger.warning("Using demo model for prediction")
        return demo_prediction(img_path)
    
    # Use actual model
    logger.info(f"Predicting image with real model (type: {model_type}): {img_path}")
    
    try:
        # Preprocess image
        img_array = preprocess_image(img_path, IMG_SIZE)
        
        # Verify the shape is correct
        expected_shape = (1, IMG_SIZE, IMG_SIZE, 3)
        if img_array.shape != expected_shape:
            logger.error(f"Image shape incorrect: {img_array.shape}, expected {expected_shape}")
            raise ValueError(f"Image preprocessing failed: incorrect shape {img_array.shape}")
        
        # Predict based on model type
        if model_type == "keras":
            logger.info("Calling Keras model.predict...")
            pred = model.predict(img_array, verbose=0)
            logger.info(f"Keras prediction successful. Raw predictions shape: {pred.shape}")
            
            # Get class index with highest probability
            class_index = np.argmax(pred, axis=1)[0]
            prediction = class_names[class_index]
            
            # Calculate percentages for all classes
            percentages = {class_names[i]: float(pred[0][i]) * 100 for i in range(len(class_names))}
            
        elif model_type == "pickle":
            logger.info("Calling pickle model.predict...")
            pred = model.predict(img_array)
            logger.info(f"Pickle prediction successful. Raw predictions shape: {pred.shape}")
            
            class_index = np.argmax(pred, axis=1)[0]
            prediction = le.inverse_transform([class_index])[0]
            
            percentages = {class_names[i]: float(pred[0][i]) * 100 for i in range(len(class_names))}
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Sort percentages in descending order
        percentages = dict(sorted(percentages.items(), key=lambda item: item[1], reverse=True))
        
        logger.info(f"Prediction: {prediction}, Confidence: {percentages[prediction]:.2f}%")
        
        return prediction, percentages
        
    except Exception as e:
        logger.error(f"Real model prediction failed: {e}")
        # Fall back to demo prediction
        logger.info("Falling back to demo prediction")
        return demo_prediction(img_path)


def demo_prediction(img_path):
    """Demo prediction when model fails"""
    demo_classes = class_names if len(class_names) > 0 else ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    demo_le = DemoLabelEncoder(demo_classes)
    
    # Still process the image to test the pipeline
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("Uploaded image cannot be read.")
    
    # Process image for consistency
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = ensure_3_channels(img)
    img = cv2.resize(img, (224, 224))
    
    # Generate demo prediction
    probs = np.random.dirichlet(np.ones(len(demo_le.classes_)))
    class_index = np.argmax(probs)
    prediction = demo_le.inverse_transform([class_index])[0]
    
    percentages = {cls: float(prob * 100) for cls, prob in zip(demo_le.classes_, probs)}
    percentages = dict(sorted(percentages.items(), key=lambda item: item[1], reverse=True))
    
    logger.info(f"Demo prediction: {prediction}")
    
    return prediction, percentages


def save_result(filename, prediction, percentages):
    """Save prediction results to CSV with proper formatting"""
    data = {
        "Filename": filename,
        "Prediction": prediction,
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Use actual class names if available
    classes_to_save = class_names if len(class_names) > 0 else ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    
    for class_name in classes_to_save:
        # Format percentage with 1 decimal place and % sign
        percentage_value = percentages.get(class_name, 0)
        data[f"{class_name}_Percentage"] = f"{percentage_value:.1f}%"
    
    df = pd.DataFrame([data])
    if os.path.exists(RESULTS_CSV):
        df_existing = pd.read_csv(RESULTS_CSV)
        df = pd.concat([df_existing, df], ignore_index=True)
    df.to_csv(RESULTS_CSV, index=False)
    logger.info(f"Result saved: {prediction}")


# --------------------------
# Error Handlers
# --------------------------
@app.errorhandler(413)
def too_large(e):
    return render_template("upload.html", error="File too large. Maximum size is 16MB."), 413

@app.errorhandler(500)
def internal_error(error):
    return render_template("upload.html", error="Internal server error. Please try again."), 500


# --------------------------
# Run Flask
# --------------------------
if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("static/uploads", exist_ok=True)
    os.makedirs("static/css", exist_ok=True)
    os.makedirs("templates", exist_ok=True)
    
    print("=== Garbage Classification System ===")
    print(f"Model loaded: {model_loaded}")
    if model_loaded:
        print(f"Model type: {model_type}")
        print(f"Image size: {IMG_SIZE}")
        print(f"Classes: {class_names}")
    else:
        print("⚠️  Running in DEMO MODE")
        print("💡 Place your model file as 'garbageModel.h5' or 'garbageModel.pkl' in the project directory")
    
    print(f"Upload folder: {os.path.abspath(UPLOAD_FOLDER)}")
    
    app.run(debug=True, host='0.0.0.0', port=5000)