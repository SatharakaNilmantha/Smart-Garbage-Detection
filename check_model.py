import pickle
import numpy as np
import os

def check_model_file():
    print("=== Garbage Model Diagnostic ===")
    
    # Check if file exists
    if not os.path.exists("garbageModel.pkl"):
        print("❌ ERROR: garbageModel.pkl file not found!")
        print("Current directory:", os.getcwd())
        print("Files in current directory:", os.listdir('.'))
        return False
    
    print("✅ garbageModel.pkl file exists")
    print("File size:", os.path.getsize("garbageModel.pkl"), "bytes")
    
    try:
        # Try to load the model
        with open("garbageModel.pkl", "rb") as f:
            model_data = pickle.load(f)
        
        print("✅ Model loaded successfully!")
        print("Keys in model_data:", list(model_data.keys()))
        
        # Check each key
        for key in ["model", "label_encoder", "img_size"]:
            if key in model_data:
                print(f"✅ {key}: Found")
                if key == "model":
                    print(f"   Model type: {type(model_data[key])}")
                    print(f"   Model summary: {model_data[key]}")
                elif key == "label_encoder":
                    print(f"   Classes: {model_data[key].classes_}")
                elif key == "img_size":
                    print(f"   Image size: {model_data[key]}")
            else:
                print(f"❌ {key}: Missing")
                
        return True
        
    except Exception as e:
        print(f"❌ ERROR loading model: {e}")
        return False

if __name__ == "__main__":
    check_model_file()