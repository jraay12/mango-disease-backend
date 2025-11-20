from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import uvicorn
import os
import glob
import cv2
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
class_names = ['Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Die Back', 
               'Gall Midge', 'Healthy', 'Powdery Mildew', 'Sooty Mould']

# CRITICAL: Same CLAHE preprocessing as training
def apply_clahe(image):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    MUST match the preprocessing used during training
    """
    # Ensure image is in uint8 format (0-255)
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = (image * 255).astype(np.uint8)
    
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l_clahe = clahe.apply(l)
    
    # Merge channels and convert back to RGB
    lab_clahe = cv2.merge([l_clahe, a, b])
    enhanced = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
    
    # Normalize to [0,1] for model input
    return enhanced.astype(np.float32) / 255.0

def find_model_file():
    """Find model file in both .keras and .h5 formats"""
    model_dir = "models" 
    
    # Check in current directory first, then models directory
    possible_paths = [
        os.path.join(model_dir, "best_mango_model.h5"),
    ]
    
    for model_path in possible_paths:
        if os.path.exists(model_path):
            print(f"✓ Found model: {model_path}")
            return model_path
    
    print("✗ No model file found. Checked for:")
    for path in possible_paths:
        print(f"   - {path}")
    return None

def load_class_names():
    """Load class names from JSON if available"""
    global class_names
    try:
        if os.path.exists('class_labels.json'):
            with open('class_labels.json', 'r') as f:
                labels = json.load(f)
                # Convert {0: 'name', 1: 'name'} to list
                class_names = [labels[str(i)] for i in range(len(labels))]
                print(f"✓ Loaded class names from JSON: {class_names}")
                return True
    except Exception as e:
        print(f"Warning: Could not load class_labels.json: {e}")
        print(f"Using default class names: {class_names}")
    return False

def load_model():
    """Load the trained model"""
    global model
    try:
        model_path = find_model_file()
        if not model_path:
            return False
            
        print(f"Loading model from: {model_path}")
        
        # Load model with custom options to avoid warnings
        model = tf.keras.models.load_model(
            model_path,
            compile=True
        )
        
        print("="*60)
        print("✓ Mango Disease Model Loaded Successfully!")
        print(f"  Format: {os.path.splitext(model_path)[1]}")
        print(f"  Input shape: {model.input_shape}")
        print(f"  Output shape: {model.output_shape}")
        print(f"  Classes: {len(class_names)}")
        print("="*60)
        return True
        
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# Load model and class names on startup
load_class_names()
if load_model():
    print("🚀 API Ready! Model is loaded and ready for predictions.\n")
else:
    print("⚠️  API started but model failed to load\n")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict mango disease from uploaded image
    Returns disease classification with confidence scores
    """
    if model is None:
        return {
            "success": False, 
            "error": "Model not loaded. Please check server logs and ensure model file exists."
        }
    
    try:
        print(f"\n📷 Received image: {file.filename}")
        
        # Read and convert image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        original_size = image.size
        
        # Resize to model input size
        image = image.resize((224, 224))
        img_array = np.array(image)
        
        # Apply CLAHE preprocessing (CRITICAL for accuracy)
        print("🔧 Applying CLAHE preprocessing...")
        img_array = apply_clahe(img_array)
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        print("🔍 Analyzing image...")
        predictions = model.predict(img_array, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        
        # Get top 3 predictions
        top_3_idx = np.argsort(predictions[0])[-3:][::-1]
        top_3_predictions = {
            class_names[i]: float(predictions[0][i]) for i in top_3_idx
        }
        
        result_emoji = "✓" if confidence > 0.7 else "⚠️"
        print(f"{result_emoji} Prediction: {class_names[predicted_class_idx]} ({confidence:.2%})")
        
        return {
            "success": True,
            "disease": class_names[predicted_class_idx],
            "confidence": confidence,
            "top_3_predictions": top_3_predictions,
            "all_predictions": {
                class_names[i]: float(predictions[0][i]) for i in range(len(class_names))
            },
            "metadata": {
                "original_size": original_size,
                "processed_size": (224, 224),
                "preprocessing": "CLAHE applied"
            }
        }
    except Exception as e:
        print(f"✗ Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

@app.post("/predict-batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    """
    Predict multiple images at once for efficiency
    """
    if model is None:
        return {"success": False, "error": "Model not loaded"}
    
    results = []
    for file in files:
        result = await predict(file)
        result['filename'] = file.filename
        results.append(result)
    
    return {
        "success": True,
        "count": len(results),
        "results": results
    }

@app.get("/")
async def root():
    return {
        "status": "OK", 
        "message": "Mango Disease Detection API",
        "version": "2.0",
        "model_loaded": model is not None,
        "endpoints": {
            "health": "/health (GET)",
            "predict": "/predict (POST - single image)",
            "predict_batch": "/predict-batch (POST - multiple images)",
            "model_info": "/model-info (GET)",
            "classes": "/classes (GET)"
        }
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy", 
        "model_loaded": model is not None,
        "preprocessing": "CLAHE enabled",
        "classes_count": len(class_names)
    }

@app.get("/model-info")
async def model_info():
    """Endpoint to check model details"""
    if model is None:
        return {"error": "Model not loaded"}
    
    return {
        "model_loaded": True,
        "input_shape": str(model.input_shape),
        "output_shape": str(model.output_shape),
        "num_classes": len(class_names),
        "preprocessing": {
            "clahe": True,
            "clipLimit": 2.0,
            "tileGridSize": [8, 8]
        },
        "image_size": [224, 224]
    }

@app.get("/classes")
async def get_classes():
    """Get all disease classes the model can detect"""
    return {
        "count": len(class_names),
        "classes": class_names,
        "healthy_index": class_names.index("Healthy") if "Healthy" in class_names else None
    }

if __name__ == "__main__":
    print("\n" + "="*60)
    print("Starting Mango Disease Detection API Server")
    print("="*60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)