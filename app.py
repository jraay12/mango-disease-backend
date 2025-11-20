from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import uvicorn
import os
import glob

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

def find_model_file():
    """Find model file in both .keras and .h5 formats"""
    # ilisa ang directory below
    model_dir = "models" 
    
    possible_paths = [
        os.path.join(model_dir, "my_mdeol_2.keras"),
        # os.path.join(model_dir, "mobilenetv2_mango_best_finetuned.h5"),
        # # Also check for any .keras or .h5 files in the directory
        # *glob.glob(os.path.join(model_dir, "*.keras")),
        # *glob.glob(os.path.join(model_dir, "*.h5"))
    ]
    
    for model_path in possible_paths:
        if os.path.exists(model_path):
            print(f"Found model: {model_path}")
            return model_path
    
    print("No model file found. Checked for:")
    for path in possible_paths[:2]:
        print(f"   - {path}")
    return None

def load_model():
    """Load the trained model in either .keras or .h5 format"""
    global model
    try:
        model_path = find_model_file()
        if not model_path:
            return False
            
        print(f"Loading model from: {model_path}")
        
        model = tf.keras.models.load_model(model_path)
        
        print("Mango Disease Model Loaded Successfully!")
        print(f"Model format: {os.path.splitext(model_path)[1]}")
        print(f"Model input shape: {model.input_shape}")
        print(f"Model output shape: {model.output_shape}")
        return True
        
    except Exception as e:
        print(f"Model loading failed: {e}")
        print("Try converting .h5 to .keras format if having issues")
        return False

if load_model():
    print("API Ready! Model is loaded and ready for predictions.")
else:
    print("API started but model failed to load")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return {"success": False, "error": "Model not loaded. Please check server logs."}
    
    try:
        print(f"Received image: {file.filename}")
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        image = image.resize((224, 224))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        print("Analyzing image...")
        predictions = model.predict(img_array, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        
        print(f"Prediction: {class_names[predicted_class_idx]} ({confidence:.2%})")
        
        return {
            "success": True,
            "disease": class_names[predicted_class_idx],
            "confidence": confidence,
            "all_predictions": {
                class_names[i]: float(predictions[0][i]) for i in range(len(class_names))
            }
        }
    except Exception as e:
        print(f"Prediction error: {e}")
        return {"success": False, "error": str(e)}

@app.get("/")
async def root():
    return {
        "status": "OK", 
        "message": "Mango Disease Detection API",
        "model_loaded": model is not None,
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)"
        }
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy", 
        "model_loaded": model is not None,
        "timestamp": np.datetime64('now').astype(str)
    }

@app.get("/model-info")
async def model_info():
    """Endpoint to check model details"""
    if model is None:
        return {"error": "Model not loaded"}
    
    return {
        "model_loaded": True,
        "input_shape": model.input_shape,
        "output_shape": model.output_shape,
        "num_classes": len(class_names),
        "classes": class_names
    }

if __name__ == "__main__":
    print("\n" + "="*60)
    print("Starting Mango Disease Detection API Server")
    print("="*60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)