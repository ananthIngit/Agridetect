"""
Flask API server for plant disease detection
This server handles image uploads for prediction.
"""
import os
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import numpy as np
from torchvision import transforms
from models.hybrid_model import HybridPlantNet
from utils.dataset_loader import get_class_names

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# --- Configuration ---
MODEL_PATH = "hybrid_model.pth"
# Point to where the PlantVillage sub-folders (classes) should be located
DATA_DIR = "data/PlantVillage"
# Fallback classes, based on common PlantVillage subset
CLASSES_FALLBACK = ["Tomato_Bacterial_spot", "Tomato_Early_blight", "Tomato_healthy"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ---------------------

# Global model variable
model = None
CLASSES = CLASSES_FALLBACK
NUM_CLASSES = len(CLASSES_FALLBACK)

# --- Load Classes ---
try:
    if os.path.exists(DATA_DIR):
        CLASSES = get_class_names(DATA_DIR)
        NUM_CLASSES = len(CLASSES)
        if NUM_CLASSES > 0 and CLASSES != CLASSES_FALLBACK:
            print(f"✅ Loaded {NUM_CLASSES} classes from dataset directory: {CLASSES}")
        else:
             print(f"⚠️  Dataset directory is empty or malformed. Using fallback classes.")
    else:
        print(f"⚠️  Dataset directory not found at {DATA_DIR}. Using fallback classes.")
except Exception as e:
    print(f"⚠️  Error loading classes: {e}. Using fallback classes.")

# --- Load Model ---
def load_model():
    """Loads the PyTorch HybridPlantNet model."""
    global model
    try:
        if os.path.exists(MODEL_PATH):
            model = HybridPlantNet(num_classes=NUM_CLASSES)
            # IMPORTANT: map_location must be specified to load models trained on GPU/CUDA to CPU if needed
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            model.eval()
            model.to(DEVICE)
            print(f"✅ Model loaded successfully on {DEVICE}")
            return True
        else:
            print(f"⚠️  Model file {MODEL_PATH} not found. Please run 'python train.py' first.")
            return False
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        model = None 
        return False

# --- Preprocessing Transform ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# --- API Routes ---
@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "device": DEVICE,
        "num_classes": NUM_CLASSES,
        "classes": CLASSES
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict plant disease from uploaded image"""
    if model is None:
        return jsonify({
            "success": False,
            "error": "Model not loaded. Please train the model (python train.py) and restart the server."
        }), 503
    
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "No image file provided"}), 400
    
    try:
        # Get image file
        file = request.files['image']
        if file.filename == '':
            return jsonify({"success": False, "error": "No file selected"}), 400
        
        # Read and preprocess image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)
        
        # Make prediction
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            pred_idx = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][pred_idx].item()
        
        pred_class = CLASSES[pred_idx] if pred_idx < len(CLASSES) else f"Unknown Class Index {pred_idx}"
        
        # Get top 3 predictions
        top3_probs, top3_indices = torch.topk(probabilities[0], min(3, NUM_CLASSES))
        top3_predictions = [
            {
                "class": CLASSES[idx] if idx < len(CLASSES) else f"Class {idx}",
                "confidence": prob.item()
            }
            for prob, idx in zip(top3_probs, top3_indices)
        ]
        
        return jsonify({
            "success": True,
            "prediction": pred_class,
            "confidence": confidence,
            "top3": top3_predictions
        })
        
    except Exception as e:
        print(f"Prediction processing error: {e}")
        return jsonify({
            "success": False,
            "error": f"Prediction failed due to internal error: {str(e)}"
        }), 500

@app.route('/api/classes', methods=['GET'])
def get_classes():
    """Get list of available classes"""
    return jsonify({
        "classes": CLASSES,
        "num_classes": NUM_CLASSES
    })

if __name__ == '__main__':
    load_model()
    print(f"📡 Server running on http://localhost:5000")
    print(f"📝 API endpoints:")
    print(f"   - GET  /api/health - Health check")
    print(f"   - POST /api/predict - Predict disease from image")
    print(f"   - GET  /api/classes - Get available classes")
    app.run(host='0.0.0.0', port=5000)