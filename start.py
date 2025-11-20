"""
Unified startup script to run both Flask backend (API + ML) and React frontend (Vite)
The Flask API logic has been fully integrated into the start_flask function.
"""
import subprocess
import sys
import os
import time
import webbrowser
from threading import Thread

# --- IMPORTS MERGED FROM APP.PY ---
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import numpy as np
from torchvision import transforms
# CRITICAL: Assumes CNNPlantNet is defined in models/hybrid_model.py
from models.hybrid_model import CNNPlantNet 
# ----------------------------------

def check_dependencies(silent=False):
    """Check if required Python dependencies are installed"""
    # Check for core packages used by the ML and API parts
    required = ["flask", "torch", "timm", "opencv-python", "flask_cors", "pillow"]
    missing = []
    
    for module in required:
        try:
            __import__(module)
        except ImportError:
            missing.append(module)

    if missing:
        if not silent:
            print(f"❌ Missing Python dependencies: {', '.join(missing)}")
            print("Please run: pip install -r requirements.txt to install them.")
        return False
    
    if not silent:
        print("✅ Python dependencies found")
    return True

def start_flask():
    """Start Flask backend server with integrated ML API logic"""
    # --- FLASK CONFIGURATION & SETUP (Integrated from app.py) ---
    app = Flask(__name__)
    CORS(app)

    MODEL_PATH = "efficientnet_b0_final.pth"
    
    # Definitive list of 18 classes used for prediction mapping
    CLASSES = [
        "Citrus Black spot", "Citrus canker", "Citrus greening", "Citrus Healthy",
        "Corn Common rust", "Corn Gray leaf spot", "Corn Healthy", "Corn Northern Leaf Blight",
        "Grape Black Measles", "Grape Black rot", "Grape Healthy", "Grape Isariopsis Leaf Spot",
        "Potato Early blight", "Potato Healthy", "Potato Late blight",
        "Tomato_Early_blight", "Tomato_healthy", "Tomato_Late_blight"
    ]
    NUM_CLASSES = len(CLASSES)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Global variables for model and classes access across routes
    global model 
    model = None

    # --- Load Model Function ---
    def load_model():
        """Loads the PyTorch CNNPlantNet model."""
        global model
        try:
            if os.path.exists(MODEL_PATH):
                # Initialize the model using the correct CNNPlantNet class
                model = CNNPlantNet(num_classes=NUM_CLASSES)
                model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
                model.eval()
                model.to(DEVICE)
                print(f"✅ Model loaded successfully on {DEVICE}")
                return True
            else:
                print(f"⚠️ Model file {MODEL_PATH} not found. Please run 'python train.py' first.")
                return False
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            model = None 
            return False

    # --- Preprocessing Transform ---
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
            return jsonify({"success": False, "error": "Model not loaded."}), 503
        
        if 'image' not in request.files:
            return jsonify({"success": False, "error": "No image file provided"}), 400
        
        try:
            file = request.files['image']
            image_bytes = file.read()
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            input_tensor = transform(image).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                output = model(input_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                pred_idx = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][pred_idx].item()
                
                pred_class = CLASSES[pred_idx]
                
                top3_probs, top3_indices = torch.topk(probabilities[0], min(3, NUM_CLASSES))
                top3_predictions = [
                    {"class": CLASSES[idx], "confidence": f"{prob.item() * 100:.2f}%"}
                    for prob, idx in zip(top3_probs, top3_indices)
                ]
            
            return jsonify({
                "success": True,
                "prediction": pred_class,
                "confidence": f"{confidence * 100:.2f}%",
                "top3": top3_predictions,
                "all_classes": CLASSES
            })
            
        except Exception as e:
            print(f"Prediction processing error: {e}")
            return jsonify({"success": False, "error": f"Prediction failed: {str(e)}"}), 500

    @app.route('/api/classes', methods=['GET'])
    def get_classes():
        """Get list of available classes"""
        return jsonify({
            "classes": CLASSES,
            "num_classes": NUM_CLASSES
        })
    # --- END API ROUTES ---

    # --- FLASK RUNTIME STARTUP ---
    load_model()
    print("\n=======================================================")
    print(f"📡 Flask Server is Running")
    print(f"   Listening on: http://0.0.0.0:5000")
    print("=======================================================")
    # Starts the Flask server and blocks the thread
    app.run(host='0.0.0.0', port=5000)


def start_react():
    """Start React frontend using Vite"""
    print("🚀 Starting React frontend server (port 5173)...")
    # Check if node_modules exists
    if not os.path.exists("node_modules"):
        print("📦 Installing npm dependencies...")
        npm_command = ["npm", "ci"] if os.path.exists("package-lock.json") else ["npm", "install"]
        try:
            # Use shell=True on Windows to handle npm commands better
            subprocess.run(npm_command, check=True, shell=sys.platform.startswith('win'))
            print("✅ Node.js dependencies installed successfully.")
        except subprocess.CalledProcessError:
            print("❌ Failed to install Node.js dependencies.")
            return

    # Start Vite server (blocks the thread)
    subprocess.run(["npm", "run", "dev", "--", "--port", "5173"], shell=sys.platform.startswith('win'))

def open_browser():
    """Open browser after a delay"""
    time.sleep(7)  # Wait longer for both servers to stabilize
    print("🌐 Opening browser...")
    webbrowser.open("http://localhost:5173")

if __name__ == "__main__":
    print("=" * 50)
    print("🌱 AgriDetect - Starting Unified Servers")
    print("=" * 50)
    
    # Check dependencies before starting
    check_dependencies()
    
    # 1. Start Flask (ML/API) in a separate thread
    flask_thread = Thread(target=start_flask, daemon=True)
    flask_thread.start()
    
    # 2. Start browser opener in a separate thread
    browser_thread = Thread(target=open_browser, daemon=True)
    browser_thread.start()
    
    # 3. Start React (Frontend) - this will block the main thread until interrupted
    try:
        start_react()
    except KeyboardInterrupt:
        print("\n👋 Shutting down servers...")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        sys.exit(1)