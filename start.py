"""
Unified startup script to run both Flask backend (API + ML) and React frontend (Vite)
THIS VERSION INCLUDES FULL RAG/AI ASSISTANT INTEGRATION.
"""
import subprocess
import sys
import os
import time
import webbrowser
import io
from threading import Thread

# --- CORE PYTHON IMPORTS ---
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from torchvision import transforms
from models.hybrid_model import CNNPlantNet 
# ----------------------------------

# --- RAG IMPORTS ADDED ---
from rag_bot import setup_rag_pipeline, create_rag_prompt 
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
# -------------------------

def check_dependencies(silent=False):
    """Check if required Python dependencies are installed"""
    required = ["flask", "torch", "timm", "flask_cors", "pillow"]
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

# Global variables for ML model and RAG access
model = None
QA_CHAIN = None
RETRIEVER = None # <-- RAG Retriever (needed for conditional check)

def start_flask():
    """Start Flask backend server with integrated ML API logic and RAG."""
    global model, QA_CHAIN, RETRIEVER # <-- Retrieve/Set global variables

    # --- FLASK CONFIGURATION & SETUP ---
    app = Flask(__name__)
    CORS(app) # Enable CORS for frontend communication

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

    # --- Load Model Function ---
    def load_model():
        """Loads the PyTorch CNNPlantNet model."""
        global model
        try:
            if os.path.exists(MODEL_PATH):
                model = CNNPlantNet(num_classes=NUM_CLASSES)
                model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
                model.eval()
                model.to(DEVICE)
                print(f"✅ ML Model loaded successfully on {DEVICE}")
                return True
            else:
                print(f"⚠️ Model file {MODEL_PATH} not found. Please run 'python train.py' first.")
                return False
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            model = None 
            return False

    # --- RAG INITIALIZATION FUNCTION ---
    def load_rag():
        """Initializes the RAG components (LLM, Retriever, QA_CHAIN)."""
        global QA_CHAIN, RETRIEVER
        LLM, RETRIEVER = setup_rag_pipeline() 
        
        if LLM and RETRIEVER:
            rag_prompt = create_rag_prompt()
            
            # LCEL Chain Construction (The working version)
            rag_chain_from_docs = (
                RunnablePassthrough.assign(context=(lambda x: x['context']))
                | rag_prompt
                | LLM
                | StrOutputParser()
            )

            QA_CHAIN = (
                {"context": RETRIEVER, "question": RunnablePassthrough()}
                | rag_chain_from_docs
            )
            print("✅ RAG/AI Assistant loaded successfully.")
        else:
            print("❌ RAG/AI Assistant failed to load. Check Ollama server and rag_bot.py logs.")
            
    # --- Preprocessing Transform ---
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # --- Load Model and RAG at Startup ---
    load_model()
    load_rag() 
    
    # ----------------------------------------------------
    # --- API Routes ---
    # ----------------------------------------------------
    
    @app.route('/api/predict', methods=['POST'])
    def predict():
        """Predict plant disease from uploaded image (Existing ML route)"""
        if model is None:
            return jsonify({"success": False, "error": "Model not loaded."}), 503
        
        # ... (Prediction logic remains the same) ...
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

    # --- NEW: RAG CHAT API ROUTE WITH CONDITIONAL LOGIC ---
    @app.route('/api/chat', methods=['POST'])
    def api_chat():
        """Handles conversational requests from the Chat.tsx frontend via local RAG."""
        global QA_CHAIN, RETRIEVER # Access Retriever for conditional check
        
        if QA_CHAIN is None or RETRIEVER is None:
            return jsonify({"error": "AI Assistant not initialized (RAG failed to load)."}), 503

        try:
            data = request.get_json()
            user_message = data.get('message')
            disease_context = data.get('disease_context') 
            
            if not user_message:
                return jsonify({"error": "Missing 'message' in JSON payload."}), 400

            # --- 1. CONDITIONAL RAG CHECK ---
            # Use the retriever to find relevant chunks based on the user's message.
            retrieved_docs = RETRIEVER.invoke(user_message)
            
            # --- 2. DETERMINE MODE ---
            # If no documents are retrieved, or the user is asking a very short general question, 
            # we switch to general conversational mode.
            if not retrieved_docs or len(retrieved_docs) == 0:
                # GENERAL CHAT MODE (Bypass RAG Chain)
                prompt = f"Act as a friendly chat friend specializing in agriculture. Answer this user question conversationally and concisely: {user_message}"
                
                # Directly invoke the LLM for a general/conversational answer
                ai_response = LLM.invoke(prompt) 
            
            else:
                # RAG MODE (Use the full chain for expert advice using context)
                if disease_context:
                    # First question after detection: include context to keep LLM focused
                    full_query = f"The user is dealing with '{disease_context}'. The user is asking a follow-up question: {user_message}"
                else:
                    # Regular farming question that hit the knowledge base
                    full_query = user_message

                # Invoke the RAG chain (which uses the prompt template from Step 1.1)
                ai_response = QA_CHAIN.invoke(full_query)


            return jsonify({
                "response": ai_response.strip(),
                "success": True
            })

        except Exception as e:
            print(f"Error during /api/chat query: {e}")
            return jsonify({"error": f"Internal chat server error: {str(e)}"}), 500

    @app.route('/api/classes', methods=['GET'])
    def get_classes():
        """Get list of available classes"""
        return jsonify({
            "classes": CLASSES,
            "num_classes": NUM_CLASSES
        })
    
    @app.route('/api/health', methods=['GET'])
    def health():
        """Health check endpoint (Updated to include RAG status)"""
        return jsonify({
            "status": "healthy",
            "model_loaded": model is not None,
            "rag_loaded": QA_CHAIN is not None,
            "device": DEVICE,
            "num_classes": NUM_CLASSES,
            "classes": CLASSES
        })
    # --- END API ROUTES ---

    # --- FLASK RUNTIME STARTUP ---
    print("\n=======================================================")
    print(f"📡 Flask Server is Running")
    print(f"   Listening on: http://0.0.0.0:5000")
    print("=======================================================")
    app.run(host='0.0.0.0', port=5000)


def start_react():
    """Start React frontend using Vite"""
    print("🚀 Starting React frontend server (port 5173)...")
    if not os.path.exists("node_modules"):
        print("📦 Installing npm dependencies...")
        npm_command = ["npm", "ci"] if os.path.exists("package-lock.json") else ["npm", "install"]
        try:
            subprocess.run(npm_command, check=True, shell=sys.platform.startswith('win'))
            print("✅ Node.js dependencies installed successfully.")
        except subprocess.CalledProcessError:
            print("❌ Failed to install Node.js dependencies.")
            return

    subprocess.run(["npm", "run", "dev", "--", "--port", "5173"], shell=sys.platform.startswith('win'))

def open_browser():
    """Open browser after a delay"""
    time.sleep(7)
    print("🌐 Opening browser...")
    webbrowser.open("http://localhost:5173")

if __name__ == "__main__":
    print("=" * 50)
    print("🌱 AgriDetect - Starting Unified Servers")
    print("=" * 50)
    
    check_dependencies()
    
    flask_thread = Thread(target=start_flask, daemon=True)
    flask_thread.start()
    
    browser_thread = Thread(target=open_browser, daemon=True)
    browser_thread.start()
    
    try:
        start_react()
    except KeyboardInterrupt:
        print("\n👋 Shutting down servers...")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        sys.exit(1)