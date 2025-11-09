import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.dataset_loader import PlantDataset, get_class_names
from models.hybrid_model import HybridPlantNet
import os

# --- Metrics and Plotting ---
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "data/PlantVillage"
MODEL_PATH = "hybrid_model.pth" # Your trained model
BATCH_SIZE = 16
# ---------------------

# --- Validation Transform (Simple, no augmentations) ---
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
# ----------------------------

if __name__ == '__main__':
    
    # --- 1. Load Class Names ---
    try:
        if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
             print(f"ERROR: Data directory not found or is empty at: {DATA_DIR}")
             print("Please place your class folders (e.g., 'Tomato_healthy') inside 'data/PlantVillage'.")
             exit()
             
        class_names = get_class_names(DATA_DIR)
        num_classes = len(class_names)
        
        if num_classes == 0:
             print(f"ERROR: No class folders found in {DATA_DIR}")
             exit()

        print(f"Found {num_classes} classes: {class_names}")

    except Exception as e:
        print(f"Error reading data directory: {e}")
        exit()

    # --- 2. Load Model ---
    try:
        if not os.path.exists(MODEL_PATH):
            print(f"ERROR: Model file not found at {MODEL_PATH}")
            print("Please train the model first by running 'python train.py'")
            exit()
            
        model = HybridPlantNet(num_classes).to(DEVICE)
        # Load the saved model weights
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"✅ Model loaded successfully from {MODEL_PATH} on {DEVICE}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("This often means your 'hybrid_model.pth' was trained on a DIFFERENT number of classes")
        print(f"Model file might be expecting a different number than the {num_classes} classes found in your data folder.")
        exit()

    # --- 3. Load Dataset ---
    try:
        # We will test on the *entire* dataset
        val_dataset = PlantDataset(DATA_DIR, transform=val_transform)
        NUM_WORKERS = 4 if DEVICE == "cuda" else 0
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        print(f"✅ Evaluation dataset loaded: {len(val_dataset)} images.")
        
    except Exception as e:
        print(f"ERROR: Data loading failed. {e}")
        exit()

    #
    # --- 4. Run Evaluation ---
    #
    
    print("\n" + "="*50)
    print("📊 Starting Model Evaluation...")
    print("="*50)
    
    model.eval() # Set model to evaluation mode
    all_labels = []
    all_preds = []
    
    with torch.no_grad(): # No gradients needed
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            _, predicted = torch.max(outputs, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    # --- 5. Print Metrics ---
    print("📈 Classification Report:")
    # This prints Accuracy, Precision, F1-Score, and Recall all at once!
    report = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)
    print(report)
    
    # --- 6. Plot Graph (Confusion Matrix) ---
    print("📊 Plotting Confusion Matrix...")
    try:
        fig, ax = plt.subplots(figsize=(15, 15))
        
        cm_display = ConfusionMatrixDisplay.from_predictions(
            all_labels, 
            all_preds, 
            labels=np.arange(len(class_names)),
            display_labels=class_names,
            ax=ax
        )
        
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.title("Confusion Matrix")
        
        print("✅ Done! Showing graph...")
        plt.show()
        
    except Exception as e:
        print(f"⚠️ Could not plot confusion matrix. Error: {e}")