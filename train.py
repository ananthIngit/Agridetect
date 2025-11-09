import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.dataset_loader import PlantDataset, get_class_names
from models.hybrid_model import HybridPlantNet

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# IMPORTANT: Place your PlantVillage class folders (e.g., 'Tomato___Bacterial_spot') inside 'data/PlantVillage'
DATA_DIR = "data/PlantVillage" 
NUM_EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
# ---------------------

# --- Data Transformations ---
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
# ----------------------------

# FIX: Wrap execution code in this block to prevent RuntimeError on Windows multiprocessing
if __name__ == '__main__':
    
    # --- Data Loading ---
    try:
        # Use num_workers=0 if you continue to face PermissionError or multiprocessing issues on Windows.
        # Otherwise, 4 is usually a good setting for performance.
        NUM_WORKERS = 4 if DEVICE == "cuda" else 0
        
        dataset = PlantDataset(DATA_DIR, transform=train_transform)
        num_classes = len(get_class_names(DATA_DIR))
        train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        
        print(f"Dataset loaded: {len(dataset)} images, {num_classes} classes.")
        
    except FileNotFoundError:
        print(f"ERROR: Data directory not found. Please ensure your dataset is at: {DATA_DIR}")
        exit()

    # --- Model, Loss, Optimizer ---
    model = HybridPlantNet(num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Starting training on {DEVICE} for {NUM_EPOCHS} epochs...")

    # --- Training Loop ---
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        
        for batch_idx, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{NUM_EPOCHS}: Loss = {total_loss/len(train_loader):.4f}")

    # --- Save Model ---
    MODEL_PATH = "hybrid_model.pth"
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"✅ Model trained and saved to {MODEL_PATH}!")