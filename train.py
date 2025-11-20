import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
import timm
from sklearn.metrics import classification_report
import os
import numpy as np

# --- Project Imports ---
from models.hybrid_model import CNNPlantNet 
from utils.dataset_loader import PlantDataset, get_class_names 

# --- Configuration ---
MODEL_NAME = 'efficientnet_b0' 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "data/PlantVillage"
NUM_EPOCHS = 50 
BATCH_SIZE = 16
INITIAL_LR = 1e-6 
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
CHECKPOINT_PATH = "cnnplantnet_best_checkpoint.pth" 
WARMUP_EPOCHS = 5
# ---------------------

# --- Data Transformations ---
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
    transforms.RandomRotation(15), 
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.15, contrast=0.15), 
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    transforms.RandomErasing(p=0.5, scale=(0.05, 0.25), value=0), 
])

test_val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])
# ----------------------------

# --- Custom Warmup Function ---
def adjust_lr(optimizer, epoch, warmup_epochs, initial_lr):
    """Simple linear warmup for the first few epochs."""
    if epoch < warmup_epochs:
        lr = initial_lr * (epoch + 1) / warmup_epochs
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

# --- Utility for Dynamic Class Weight Calculation (Imbalance Fix) ---
def get_class_counts_from_subset(subset, num_classes):
    """
    Counts the number of images per class within a PyTorch Subset object.
    """
    counts = [0] * num_classes
    for idx in subset.indices:
        label = subset.dataset.labels[idx] 
        counts[label] += 1
    return counts

def calculate_inverse_frequency_weights(class_counts):
    """Calculates weights for CrossEntropyLoss based on inverse class frequency."""
    if not class_counts or all(c == 0 for c in class_counts):
        return None
    
    counts = np.array(class_counts, dtype=np.float32)
    inverse_freq = 1.0 / counts
    weights = inverse_freq / np.sum(inverse_freq) * len(class_counts)
    
    # FIX APPLIED HERE: Changed np.float32 to torch.float32
    return torch.tensor(weights, dtype=torch.float32)

if __name__ == '__main__':
    
    # --- Data Loading and Splitting ---
    try:
        NUM_WORKERS = 4 if DEVICE == "cuda" else 0
        
        full_dataset = PlantDataset(DATA_DIR, transform=test_val_transform)
        
        total_size = len(full_dataset)
        if total_size == 0:
            raise FileNotFoundError(f"No images found. Please ensure data is correctly structured at: {DATA_DIR}")

        train_size = int(TRAIN_RATIO * total_size)
        val_size = int(VAL_RATIO * total_size)
        test_size = total_size - train_size - val_size
        
        train_subset, val_subset, test_subset = random_split(
            full_dataset, [train_size, val_size, test_size]
        )
        
        train_subset.dataset.transform = train_transform
        
        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

        class_names = get_class_names(DATA_DIR)
        num_classes = len(class_names)
        
        print(f"Dataset split: Train={train_size}, Validation={val_size}, Test={test_size}. Classes: {num_classes}")
        
    except FileNotFoundError as e:
        print(f"ERROR: Data loading failed. {e}")
        exit()

    # --- Model, Loss, Optimizer ---
    model = CNNPlantNet(num_classes=num_classes).to(DEVICE)

    # --- WEIGHTED LOSS IMPLEMENTATION (Active Imbalance Correction) ---
    
    train_class_counts = get_class_counts_from_subset(train_subset, num_classes)
    print(f"Calculated Training Subset Class Counts: {train_class_counts}")
    
    CLASS_WEIGHTS = calculate_inverse_frequency_weights(train_class_counts)
    
    if CLASS_WEIGHTS is not None and len(CLASS_WEIGHTS) == num_classes:
        print(f"✅ Using Dynamic Weighted Loss for {num_classes} classes.")
        criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS.to(DEVICE))
    else:
        print(f"⚠️ Warning: Class counts could not be calculated/applied. Using unweighted loss.")
        criterion = nn.CrossEntropyLoss()
    # -------------------------------------------------------------------

    optimizer = torch.optim.Adam(model.parameters(), lr=INITIAL_LR, weight_decay=5e-4) 
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS - WARMUP_EPOCHS, eta_min=1e-7)

    print(f"Loaded {MODEL_NAME} CNN. Starting finetuning on {DEVICE} for {NUM_EPOCHS} epochs...")

    # --- Training Loop ---
    best_val_acc = 0.0
    
    for epoch in range(NUM_EPOCHS):
        
        adjust_lr(optimizer, epoch, WARMUP_EPOCHS, INITIAL_LR)
        
        model.train()
        total_train_loss = 0
        
        for batch_idx, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()

        if epoch >= WARMUP_EPOCHS:
            scheduler.step()
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        correct = 0
        
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                total_val_loss += criterion(outputs, labels).item()
                
                pred = outputs.argmax(dim=1)
                correct += pred.eq(labels).sum().item()

        val_accuracy = 100. * correct / val_size
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1}/{NUM_EPOCHS}: Train Loss = {total_train_loss/len(train_loader):.4f} | Val Loss = {total_val_loss/len(val_loader):.4f} | Val Acc: {val_accuracy:.2f}% (LR: {current_lr:.2e})")

        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"--> Saved model with improved Val Acc: {best_val_acc:.2f}% to {CHECKPOINT_PATH}")


    # --- FINAL TEST EVALUATION ---
    print("\n--- Running Final Test Evaluation ---")
    model.eval()
    all_preds = []
    all_labels = []

    try:
        model.load_state_dict(torch.load(CHECKPOINT_PATH))
        print(f"Loaded best weights from {CHECKPOINT_PATH}.")
    except Exception as e:
        print(f"Warning: Could not load best weights. Testing with final epoch weights. Error: {e}")

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            
            preds = outputs.argmax(dim=1).cpu().numpy()
            labels_np = labels.cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels_np)

    try:
        print(classification_report(all_labels, all_preds, target_names=class_names))
    except Exception as e:
        final_acc = classification_report(all_labels, all_preds, output_dict=True)['accuracy']
        print(f"Could not print detailed report (likely due to missing class names): {e}")
        print(f"Final Test Accuracy: {final_acc:.4f}")
        
    # --- Save Final Model ---
    torch.save(model.state_dict(), "efficientnet_b0_final.pth") 
    print(f"✅ Training complete. Final model saved to efficientnet_b0_final.pth!")