import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from utils.dataset_loader import PlantDataset, get_class_names
from models.hybrid_model import HybridPlantNet
from sklearn.metrics import classification_report

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "data/PlantVillage" 
NUM_EPOCHS = 30
BATCH_SIZE = 16
INITIAL_LR = 5e-5
# Define split ratios
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
# TEST_RATIO is calculated from the remainder
# ---------------------

# --- Data Transformations (Optimized for Pre-trained Models) ---

# Standard ImageNet normalization values (CRUCIAL for transfer learning)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomRotation(15), 
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1), 
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), value=0),
])

test_val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])
# ----------------------------

if __name__ == '__main__':
    
    # --- Data Loading and Splitting ---
    try:
        NUM_WORKERS = 4 if DEVICE == "cuda" else 0
        
        # 1. Load the entire dataset
        full_dataset = PlantDataset(DATA_DIR, transform=test_val_transform)
        
        # 2. Calculate split sizes
        total_size = len(full_dataset)
        train_size = int(TRAIN_RATIO * total_size)
        val_size = int(VAL_RATIO * total_size)
        test_size = total_size - train_size - val_size
        
        # 3. Split the dataset randomly
        train_subset, val_subset, test_subset = random_split(
            full_dataset, [train_size, val_size, test_size]
        )
        
        # Apply the heavy augmentations ONLY to the training subset
        train_subset.dataset.transform = train_transform
        
        # 4. Create separate DataLoaders for each split
        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

        num_classes = len(get_class_names(DATA_DIR))
        
        print(f"Dataset split: Train={train_size}, Validation={val_size}, Test={test_size}. Classes: {num_classes}")
        
    except FileNotFoundError:
        print(f"ERROR: Data directory not found. Please ensure your dataset is at: {DATA_DIR}")
        exit()

    # --- Model, Loss, Optimizer ---
    model = HybridPlantNet(num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=INITIAL_LR)
    
    # NEW: Learning Rate Scheduler (Decreases LR by 10% every 10 epochs)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1) 
    
    print(f"Starting training on {DEVICE} for {NUM_EPOCHS} epochs...")

    # --- Training Loop ---
    best_val_acc = 0.0
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_train_loss = 0
        
        # Training phase
        for batch_idx, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()

        scheduler.step() # Step the scheduler after the epoch
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        correct = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                total_val_loss += criterion(outputs, labels).item()
                
                pred = outputs.argmax(dim=1, keepdim=True)
                correct += pred.eq(labels.view_as(pred)).sum().item()

        val_accuracy = 100. * correct / val_size
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}: Train Loss = {total_train_loss/len(train_loader):.4f} | Val Loss = {total_val_loss/len(val_loader):.4f} | Val Acc: {val_accuracy:.2f}%")

        # Save model if validation accuracy improved
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(model.state_dict(), "hybrid_model_best.pth")
            print(f"--> Saved model with improved Val Acc: {best_val_acc:.2f}%")


    # --- FINAL TEST EVALUATION ---
    print("\n--- Running Final Test Evaluation ---")
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            
            preds = outputs.argmax(dim=1).cpu().numpy()
            labels_np = labels.cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels_np)

    # Use classification_report for detailed metrics
    try:
        class_names = get_class_names(DATA_DIR)
        print(classification_report(all_labels, all_preds, target_names=class_names))
    except Exception as e:
        print(f"Could not print detailed report: {e}")
        print(f"Final Test Accuracy: {classification_report(all_labels, all_preds, output_dict=True)['accuracy']:.4f}")
        
    # --- Save Model ---
    torch.save(model.state_dict(), "hybrid_model_final.pth")
    print(f"✅ Training complete. Final model saved to hybrid_model_final.pth!")