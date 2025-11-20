import os
import random
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

# --- Placeholder for spectral_stress ---
# This is a temporary function to ensure the file runs without the external utility.
# In a full project, you must ensure your 'spectral_aug.py' file is present.
def spectral_stress(img):
    """Placeholder for applying spectral stress augmentation."""
    # Since we can't implement the full logic, we return the original image.
    # In your actual 'spectral_aug.py', this should contain the detailed color modification logic.
    return img

# --- CONSTANT FOR SAMPLING ---
# Limits the maximum number of images loaded per class to mitigate imbalance.
MAX_SAMPLES_PER_CLASS = 500
# -----------------------------

class PlantDataset(Dataset):
    def __init__(self, root, transform=None):
        self.imgs, self.labels = [], []
        self.transform = transform
        self.root = root
        
        # --- FIX: Ensure classes are loaded in a deterministic (sorted) order ---
        # This order must match the order in app.py and train.py's expectation.
        valid_classes = sorted([
            cls for cls in os.listdir(root) 
            if os.path.isdir(os.path.join(root, cls))
        ])

        self.class_to_idx = {
            cls: idx 
            for idx, cls in enumerate(valid_classes) 
        }
        
        # Load all image paths and labels
        for cls in valid_classes:
            folder = os.path.join(root, cls)
            
            # 1. Get all files in the folder (filtering out potential hidden files)
            all_files = [
                f for f in os.listdir(folder) 
                if os.path.isfile(os.path.join(folder, f)) and not f.startswith('.')
            ]
            
            # 2. Randomly sample up to MAX_SAMPLES_PER_CLASS files
            # This handles class imbalance by downsampling large classes.
            sampled_files = random.sample(all_files, min(len(all_files), MAX_SAMPLES_PER_CLASS))
            
            # 3. Collect the paths and labels for the sampled files
            for f in sampled_files:
                filepath = os.path.join(folder, f)
                self.imgs.append(filepath)
                self.labels.append(self.class_to_idx[cls])

    def __getitem__(self, idx):
        path = self.imgs[idx]
        label = self.labels[idx]
        
        # Open image and convert to RGB (standard input format)
        img = Image.open(path).convert('RGB') 

        # 25% chance to apply spectral early-stress, skip 'healthy' images
        # This is a domain-specific augmentation to make the model robust to subtle symptoms.
        if 'healthy' not in path.lower() and random.random() < 0.25:
            img = spectral_stress(img)

        if self.transform:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.imgs)

def get_class_names(root):
    """Retrieves a sorted list of directory names (classes) from the dataset root folder."""
    return sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])