import os
import random
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from utils.spectral_aug import spectral_stress

# --- CONSTANT FOR SAMPLING ---
MAX_SAMPLES_PER_CLASS = 250
# -----------------------------

class PlantDataset(Dataset):
    def __init__(self, root, transform=None):
        self.imgs, self.labels = [], []
        self.transform = transform
        
        # Determine valid class folders
        self.class_to_idx = {
            cls: idx 
            for idx, cls in enumerate(os.listdir(root)) 
            if os.path.isdir(os.path.join(root, cls))
        }
        
        # Load all image paths and labels
        for cls in self.class_to_idx:
            folder = os.path.join(root, cls)
            
            # 1. Get all files in the folder
            all_files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
            
            # 2. Randomly sample up to MAX_SAMPLES_PER_CLASS files
            # This ensures we don't load the entire folder if it's too large.
            sampled_files = random.sample(all_files, min(len(all_files), MAX_SAMPLES_PER_CLASS))
            
            # 3. Collect the paths and labels for the sampled files
            for f in sampled_files:
                filepath = os.path.join(folder, f)
                self.imgs.append(filepath)
                self.labels.append(self.class_to_idx[cls])

    def __getitem__(self, idx):
        path = self.imgs[idx]
        label = self.labels[idx]
        
        img = Image.open(path).convert('RGB') 

        # 25% chance to apply spectral early-stress, skip 'healthy' images
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