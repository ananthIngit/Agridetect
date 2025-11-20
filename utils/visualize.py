import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# --- Imports from Project Files ---
from models.hybrid_model import CNNPlantNet
from utils.dataset_loader import PlantDataset, get_class_names # Need get_class_names for context
# ----------------------------------

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "data/PlantVillage"
MODEL_PATH = "efficientnet_b0_final.pth" 
# ---------------------

# --- Helper for Denormalization ---
def denormalize_image(img_tensor):
    """Converts a normalized PyTorch tensor back to a displayable NumPy array."""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    # Reshape mean and std for element-wise operation (C, H, W)
    mean = torch.tensor(mean).view(3, 1, 1).to(img_tensor.device)
    std = torch.tensor(std).view(3, 1, 1).to(img_tensor.device)
    
    # Denormalize (undo: (x - mean) / std)
    denorm_img = img_tensor.clone() * std + mean
    
    # Clip to [0, 1] range and convert to NumPy (H, W, C)
    denorm_img = torch.clamp(denorm_img, 0, 1)
    return np.transpose(denorm_img.cpu().numpy(), (1, 2, 0))

def visualize_cam(model, input_tensor, class_names, target_index=None):
    """
    Generates and displays Grad-CAM visualization for the CNN branch.
    
    Args:
        model (nn.Module): The trained CNNPlantNet model.
        input_tensor (torch.Tensor): The preprocessed image tensor (1, C, H, W).
        class_names (list): List of class names for the title.
        target_index (int, optional): The class index to target for visualization. 
                                      If None, the predicted class is used.
    """
    
    # --- CORRECTED TARGET LAYER for EfficientNetV2-M (timm) ---
    # The last block in the 'blocks' list is the final feature extractor.
    target_layer = model.cnn.blocks[-1]
    
    use_cuda = torch.cuda.is_available()
    
    # Ensure model is on the correct device for CAM calculation
    cam_model = model.to('cuda' if use_cuda else 'cpu')
    cam = GradCAM(model=cam_model, target_layers=[target_layer], use_cuda=use_cuda)
    
    # Compute CAM
    input_tensor = input_tensor.to('cuda' if use_cuda else 'cpu')
    
    # If target_index is not provided, use the model's prediction
    if target_index is None:
        with torch.no_grad():
            output = model(input_tensor)
            target_index = output.argmax(dim=1).item()
            
    # Compute the CAM for the specified target index
    grayscale_cam = cam(input_tensor=input_tensor, targets=[torch.nn.functional.one_hot(torch.tensor([target_index]), num_classes=model.classifier[1].out_features).float().to(input_tensor.device)])[0, :]
    
    # Denormalize the input tensor back to RGB for display
    rgb_img = denormalize_image(input_tensor[0])
    
    # Overlay the CAM
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    
    # Display the result
    plt.figure(figsize=(8, 8))
    plt.imshow(visualization)
    
    predicted_class = class_names[target_index]
    plt.title(f"Grad-CAM: Focus for Predicted Class ({predicted_class})")
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    
    # --- 1. Load Data/Classes and Setup Transform ---
    # Use the same evaluation transform
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    try:
        class_names = get_class_names(DATA_DIR)
        num_classes = len(class_names)
        
        # Load the dataset to get a sample image
        full_dataset = PlantDataset(DATA_DIR, transform=test_transform)
        if len(full_dataset) == 0:
            raise FileNotFoundError("Dataset is empty.")
            
        print(f"Found {num_classes} classes and {len(full_dataset)} images.")
        
    except Exception as e:
        print(f"ERROR: Data setup failed. Make sure {DATA_DIR} exists. Error: {e}")
        exit()

    # --- 2. Load Model ---
    try:
        if not os.path.exists(MODEL_PATH):
            print(f"ERROR: Model file not found at {MODEL_PATH}. Run 'python train.py' first.")
            exit()
            
        model = CNNPlantNet(num_classes).to(DEVICE)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        print(f"✅ Model loaded successfully.")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        exit()
        
    # --- 3. Get a Sample Image and Run CAM ---
    
    # We will pick the first image in the dataset for visualization
    sample_index = 0 
    sample_image, sample_label = full_dataset[sample_index]
    
    # Create the required batch dimension (1, C, H, W)
    input_tensor = sample_image.unsqueeze(0) 

    print(f"\nVisualizing focus region for image of: {class_names[sample_label]} (Index: {sample_index})")
    
    # Run visualization
    # We pass the predicted label index as the target (None)
    visualize_cam(model, input_tensor, class_names, target_index=None) 
    
    print("Execution complete. Check the displayed plot.")