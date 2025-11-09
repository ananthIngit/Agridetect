from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt
import numpy as np
import torch

def visualize_cam(model, input_tensor):
    """Generates and displays Grad-CAM visualization for the CNN branch."""
    
    target_layer = model.cnn.layer4[-1] 
    use_cuda = torch.cuda.is_available()
    
    # Move model to CPU if CUDA is not available for CAM calculation stability
    cam_model = model.to('cuda' if use_cuda else 'cpu')
    cam = GradCAM(model=cam_model, target_layers=[target_layer], use_cuda=use_cuda)
    
    # Compute CAM
    input_tensor = input_tensor.to('cuda' if use_cuda else 'cpu')
    grayscale_cam = cam(input_tensor=input_tensor)[0, :]
    
    # Convert input tensor (1, C, H, W) to NumPy RGB image (H, W, C)
    rgb_img = np.transpose(input_tensor[0].cpu().numpy(), (1, 2, 0))
    
    # Overlay the CAM
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    
    # Display the result
    plt.figure(figsize=(6, 6))
    plt.imshow(visualization)
    plt.title("Early Disease Focus Region (Grad-CAM)")
    plt.axis('off')
    plt.show()