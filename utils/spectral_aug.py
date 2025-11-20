import cv2
import numpy as np
from PIL import Image
import random

def spectral_stress(img):
    """
    Simulates early stress by modifying vegetation spectra.
    It applies a slight shift toward yellow hue and reduced green saturation 
    to make the model robust to subtle, early-stage disease symptoms.
    
    Args:
        img (PIL.Image): The input image to be augmented.
        
    Returns:
        PIL.Image: The spectrally stressed image.
    """
    # Convert PIL Image to NumPy array for OpenCV processing
    img = np.array(img)
    
    # Convert RGB to HSV (Hue, Saturation, Value)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    
    # 1. Slight drop in chlorophyll (reduce green saturation)
    # Saturation (index 1) controls the intensity/purity of the color. Lowering it fades the green.
    hsv[..., 1] *= random.uniform(0.6, 0.8)
    
    # 2. Add mild yellow hue to simulate nitrogen deficiency
    # Hue (index 0) controls the color tone. Adding a small positive value shifts it towards yellow/orange.
    hsv[..., 0] += random.uniform(2, 8)
    
    # Ensure values stay within the valid range (0-255) and convert back to uint8
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    stressed = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    # Convert back to PIL Image format
    return Image.fromarray(stressed)