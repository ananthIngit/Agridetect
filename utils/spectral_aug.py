import cv2
import numpy as np
from PIL import Image
import random

def spectral_stress(img):
    """Simulates early stress by modifying vegetation spectra"""
    img = np.array(img)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    
    # Slight drop in chlorophyll (reduce green saturation)
    hsv[..., 1] *= random.uniform(0.6, 0.8)
    
    # Add mild yellow hue to simulate nitrogen deficiency
    hsv[..., 0] += random.uniform(2, 8)
    
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    stressed = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    return Image.fromarray(stressed)