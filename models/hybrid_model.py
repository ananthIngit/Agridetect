import torch
import torch.nn as nn
import timm

class CNNPlantNet(nn.Module):
    """
    A pure CNN classifier based on the EfficientNet B0 backbone.
    
    We switch to 'efficientnet_b0' to guarantee access to pre-trained weights, 
    resolving the timm RuntimeError with EfficientNetV2.
    """
    def __init__(self, num_classes):
        super(CNNPlantNet, self).__init__()
        
        # --- CNN Backbone: EfficientNet B0 (FINAL FIX) ---
        # Using 'efficientnet_b0' to ensure pretrained weights are available.
        self.cnn = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0) 
        
        # The feature size from EfficientNet B0's final block is 1280.
        FEATURE_DIM = 1280
        
        # --- Classification Head ---
        # Takes the global pooled features (1280) and maps them to the number of output classes.
        self.classifier = nn.Sequential(
            nn.Dropout(0.4), # Dropout for regularization
            nn.Linear(FEATURE_DIM, num_classes)
        )

    def forward(self, x):
        # 1. CNN Feature Extraction
        features = self.cnn.forward_features(x)
        
        # 2. Global Average Pooling (GAP)
        pooled_features = features.mean([2, 3]) 
        
        # 3. Classification
        logits = self.classifier(pooled_features)
        
        return logits