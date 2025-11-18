import torch
import torch.nn as nn
import timm

<<<<<<< HEAD
# Spectral Attention remains the same, as it's a structural component
=======
>>>>>>> 049551bb1212d2b363f8ae263f6df3e07cef2aeb
class SpectralAttention(nn.Module):
    def __init__(self, channels):
        super(SpectralAttention, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels//4),
            nn.ReLU(),
            nn.Linear(channels//4, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
<<<<<<< HEAD
        w = x.mean(dim=[2,3])
        w = self.fc(w).unsqueeze(-1).unsqueeze(-1)
=======
        w = x.mean(dim=[2,3])  # Global avg pooling (B, C)
        w = self.fc(w).unsqueeze(-1).unsqueeze(-1) # Output (B, C, 1, 1)
>>>>>>> 049551bb1212d2b363f8ae263f6df3e07cef2aeb
        return x * w

class HybridPlantNet(nn.Module):
    def __init__(self, num_classes):
        super(HybridPlantNet, self).__init__()
<<<<<<< HEAD
        
        # --- UPGRADE 1: EfficientNetV2-M for CNN Backbone ---
        # Output feature size: 1280 (was 512 for ResNet34)
        self.cnn = timm.create_model('efficientnetv2_m', pretrained=True, num_classes=0) 
        self.attn = SpectralAttention(1280) # Update channels to match EfficienNetV2-M output

        # --- UPGRADE 2: Use ViT-Base (Keep this for size balance, or use a larger one) ---
        # Output feature size: 768 (unchanged)
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
        
        # --- Fusion Layer Update ---
        # Input size is now 1280 (from CNN) + 768 (from ViT) = 2048
        self.fc = nn.Sequential(
            nn.Linear(1280 + 768, 512), # Increased intermediate layer to 512
            nn.ReLU(),
            nn.Dropout(0.4), # Increased dropout slightly
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # CNN Branch
        f_cnn = self.cnn.forward_features(x) # Output shape: (B, 1280, H, W)
        f_cnn = self.attn(f_cnn).mean([2,3]) 
        
        # ViT Branch
        f_vit = self.vit(x) # Output shape: (B, 768)
        
        # Feature Fusion
=======
        self.cnn = timm.create_model('resnet34', pretrained=True, num_classes=0) 
        self.attn = SpectralAttention(512)
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
        
        self.fc = nn.Sequential(
            nn.Linear(512 + 768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # CNN features + Attention
        f_cnn = self.cnn.forward_features(x)
        f_cnn = self.attn(f_cnn).mean([2,3]) 
        
        # ViT features (CLS token)
        f_vit = self.vit(x)
        
        # Fusion
>>>>>>> 049551bb1212d2b363f8ae263f6df3e07cef2aeb
        f = torch.cat([f_cnn, f_vit], dim=1)
        
        return self.fc(f)