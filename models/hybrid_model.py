import torch
import torch.nn as nn
import timm

# Spectral Attention remains the same, as it's a structural component
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
        w = x.mean(dim=[2,3])
        w = self.fc(w).unsqueeze(-1).unsqueeze(-1)
        return x * w

class HybridPlantNet(nn.Module):
    def __init__(self, num_classes):
        super(HybridPlantNet, self).__init__()
        
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
        f = torch.cat([f_cnn, f_vit], dim=1)
        
        return self.fc(f)