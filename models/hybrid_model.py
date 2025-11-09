import torch
import torch.nn as nn
import timm

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
        w = x.mean(dim=[2,3])  # Global avg pooling (B, C)
        w = self.fc(w).unsqueeze(-1).unsqueeze(-1) # Output (B, C, 1, 1)
        return x * w

class HybridPlantNet(nn.Module):
    def __init__(self, num_classes):
        super(HybridPlantNet, self).__init__()
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
        f = torch.cat([f_cnn, f_vit], dim=1)
        
        return self.fc(f)