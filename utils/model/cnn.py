import torch
from torch import nn
from torch.nn import functional as F

class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.InstanceNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.InstanceNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.InstanceNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(64 * 48 * 48, 128),  # 384 → 192 → 96 → 48
            nn.ReLU()
        )

    def forward(self, x):  # x: (1, 1, 384, 384)
        x = self.features(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x  # (1, 128)

class CNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.slice_model = FeatureExtractor()
        self.fc = nn.Linear(128, num_classes)  # feature size는 아래에서 정함

    def normalize(self, x):
        # x: (S, 1, H, W)
        mean = x.mean(dim=(1, 2, 3), keepdim=True)  # 슬라이스별 평균
        std = x.std(dim=(1, 2, 3), keepdim=True) + 1e-6  # 슬라이스별 표준편차, 0 나누기 방지
        return (x - mean) / std

    def forward(self, x):  # x: (S, 1, 384, 384)
        x = self.normalize(x)             # 정규화
        features = self.slice_model(x)    # (S, 128)
        pooled = features.mean(dim=0)     # (128,)
        return self.fc(pooled.unsqueeze(0))  # (1, num_classes)