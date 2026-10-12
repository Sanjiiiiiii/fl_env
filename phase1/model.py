import torch
import torch.nn as nn

class SimpleCNN(nn.Module):

    def __init__(self, in_channels=3, num_classes=8):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),  nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten()
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 28, 28)
            feat_dim = self.network(dummy).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        out = self.network(x)
        return self.fc(out)