"""Neural-network definitions used in phase 1."""

from __future__ import annotations

import torch
from torch import nn


class SimpleCNN(nn.Module):
    """Small CNN baseline for 28x28 MedMNIST images.

    The architecture deliberately remains monolithic in phase 1.  The
    feature-extractor/head split is introduced by the GPFL work in phase 2.
    """

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))