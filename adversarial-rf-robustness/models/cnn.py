"""
CNN baseline for I/Q modulation classification.
Architecture inspired by VT-CNN2 (O'Shea et al., 2016) with modern improvements.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RFClassifierCNN(nn.Module):
    """
    CNN classifier for RF modulation recognition from I/Q data.

    Input shape: (batch, 2, T) where T is the number of I/Q samples.
    Output: (batch, num_classes) logits.
    """

    def __init__(self, num_classes: int = 11, input_length: int = 128, dropout: float = 0.5):
        super().__init__()
        self.num_classes = num_classes
        self.input_length = input_length

        # Convolutional feature extractor
        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(2, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            # Block 2
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            # Block 3
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            # Block 4
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, 2, T).
        Returns:
            Logits of shape (batch, num_classes).
        """
        features = self.features(x)
        logits = self.classifier(features)
        return logits

    def get_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
