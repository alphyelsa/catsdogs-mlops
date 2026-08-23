"""
Baseline CNN for binary cat/dog classification.

Kept intentionally simple (no pretrained backbone) so the whole
pipeline — including the Docker image — has no external weight
download at build or serve time, which matters for reproducible
CI/CD. Swapping in a transfer-learning backbone later only requires
changing this file; train.py and app.py are agnostic to the
architecture as long as it outputs 2 logits for a
(batch, 3, IMAGE_SIZE, IMAGE_SIZE) input.
"""
import torch
import torch.nn as nn


class CatsDogsCNN(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 224 -> 112

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 112 -> 56

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 56 -> 28

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),  # -> (256, 1, 1), size-independent
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def build_model(num_classes: int = 2) -> CatsDogsCNN:
    return CatsDogsCNN(num_classes=num_classes)
