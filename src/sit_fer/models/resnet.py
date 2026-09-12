"""ResNet-18 backbone for facial expression recognition"""

import torch
import torch.nn as nn
from torchvision.models import resnet18


class ResNet18(nn.Module):
    """ResNet-18 backbone with custom classifier"""

    def __init__(self, num_classes: int = 7, feature_dim: int = 512, pretrained: bool = True):
        super(ResNet18, self).__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim

        # Load pretrained ResNet-18
        base_model = resnet18(pretrained=pretrained)

        # Remove the final FC layer
        self.features = nn.Sequential(*list(base_model.children())[:-1])

        # Custom classifier
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor):
        """
        Forward pass

        Args:
            x: Input images [batch_size, 3, H, W]

        Returns:
            logits: Classification logits [batch_size, num_classes]
            features: Feature embeddings [batch_size, feature_dim]
        """
        features = self.features(x)
        features = features.view(features.size(0), -1)
        logits = self.classifier(features)
        return logits, features


# For backward compatibility with original code
def ResNet_18(num_classes: int = 7):
    """Create ResNet-18 model"""
    return ResNet18(num_classes=num_classes, feature_dim=512, pretrained=True)
