"""ResNet-18 backbone for facial expression recognition

Matches the original SIT-FER design: a face-recognition-pretrained backbone
(MS-Celeb-1M) with L2-normalized features feeding the classifier, since the
three-level fusion (text/instance similarity) is computed as raw dot products
between these features and assumes unit norm.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

logger = logging.getLogger(__name__)


def _load_backbone_state_dict(checkpoint_path: str) -> dict:
    """
    Load a ResNet-18 state dict from a checkpoint file.

    Accepts either a raw state dict or one wrapped under a 'state_dict' key
    (the convention used by the original MS-Celeb-1M pretrained checkpoint),
    and strips a 'module.' prefix left over from DataParallel checkpoints.
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
    return {k.replace('module.', '', 1) if k.startswith('module.') else k: v
            for k, v in state_dict.items()}


class ResNet18(nn.Module):
    """ResNet-18 backbone with custom classifier over L2-normalized features"""

    def __init__(
        self,
        num_classes: int = 7,
        feature_dim: int = 512,
        pretrained: bool = True,
        pretrained_path: Optional[str] = None,
        dropout: float = 0.5,
    ):
        """
        Args:
            num_classes: Number of emotion classes
            feature_dim: Dimension of the pooled feature vector (512 for ResNet-18)
            pretrained: Whether to use ImageNet-pretrained weights when
                `pretrained_path` is not given
            pretrained_path: Path to a face-recognition-pretrained checkpoint
                (e.g. ResNet-18 pretrained on MS-Celeb-1M). Takes precedence
                over `pretrained`. Not bundled with this repo — see
                `legacy/README.md` for background.
            dropout: Dropout probability applied to pooled features before
                classification
        """
        super(ResNet18, self).__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim

        base_model = resnet18(pretrained=pretrained and pretrained_path is None)

        if pretrained_path is not None:
            state_dict = _load_backbone_state_dict(pretrained_path)
            try:
                base_model.load_state_dict(state_dict, strict=True)
            except RuntimeError as e:
                logger.warning(
                    f"Strict load of '{pretrained_path}' failed ({e}); "
                    "retrying with strict=False (e.g. mismatched fc layer is expected "
                    "when the checkpoint was trained on a different identity count)."
                )
                base_model.load_state_dict(state_dict, strict=False)

        # Drop avgpool + fc; pooling/classification are handled explicitly below
        # so feature_dim and num_classes stay independently configurable.
        self.base = nn.Sequential(*list(base_model.children())[:-2])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor):
        """
        Forward pass

        Args:
            x: Input images [batch_size, 3, H, W]

        Returns:
            logits: Classification logits [batch_size, num_classes]
            features: L2-normalized feature embeddings [batch_size, feature_dim]
        """
        feature_map = self.base(x)
        features = F.adaptive_avg_pool2d(feature_map, 1).flatten(1)
        features = self.dropout(features)
        features = F.normalize(features, dim=1)
        logits = self.classifier(features)
        return logits, features


# For backward compatibility with original code
def ResNet_18(num_classes: int = 7, pretrained_path: Optional[str] = None):
    """Create ResNet-18 model"""
    return ResNet18(num_classes=num_classes, feature_dim=512, pretrained=True, pretrained_path=pretrained_path)
