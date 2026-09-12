"""Tests for loss functions"""

import torch
import torch.nn.functional as F

from sit_fer.losses import SupConLoss, PartialLoss


def test_supcon_loss_is_finite_with_normalized_features():
    loss_fn = SupConLoss(temperature=0.07)
    features = F.normalize(torch.randn(4, 2, 128), dim=-1)
    labels = torch.tensor([0, 0, 1, 1])

    loss = loss_fn(features, labels=labels)

    assert torch.isfinite(loss)


def test_supcon_loss_requires_3d_features():
    loss_fn = SupConLoss()
    features_2d = torch.randn(4, 128)

    try:
        loss_fn(features_2d)
        assert False, "expected ValueError for < 3D features"
    except ValueError:
        pass


def test_partial_loss_forward_and_confidence_update():
    num_samples, num_classes = 5, 7
    confidence = torch.full((num_samples, num_classes), 1.0 / num_classes)
    loss_fn = PartialLoss(confidence, conf_ema_m=0.9)

    outputs = torch.randn(3, num_classes)
    index = torch.tensor([0, 1, 2])

    loss = loss_fn(outputs, index)
    assert torch.isfinite(loss)

    batchY = F.one_hot(torch.tensor([1, 2, 3]), num_classes).float()
    loss_fn.confidence_update(outputs.softmax(dim=1), index, batchY)
    # Updated rows should still sum to ~1 (mix of prior confidence and one-hot pseudo-label)
    assert torch.allclose(loss_fn.confidence[index].sum(dim=1), torch.ones(3), atol=1e-5)
