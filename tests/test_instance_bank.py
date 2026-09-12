"""Tests for the instance memory bank used in three-level pseudo-label fusion"""

import torch

from sit_fer.utils import InstanceBank


def test_update_and_retrieve():
    bank = InstanceBank(feature_dim=4, bank_size=6, device="cpu")

    features = torch.eye(4)[:3]  # 3 orthonormal-ish vectors
    labels = torch.tensor([0, 1, 2])
    indices = torch.tensor([0, 1, 2])

    bank.update(features, labels, indices)

    assert torch.equal(bank.get_bank()[:, :3], features.t())
    assert torch.equal(bank.get_labels()[:3], labels.int())


def test_compute_instance_similarity_shape_and_argmax():
    bank = InstanceBank(feature_dim=4, bank_size=3, device="cpu")

    features = torch.eye(4)[:3]
    labels = torch.tensor([0, 1, 2])
    indices = torch.tensor([0, 1, 2])
    bank.update(features, labels, indices)

    # Query with the exact same vector as class 1's stored instance
    query = torch.eye(4)[1:2]
    similarity = bank.compute_instance_similarity(query, num_classes=7)

    assert similarity.shape == (1, 7)
    assert torch.argmax(similarity[0]).item() == 1
