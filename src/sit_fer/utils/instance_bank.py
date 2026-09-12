"""Instance bank for storing labeled sample features"""

import torch
import torch.nn as nn


class InstanceBank:
    """Memory bank for instance-level features"""

    def __init__(self, feature_dim: int, bank_size: int, device: str = 'cuda'):
        """
        Initialize instance bank

        Args:
            feature_dim: Dimension of feature vectors
            bank_size: Maximum number of samples in bank
            device: Device to store bank on
        """
        self.feature_dim = feature_dim
        self.bank_size = bank_size
        self.device = device

        # Initialize bank and labels
        self.bank = torch.zeros((feature_dim, bank_size), device=device).detach()
        self.labels = torch.zeros(bank_size, dtype=torch.int, device=device).detach()

    @torch.no_grad()
    def update(self, features: torch.Tensor, labels: torch.Tensor, indices: torch.Tensor):
        """
        Update bank with new features

        Args:
            features: Feature vectors [batch_size, feature_dim]
            labels: Corresponding labels [batch_size]
            indices: Indices in the dataset [batch_size]
        """
        self.bank[:, indices] = features.t()
        self.labels[indices] = labels.int()

    def get_bank(self) -> torch.Tensor:
        """Get feature bank"""
        return self.bank

    def get_labels(self) -> torch.Tensor:
        """Get label bank"""
        return self.labels

    def compute_instance_similarity(
        self,
        query_features: torch.Tensor,
        num_classes: int = 7
    ) -> torch.Tensor:
        """
        Compute instance-level similarity scores

        Args:
            query_features: Query feature vectors [batch_size, feature_dim]
            num_classes: Number of emotion classes

        Returns:
            Class-wise max similarities [batch_size, num_classes]
        """
        batch_size = query_features.size(0)

        # Compute similarity with all instances
        logits = torch.mm(query_features, self.bank)  # [batch_size, bank_size]

        # Aggregate by class (max pooling)
        similarities = torch.zeros((batch_size, num_classes), device=self.device)
        for i in range(batch_size):
            for j, label in enumerate(self.labels):
                similarities[i, label] = torch.max(
                    similarities[i, label],
                    logits[i, j]
                )

        return similarities

    def to(self, device: str):
        """Move bank to device"""
        self.device = device
        self.bank = self.bank.to(device)
        self.labels = self.labels.to(device)
        return self
