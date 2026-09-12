"""Partial Loss for confident learning"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PartialLoss(nn.Module):
    """Partial loss with confidence matrix"""

    def __init__(self, confidence: torch.Tensor, conf_ema_m: float = 0.99):
        super().__init__()
        self.confidence = confidence
        self.init_conf = confidence.detach()
        self.conf_ema_m = conf_ema_m

    def set_conf_ema_m(self, epoch: int, epochs: int, conf_ema_range: tuple = (0.99, 0.999)):
        """Set confidence EMA momentum based on epoch"""
        start, end = conf_ema_range
        self.conf_ema_m = 1.0 * epoch / epochs * (end - start) + start

    def forward(self, outputs: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        """
        Compute partial loss

        Args:
            outputs: Model predictions [batch_size, num_classes]
            index: Sample indices in the dataset

        Returns:
            Average loss scalar
        """
        logsm_outputs = F.log_softmax(outputs, dim=1)
        final_outputs = logsm_outputs * self.confidence[index, :]
        average_loss = -((final_outputs).sum(dim=1)).mean()
        return average_loss

    def confidence_update(
        self,
        temp_un_conf: torch.Tensor,
        batch_index: torch.Tensor,
        batchY: torch.Tensor
    ):
        """Update confidence matrix with EMA"""
        with torch.no_grad():
            _, prot_pred = (temp_un_conf * batchY).max(dim=1)
            pseudo_label = F.one_hot(prot_pred, batchY.shape[1]).float().cuda().detach()
            self.confidence[batch_index, :] = (
                self.conf_ema_m * self.confidence[batch_index, :] +
                (1 - self.conf_ema_m) * pseudo_label
            )
        return None
