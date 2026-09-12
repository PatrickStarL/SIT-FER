import numpy as np
import torch


def generate_candidate_labels(train_labels):

    K = 7
    #n = train_labels.shape[0]

    partialY = torch.zeros(1, 9).squeeze()
    partialY[train_labels] = 1.0
    partialY[7] = train_labels

    return partialY
