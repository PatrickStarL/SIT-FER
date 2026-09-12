"""Package-level smoke test: everything importable via the public API"""


def test_public_api_imports():
    from sit_fer.core import Config, Trainer
    from sit_fer.data import get_raf, RandAugment, RAFDataset
    from sit_fer.models import ResNet18, TextEncoder, tokenize
    from sit_fer.losses import SupConLoss, PartialLoss
    from sit_fer.utils import InstanceBank, setup_seed, AverageMeter, accuracy

    assert all([
        Config, Trainer, get_raf, RandAugment, RAFDataset,
        ResNet18, TextEncoder, tokenize, SupConLoss, PartialLoss,
        InstanceBank, setup_seed, AverageMeter, accuracy,
    ])
