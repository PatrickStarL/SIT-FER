"""CLI entrypoint: train SIT-FER on RAF-DB

Usage:
    python scripts/train.py --config configs/base.yaml --gpu 0
"""

import argparse
import os
import sys
from pathlib import Path

import torchvision.transforms as transforms
import torch.utils.data as data

# Add src to path so `sit_fer` is importable without installation
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sit_fer.core import Config, Trainer
from sit_fer.data import get_raf


def build_dataloaders(config: Config):
    mean = config.get('dataset.mean')
    std = config.get('dataset.std')
    image_size = config.get('dataset.image_size')

    transform_train = transforms.Compose([
        transforms.Resize([image_size, image_size]),
        transforms.RandomApply([
            transforms.RandomCrop(image_size, padding=8)
        ], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    transform_val = transforms.Compose([
        transforms.Resize([image_size, image_size]),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    train_labeled_set, train_unlabeled_set, test_set = get_raf(
        config.get('dataset.train_root'),
        config.get('dataset.label_train'),
        config.get('dataset.test_root'),
        config.get('dataset.label_test'),
        config.get('ssl.n_labeled'),
        transform_train=transform_train,
        transform_val=transform_val,
        num_classes=config.get('model.num_classes'),
    )

    batch_size = config.get('training.batch_size')
    num_workers = config.get('training.num_workers')

    labeled_loader = data.DataLoader(
        train_labeled_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True
    )
    unlabeled_loader = data.DataLoader(
        train_unlabeled_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True
    )
    test_loader = data.DataLoader(
        test_set, batch_size=64, shuffle=False, num_workers=num_workers
    )

    return labeled_loader, unlabeled_loader, test_loader


def main():
    parser = argparse.ArgumentParser(description='Train SIT-FER')
    parser.add_argument('--config', type=str, default='configs/base.yaml',
                       help='Path to config file')
    parser.add_argument('--gpu', type=str, default='0',
                       help='GPU ID')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    config = Config(config_path=args.config)

    labeled_loader, unlabeled_loader, test_loader = build_dataloaders(config)

    trainer = Trainer(config)
    trainer.train(labeled_loader, unlabeled_loader, test_loader)


if __name__ == '__main__':
    main()
