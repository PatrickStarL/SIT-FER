"""Trainer for SIT-FER: three-level (semantic/instance/text) pseudo-label fusion"""

import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .config import Config
from ..models import ResNet18, TextEncoder, tokenize
from ..utils import (
    setup_seed, setup_logger, accuracy, AverageMeter,
    save_checkpoint, InstanceBank
)


class Trainer:
    """Main trainer class for SIT-FER"""

    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device('cuda' if config.get('device.use_cuda') else 'cpu')

        # Setup logger
        exp_dir = Path(config.get('experiment.output_dir')) / time.strftime('%Y%m%d_%H%M%S')
        exp_dir.mkdir(parents=True, exist_ok=True)
        self.exp_dir = exp_dir

        self.logger = setup_logger(
            'SIT-FER',
            log_file=str(exp_dir / 'train.log')
        )
        self.writer = SummaryWriter(log_dir=str(exp_dir / 'tensorboard'))

        # Save config
        config.save(str(exp_dir / 'config.yaml'))

        # Setup seed
        setup_seed(config.get('experiment.seed', 42))

        # Build models
        self.build_models()

        # Build instance bank
        self.instance_bank = InstanceBank(
            feature_dim=config.get('instance_bank.dim'),
            bank_size=config.get('instance_bank.size'),
            device=self.device
        )

        # Emotion vocabulary
        self.emotions = config.get('emotions')
        self.vocab = self.emotions

        # Best accuracy tracker
        self.best_acc = 0.0

    def build_models(self):
        """Build vision and text models"""
        # Vision model
        self.model = ResNet18(
            num_classes=self.config.get('model.num_classes'),
            feature_dim=self.config.get('model.feature_dim')
        ).to(self.device)

        # Text model
        self.text_model = TextEncoder(
            vocab_size=self.config.get('text_encoder.vocab_size'),
            d_model=self.config.get('text_encoder.d_model'),
            nhead=self.config.get('text_encoder.nhead'),
            num_layers=self.config.get('text_encoder.num_layers')
        ).to(self.device)

        # Optimizers
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.get('training.lr')
        )
        self.text_optimizer = optim.Adam(
            self.text_model.parameters(),
            lr=self.config.get('training.lr_text')
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss(reduction='none')

        self.logger.info(f'Model parameters: {sum(p.numel() for p in self.model.parameters()) / 1e6:.2f}M')

    def compute_text_features(self):
        """Compute text features for all emotions"""
        text_features_list = []

        with torch.no_grad():
            for emotion in self.emotions:
                text = f"This is a face image of {emotion}"
                text_tokens = tokenize(text, self.vocab).to(self.device)
                text_feature = self.text_model(text_tokens)
                text_feature = F.normalize(text_feature, dim=-1)
                text_features_list.append(text_feature)

        text_features_matrix = torch.cat(text_features_list, dim=0).detach()
        return text_features_matrix

    def train_epoch(self, labeled_loader: DataLoader, unlabeled_loader: DataLoader, epoch: int):
        """Train for one epoch"""
        self.model.train()
        self.text_model.eval()

        losses = AverageMeter()
        losses_supervised = AverageMeter()
        losses_text = AverageMeter()
        losses_consistency = AverageMeter()

        labeled_iter = iter(labeled_loader)
        unlabeled_iter = iter(unlabeled_loader)

        text_features_matrix = self.compute_text_features()

        train_iteration = self.config.get('training.train_iteration')
        pbar = tqdm(range(train_iteration), desc=f'Epoch {epoch}')

        for batch_idx in pbar:
            # Get labeled data
            try:
                inputs_x, targets_x, index_x, _ = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(labeled_loader)
                inputs_x, targets_x, index_x, _ = next(labeled_iter)

            # Get unlabeled data
            try:
                (inputs_u, inputs_u2, inputs_strong), _, index_u, _ = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                (inputs_u, inputs_u2, inputs_strong), _, index_u, _ = next(unlabeled_iter)

            inputs_x = inputs_x.to(self.device)
            targets_x = targets_x.to(self.device)
            inputs_u = inputs_u.to(self.device)
            inputs_strong = inputs_strong.to(self.device)

            batch_size = inputs_x.size(0)

            # Forward pass - labeled data
            output_x, feature_x = self.model(inputs_x)
            self.instance_bank.update(feature_x, targets_x, index_x)

            # Supervised loss
            L_supervised = self.criterion(output_x, targets_x.long()).mean()

            # Text-supervised loss
            similarity_x = torch.mm(feature_x, text_features_matrix.t())
            L_text = self.criterion(similarity_x, targets_x).mean()

            # Unlabeled data
            outputs_u, feature_u = self.model(inputs_u)
            output_strong, _ = self.model(inputs_strong)

            # Compute pseudo-labels with three-level fusion
            with torch.no_grad():
                # Semantic-level
                p_semantic = outputs_u

                # Text-level
                similarity_u = torch.mm(feature_u, text_features_matrix.t())
                p_text = similarity_u

                # Instance-level (after epoch threshold)
                epoch_threshold = self.config.get('training.pseudo_label.epoch_threshold')
                if epoch > epoch_threshold:
                    p_instance = self.instance_bank.compute_instance_similarity(
                        feature_u,
                        num_classes=self.config.get('model.num_classes')
                    )
                    weights = self.config.get('training.pseudo_label.fusion_weights')
                    p_fused = (weights['semantic'] * p_semantic +
                              weights['text'] * p_text +
                              weights['instance'] * p_instance)
                else:
                    weights = self.config.get('training.pseudo_label.fusion_weights_early')
                    p_fused = (weights['semantic'] * p_semantic +
                              weights['text'] * p_text)

                p_fused = F.softmax(p_fused, dim=1)
                max_probs, max_idx = torch.max(p_fused, dim=1)

                # Thresholding
                threshold = self.config.get('training.pseudo_label.threshold')
                mask = max_probs > threshold

            # Consistency loss
            if mask.sum() > 0:
                selected_outputs = output_strong[mask]
                selected_targets = max_idx[mask]
                L_consistency = self.criterion(selected_outputs, selected_targets).mean()
            else:
                L_consistency = torch.tensor(0.0, device=self.device)

            # Total loss
            loss_weights = self.config.get('training.loss_weights')
            loss = (loss_weights['supervised'] * L_supervised +
                   loss_weights['text'] * L_text +
                   loss_weights['consistency'] * L_consistency)

            # Backward
            self.optimizer.zero_grad()
            self.text_optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.text_optimizer.step()

            # Update meters
            losses.update(loss.item(), batch_size)
            losses_supervised.update(L_supervised.item(), batch_size)
            losses_text.update(L_text.item(), batch_size)
            losses_consistency.update(L_consistency.item(), batch_size)

            pbar.set_postfix({
                'loss': f'{losses.avg:.4f}',
                'L_s': f'{losses_supervised.avg:.4f}',
                'L_t': f'{losses_text.avg:.4f}',
                'L_c': f'{losses_consistency.avg:.4f}'
            })

        return losses.avg, text_features_matrix

    @torch.no_grad()
    def validate(self, val_loader: DataLoader, text_features_matrix: torch.Tensor, epoch: int):
        """Validate the model"""
        self.model.eval()
        top1 = AverageMeter()

        pbar = tqdm(val_loader, desc='Validating')
        for inputs, targets, _, _ in pbar:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Forward
            logits, features = self.model(inputs)

            # Three-level fusion
            p_semantic = F.softmax(logits, dim=-1)
            similarity = torch.mm(features, text_features_matrix.t())
            p_text = F.softmax(similarity, dim=-1)
            p_instance = self.instance_bank.compute_instance_similarity(
                features,
                num_classes=self.config.get('model.num_classes')
            )
            p_instance = F.softmax(p_instance, dim=-1)

            weights = self.config.get('training.pseudo_label.fusion_weights')
            p_fused = (weights['semantic'] * p_semantic +
                      weights['text'] * p_text +
                      weights['instance'] * p_instance)

            # Compute accuracy
            prec1, _ = accuracy(p_fused, targets, topk=(1, 5))
            top1.update(prec1.item(), inputs.size(0))

            pbar.set_postfix({'acc': f'{top1.avg:.2f}%'})

        return top1.avg

    def train(self, labeled_loader: DataLoader, unlabeled_loader: DataLoader, val_loader: DataLoader):
        """Main training loop"""
        epochs = self.config.get('training.epochs')
        start_epoch = self.config.get('training.start_epoch')

        for epoch in range(start_epoch, epochs + 1):
            self.logger.info(f'\nEpoch: [{epoch}/{epochs}]')

            # Train
            train_loss, text_features = self.train_epoch(
                labeled_loader, unlabeled_loader, epoch
            )

            # Validate
            val_acc = self.validate(val_loader, text_features, epoch)

            self.logger.info(f'Train Loss: {train_loss:.4f}, Val Acc: {val_acc:.2f}%')

            # TensorBoard logging
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Accuracy/val', val_acc, epoch)

            # Save checkpoint
            is_best = val_acc > self.best_acc
            self.best_acc = max(val_acc, self.best_acc)

            if epoch % self.config.get('experiment.save_freq') == 0 or is_best:
                save_checkpoint({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'text_model_state_dict': self.text_model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'text_optimizer_state_dict': self.text_optimizer.state_dict(),
                    'best_acc': self.best_acc,
                    'config': self.config.to_dict()
                }, save_dir=str(self.exp_dir), filename=f'checkpoint_epoch_{epoch}.pth', is_best=is_best)

        self.logger.info(f'\nBest Accuracy: {self.best_acc:.2f}%')
        self.writer.close()
