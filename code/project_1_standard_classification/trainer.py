import sys

import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import CarBrandClsConfig


class CarTrainer:
    def __init__(self, device: str, model: torch.nn.Module, train_loader, optimizer):

        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.lr_max = CarBrandClsConfig.lr_max
        self.total_epoch = CarBrandClsConfig.epoch_total
        self.train_loader = train_loader

        self.bs_print = 2
        self.tqdm_bar = tqdm(total=len(self.train_loader) * self.total_epoch,
                             file=sys.stdout, position=0, ncols=120)

    def train_epoch(self, idx_epoch):
        """
        idx_epoch should start from 1
        """
        lr = self._change_lr(idx_epoch)

        self.model.train()

        for batch_idx, (data, target) in enumerate(self.train_loader, 1):

            self.optimizer.zero_grad()

            logits = self.model(data.to(self.device))
            loss = F.cross_entropy(logits, target.to(self.device))

            loss.backward()
            self.optimizer.step()

            if batch_idx % self.bs_print == 0:
                self.tqdm_bar.write(f'[{idx_epoch:<2}, {batch_idx + 1:<2}] '
                                    f'loss: {loss:<6.4f} '
                                    f'lr: {lr:.4f} ')

            self.tqdm_bar.update(1)
            self.tqdm_bar.set_description(f'epoch-{idx_epoch:<3} '
                                          f'batch-{batch_idx + 1:<3} '
                                          f'loss-{loss:<.2f} '
                                          f'lr-{lr:.3f}')

        if idx_epoch >= self.total_epoch:
            self.tqdm_bar.close()

    def _change_lr(self, idx_epoch):

        if idx_epoch >= 0.75 * self.total_epoch:
            lr = self.lr_max * 0.01
        elif idx_epoch >= 0.5 * self.total_epoch:
            lr = self.lr_max * 0.1
        else:
            lr = self.lr_max

        self.optimizer.param_groups[0].update(lr=lr)

        return lr
