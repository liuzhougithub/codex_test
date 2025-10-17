"""Generic training loop for classification models."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Dict, Optional, Type

import torch
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from .config import ObjectConfig, TrainingConfig
from .data import to_device


class Trainer:
    """Train and evaluate classification models with pluggable components."""

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.output_dir = config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)

        self.data_module = self._instantiate(config.dataset)
        self.model: nn.Module = self._instantiate(config.model)
        self.model.to(self.device)

        self.optimizer = self._instantiate_optimizer(config.optimizer)
        self.scheduler = self._instantiate_scheduler(config.scheduler)

        self.criterion = nn.CrossEntropyLoss()

    def _instantiate(self, cfg: ObjectConfig) -> Any:
        module_name, class_name = cfg.target.rsplit(".", 1)
        module = import_module(module_name)
        target_cls: Any = getattr(module, class_name)
        return target_cls(**cfg.params)

    def _instantiate_optimizer(self, cfg: ObjectConfig) -> torch.optim.Optimizer:
        module_name, class_name = cfg.target.rsplit(".", 1)
        module = import_module(module_name)
        optim_cls: Type[torch.optim.Optimizer] = getattr(module, class_name)
        return optim_cls(self.model.parameters(), **cfg.params)

    def _instantiate_scheduler(self, cfg: Optional[ObjectConfig]) -> Optional[_LRScheduler]:
        if cfg is None:
            return None
        module_name, class_name = cfg.target.rsplit(".", 1)
        module = import_module(module_name)
        scheduler_cls: Type[_LRScheduler] = getattr(module, class_name)
        return scheduler_cls(self.optimizer, **cfg.params)

    def fit(self) -> None:
        best_val_acc = 0.0
        for epoch in range(1, self.config.max_epochs + 1):
            train_metrics = self._run_epoch(self.data_module.train_dataloader(), train=True)
            val_metrics = self._run_epoch(self.data_module.val_dataloader(), train=False)

            val_acc = val_metrics["accuracy"]
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self._save_checkpoint(epoch)

            print(
                f"Epoch {epoch}/{self.config.max_epochs} "
                f"- train_loss: {train_metrics['loss']:.4f} train_acc: {train_metrics['accuracy']:.4f} "
                f"- val_loss: {val_metrics['loss']:.4f} val_acc: {val_metrics['accuracy']:.4f}"
            )

            if self.scheduler is not None:
                self.scheduler.step()

        test_loader = self.data_module.test_dataloader()
        if test_loader is not None:
            test_metrics = self._run_epoch(test_loader, train=False)
            print(
                f"Test - loss: {test_metrics['loss']:.4f} accuracy: {test_metrics['accuracy']:.4f}"
            )

    def _run_epoch(self, dataloader: DataLoader, *, train: bool) -> Dict[str, float]:
        if train:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for step, batch in enumerate(dataloader, start=1):
            inputs, targets = to_device(batch, self.device)

            with torch.set_grad_enabled(train):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == targets).sum().item()
            total_samples += inputs.size(0)

            if train and step % self.config.log_every_n_steps == 0:
                accuracy = total_correct / max(total_samples, 1)
                print(f"Step {step} - loss: {loss.item():.4f} accuracy: {accuracy:.4f}")

        if total_samples == 0:
            return {"loss": float("nan"), "accuracy": 0.0}

        average_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        return {"loss": average_loss, "accuracy": accuracy}

    def _save_checkpoint(self, epoch: int) -> None:
        checkpoint_path = self.output_dir / "best_model.pt"
        payload = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
        }
        torch.save(payload, checkpoint_path)


def run_training(config: TrainingConfig) -> None:
    """Convenience function for running the full training loop."""

    trainer = Trainer(config)
    trainer.fit()
