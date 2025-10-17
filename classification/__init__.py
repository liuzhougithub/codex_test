"""Utilities for training interchangeable classification models."""

from .config import TrainingConfig, OptimConfig, ObjectConfig
from .data import ClassificationDataModule, ToyClassificationDataModule
from .models import MLPClassifier
from .trainer import Trainer

__all__ = [
    "TrainingConfig",
    "OptimConfig",
    "ObjectConfig",
    "ClassificationDataModule",
    "ToyClassificationDataModule",
    "MLPClassifier",
    "Trainer",
]
