"""Configuration dataclasses used by the generic classification trainer."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class ObjectConfig:
    """Configuration for lazily importing and instantiating an object."""

    target: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimConfig(ObjectConfig):
    """Configuration for optimizers."""


@dataclass
class TrainingConfig:
    """Top-level training configuration."""

    seed: int = 42
    max_epochs: int = 20
    device: str = "cpu"
    log_every_n_steps: int = 10
    output_dir: Path = Path("outputs")
    dataset: ObjectConfig = field(default_factory=lambda: ObjectConfig(
        target="classification.data.ToyClassificationDataModule",
        params={},
    ))
    model: ObjectConfig = field(default_factory=lambda: ObjectConfig(
        target="classification.models.MLPClassifier",
        params={},
    ))
    optimizer: OptimConfig = field(default_factory=lambda: OptimConfig(
        target="torch.optim.Adam",
        params={"lr": 1e-3},
    ))
    scheduler: Optional[ObjectConfig] = None


def load_config(path: Path) -> TrainingConfig:
    """Load a :class:`TrainingConfig` from a JSON file."""

    import json

    defaults = TrainingConfig()

    with path.open("r", encoding="utf-8") as fp:
        payload: Dict[str, Any] = json.load(fp)

    def parse_object(data: Optional[Dict[str, Any]], default: ObjectConfig) -> ObjectConfig:
        if data is None:
            return default
        return ObjectConfig(target=data["target"], params=data.get("params", {}))

    dataset_cfg = parse_object(payload.get("dataset"), defaults.dataset)
    model_cfg = parse_object(payload.get("model"), defaults.model)

    optim_cfg = payload.get("optimizer")
    if optim_cfg is None:
        optimizer = defaults.optimizer
    else:
        optimizer = OptimConfig(target=optim_cfg["target"], params=optim_cfg.get("params", {}))

    scheduler_cfg = payload.get("scheduler")
    scheduler = None
    if scheduler_cfg is not None:
        scheduler = ObjectConfig(target=scheduler_cfg["target"], params=scheduler_cfg.get("params", {}))

    return TrainingConfig(
        seed=payload.get("seed", defaults.seed),
        max_epochs=payload.get("max_epochs", defaults.max_epochs),
        device=payload.get("device", defaults.device),
        log_every_n_steps=payload.get("log_every_n_steps", defaults.log_every_n_steps),
        output_dir=Path(payload.get("output_dir", str(defaults.output_dir))),
        dataset=dataset_cfg,
        model=model_cfg,
        optimizer=optimizer,
        scheduler=scheduler,
    )
