"""Device-aware training and evaluation loops for the detector."""

from collections.abc import Iterable

import torch
from torch import nn

from .loss import yolo_loss


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested not in {"cpu", "cuda", "mps"}:
        raise ValueError("device must be auto, cpu, cuda, or mps")
    return torch.device(requested)


def _run_epoch(
    model: nn.Module,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    boxes_per_cell: int,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.to(device).train(training)
    totals = {name: 0.0 for name in ("total", "classification", "localization", "confidence")}
    examples = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            losses = yolo_loss(model(images), targets, boxes_per_cell=boxes_per_cell)
            if optimizer is not None:
                losses.total.backward()
                optimizer.step()
            batch_size = images.shape[0]
            for name in totals:
                totals[name] += float(getattr(losses, name).item())
            examples += batch_size
    if examples == 0:
        raise ValueError("cannot process an empty data loader")
    return {name: value / examples for name, value in totals.items()}


def train_epoch(
    model: nn.Module,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    boxes_per_cell: int = 2,
) -> dict[str, float]:
    return _run_epoch(model, loader, device, boxes_per_cell, optimizer)


def evaluate(
    model: nn.Module,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    boxes_per_cell: int = 2,
) -> dict[str, float]:
    return _run_epoch(model, loader, device, boxes_per_cell, optimizer=None)
