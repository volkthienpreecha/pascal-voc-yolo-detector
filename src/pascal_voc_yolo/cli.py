"""Train the YOLOv1-style detector on Pascal VOC 2007."""

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from .data import VOCGridDataset
from .engine import evaluate, resolve_device, train_epoch
from .model import YOLOv1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--data-dir", type=Path, default=Path("data/voc"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/yolo"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.batch_size <= 0 or args.epochs <= 0 or args.learning_rate <= 0:
        raise SystemExit("batch size, epochs, and learning rate must be positive")

    from torch.utils.data import DataLoader, random_split
    from torchvision import models, transforms
    from torchvision.datasets import VOCDetection

    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    raw_dataset = VOCDetection(args.data_dir, year="2007", image_set="trainval", download=True)
    dataset = VOCGridDataset(raw_dataset, transform)
    validation_size = max(1, len(dataset) // 5)
    training_size = len(dataset) - validation_size
    generator = torch.Generator().manual_seed(args.seed)
    training_set, validation_set = random_split(
        dataset, [training_size, validation_size], generator=generator
    )
    train_loader = DataLoader(
        training_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    validation_loader = DataLoader(
        validation_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    backbone_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    backbone_channels = backbone_model.fc.in_features
    backbone = nn.Sequential(*list(backbone_model.children())[:-2])
    model = YOLOv1(backbone, backbone_channels)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=5e-4
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        training = train_epoch(model, train_loader, optimizer, device)
        validation = evaluate(model, validation_loader, device)
        history.append({"epoch": epoch, "train": training, "validation": validation})
        print(json.dumps(history[-1]))
        if validation["total"] < best_loss:
            best_loss = validation["total"]
            torch.save(model.state_dict(), args.output_dir / "best_model.pth")
    (args.output_dir / "metrics.json").write_text(json.dumps(history, indent=2) + "\n")


if __name__ == "__main__":
    main()
