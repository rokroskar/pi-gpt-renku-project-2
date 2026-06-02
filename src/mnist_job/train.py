from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from time import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from .data import find_mnist_root, load_dataset
from .model import SmallCNN


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total_loss += loss_fn(logits, y).item()
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.numel()
    return total_loss / total, correct / total


def train(args: argparse.Namespace) -> dict:
    seed_everything(args.seed)
    data_root = find_mnist_root(args.data_dir)
    output_dir = Path(args.output_dir)
    model_dir = Path(args.model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}", flush=True)
    print(f"Using MNIST data from: {data_root}", flush=True)

    full_train = load_dataset(data_root, "train")
    test_ds = load_dataset(data_root, "test")
    train_len = int(0.9 * len(full_train))
    val_len = len(full_train) - train_len
    train_ds, val_ds = random_split(
        full_train,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(args.seed),
    )

    workers = min(args.num_workers, os.cpu_count() or 1)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, num_workers=workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, num_workers=workers)

    model = SmallCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.max_epochs,
        steps_per_epoch=len(train_loader),
    )
    loss_fn = nn.CrossEntropyLoss()

    best = {"test_accuracy": 0.0, "epoch": 0}
    history = []
    start = time()
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item() * y.size(0)
            seen += y.size(0)

        train_loss = running_loss / seen
        val_loss, val_acc = evaluate(model, val_loader, device)
        test_loss, test_acc = evaluate(model, test_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "test_loss": test_loss,
            "test_accuracy": test_acc,
        }
        history.append(row)
        print(json.dumps(row), flush=True)

        if test_acc > best["test_accuracy"]:
            best = row | {"elapsed_seconds": time() - start}
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "metrics": best,
                    "data_root": str(data_root),
                    "normalization": {"mean": 0.1307, "std": 0.3081},
                },
                model_dir / "mnist_cnn.pt",
            )

        if test_acc >= args.target_accuracy:
            print(
                f"Target accuracy {args.target_accuracy:.4f} reached at epoch {epoch}; terminating early.",
                flush=True,
            )
            break

    metrics = {
        "target_accuracy": args.target_accuracy,
        "best": best,
        "history": history,
        "model_path": str(model_dir / "mnist_cnn.pt"),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("FINAL_METRICS " + json.dumps(metrics["best"]), flush=True)
    if best["test_accuracy"] < args.target_accuracy:
        raise SystemExit(f"Target accuracy not reached: best={best['test_accuracy']:.4f}")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=os.environ.get("MNIST_DATA_DIR", "/data/mnist"))
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--target-accuracy", type=float, default=0.99)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
