# Trains a clean baseline LeNet model on GTSRB and saves checkpoint
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import yaml
import torch
import torch.nn as nn
from tqdm import tqdm

from src.utils.seed import set_seed
from src.utils.checkpoint import save_model
from src.utils.visualize import plot_learning_curve
from src.data.loader import GTSRBDataModule
from src.data.preprocessing import build_train_transform, build_eval_transform
from src.models.lenet import LeNet
from src.evaluation.metrics import compute_ca
from src.evaluation.reporter import ResultsReporter


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/configs/base.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    device = cfg["device"] if torch.cuda.is_available() else "cpu"

    train_tf = build_train_transform(
        use_grayscale=cfg["data"].get("use_grayscale", False),
        use_hist_eq=cfg["data"].get("use_hist_eq", True),
    )
    eval_tf = build_eval_transform(
        use_grayscale=cfg["data"].get("use_grayscale", False),
        use_hist_eq=cfg["data"].get("use_hist_eq", True),
    )

    dm = GTSRBDataModule(
        train_pickle=cfg["data"]["train_pickle"],
        test_pickle=cfg["data"]["test_pickle"],
        batch_size=cfg["data"]["batch_size"],
        val_split=cfg["data"]["val_split"],
        num_workers=cfg["data"]["num_workers"],
        train_transform=train_tf,
        eval_transform=eval_tf,
        seed=cfg["seed"],
    )
    dm.setup()

    model = LeNet(
        num_classes=cfg["model"]["num_classes"],
        in_channels=cfg["model"]["in_channels"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["training"]["epochs"]
    )
    criterion = nn.CrossEntropyLoss()

    train_accs = []
    val_accs = []
    best_val = 0.0

    for epoch in range(cfg["training"]["epochs"]):
        train_loss, train_acc = train_epoch(model, dm.train_loader(), optimizer, criterion, device)
        val_acc = compute_ca(model, dm.val_loader(), device)
        scheduler.step()
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        print(f"Epoch {epoch+1:02d} | loss {train_loss:.4f} | train_acc {train_acc:.4f} | val_acc {val_acc:.4f}")
        if val_acc > best_val:
            best_val = val_acc
            save_model(model, os.path.join(cfg["checkpoint_dir"], "clean_lenet.pth"),
                       metadata={"epoch": epoch+1, "val_acc": val_acc})

    test_acc = compute_ca(model, dm.test_loader(), device)
    print(f"\nTest CA: {test_acc:.4f}")

    plot_learning_curve(
        train_accs, val_accs,
        save_path=os.path.join(cfg["results_dir"], "clean_learning_curve.png")
    )

    reporter = ResultsReporter(cfg["results_dir"], "clean_baseline")
    reporter.log_run({"attack": "none"}, {"test_ca": test_acc, "best_val_ca": best_val})
    reporter.print_table()

    study_dir = "docs/experiments/clean_baseline"
    os.makedirs(study_dir, exist_ok=True)
    with open(os.path.join(study_dir, "study-2026-04-13.md"), "w") as f:
        f.write("# Clean Baseline Training Study\n\n")
        f.write("## Process\n")
        f.write("Trained LeNet on clean GTSRB training set (no poisoning).\n")
        f.write(f"- Device: {device}\n")
        f.write(f"- Epochs: {cfg['training']['epochs']}\n")
        f.write(f"- Optimiser: Adam lr={cfg['training']['lr']} wd={cfg['training']['weight_decay']}\n")
        f.write(f"- Scheduler: CosineAnnealingLR\n")
        f.write(f"- Batch size: {cfg['data']['batch_size']}\n")
        f.write(f"- Train samples: {len(dm.train_dataset)}\n")
        f.write(f"- Val samples: {len(dm.val_dataset)}\n")
        f.write(f"- Test samples: {len(dm.test_dataset)}\n\n")
        f.write("## Results\n")
        f.write(f"| Metric | Value |\n|---|---|\n")
        f.write(f"| Best val CA | {best_val:.4f} |\n")
        f.write(f"| Test CA | {test_acc:.4f} |\n\n")
        f.write("## Epoch Log\n")
        f.write("| Epoch | Train Acc | Val Acc |\n|---|---|---|\n")
        for i, (ta, va) in enumerate(zip(train_accs, val_accs)):
            f.write(f"| {i+1} | {ta:.4f} | {va:.4f} |\n")
        f.write("\n## Notes\n")
        f.write("Target CA > 0.95. Checkpoint saved at best val epoch.\n")


if __name__ == "__main__":
    main()
