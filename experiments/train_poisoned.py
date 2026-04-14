# Trains a backdoor-poisoned LeNet model using a specified attack configuration
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import yaml
import torch
import torch.nn as nn

from src.utils.seed import set_seed
from src.utils.checkpoint import save_model, load_model
from src.data.loader import GTSRBDataModule, GTSRBDataset
from src.data.preprocessing import build_train_transform, build_eval_transform
from src.data.poisoner import PoisonedDataset
from src.models.lenet import LeNet
from src.attacks.badnets import BadNets
from src.attacks.blended import BlendedInjection
from src.attacks.label_consistent import LabelConsistentAttack
from src.evaluation.metrics import compute_ca, compute_asr
from src.evaluation.reporter import ResultsReporter
from torch.utils.data import DataLoader


def build_attack(cfg, device):
    acfg = cfg["attack"]
    if acfg["name"] == "badnets":
        return BadNets(
            trigger_size=acfg["trigger_size"],
            trigger_position=acfg["trigger_position"],
            trigger_color=tuple(acfg["trigger_color"]),
        )
    if acfg["name"] == "blended":
        return BlendedInjection(
            trigger_path=acfg.get("trigger_path"),
            alpha=acfg["alpha"],
            trigger_size=acfg["trigger_size"],
            random_noise=acfg.get("random_noise", True),
        )
    if acfg["name"] == "label_consistent":
        clean_model = LeNet(cfg["model"]["num_classes"], cfg["model"]["in_channels"]).to(device)
        load_model(clean_model, acfg["clean_model_path"], device=device)
        bcfg = acfg["base_trigger"]
        base = BadNets(
            trigger_size=bcfg["trigger_size"],
            trigger_position=bcfg["trigger_position"],
            trigger_color=tuple(bcfg["trigger_color"]),
        )
        return LabelConsistentAttack(
            model=clean_model,
            base_trigger=base,
            target_class=acfg["target_class"],
            epsilon=acfg["epsilon"],
            pgd_steps=acfg["pgd_steps"],
            pgd_alpha=acfg["pgd_alpha"],
            device=device,
        )
    raise ValueError(f"Unknown attack: {acfg['name']}")


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
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

    attack = build_attack(cfg, device)
    acfg = cfg["attack"]

    poisoned_ds = PoisonedDataset(
        base_dataset=dm.train_dataset,
        attack=attack,
        poison_rate=acfg["poison_rate"],
        target_class=acfg["target_class"],
        source_classes=acfg.get("source_classes"),
        relabel=acfg.get("relabel", True),
        seed=cfg["seed"],
    )
    poison_count = len(poisoned_ds.poison_indices)
    print(f"Poisoned {poison_count}/{len(poisoned_ds)} samples ({poison_count/len(poisoned_ds)*100:.1f}%)")

    poisoned_loader = DataLoader(
        poisoned_ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )

    model = LeNet(cfg["model"]["num_classes"], cfg["model"]["in_channels"]).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["training"]["epochs"])
    criterion = nn.CrossEntropyLoss()

    for epoch in range(cfg["training"]["epochs"]):
        train_epoch(model, poisoned_loader, optimizer, criterion, device)
        val_acc = compute_ca(model, dm.val_loader(), device)
        scheduler.step()
        print(f"Epoch {epoch+1:02d} | val_acc {val_acc:.4f}")

    test_ca = compute_ca(model, dm.test_loader(), device)
    test_asr = compute_asr(
        model,
        dm.test_dataset.features,
        dm.test_dataset.labels,
        attack,
        acfg["target_class"],
        eval_tf,
        device,
    )
    print(f"\nTest CA: {test_ca:.4f} | ASR: {test_asr:.4f}")

    ckpt_name = f"{acfg['name']}_poisoned_lenet.pth"
    save_model(model, os.path.join(cfg["checkpoint_dir"], ckpt_name),
               metadata={"test_ca": test_ca, "asr": test_asr, "attack": acfg["name"]})

    reporter = ResultsReporter(cfg["results_dir"], "attack_results")
    reporter.log_run(attack.get_config() | {"poison_rate": acfg["poison_rate"]},
                     {"test_ca": test_ca, "asr": test_asr})
    reporter.print_table()

    study_dir = f"docs/experiments/{acfg['name']}"
    os.makedirs(study_dir, exist_ok=True)
    with open(os.path.join(study_dir, "study-2026-04-13.md"), "w") as f:
        f.write(f"# Poisoned Training Study: {acfg['name']}\n\n")
        f.write("## Process\n")
        f.write(f"Trained LeNet on GTSRB with {acfg['name']} backdoor attack.\n\n")
        f.write("### Attack Configuration\n")
        for k, v in attack.get_config().items():
            f.write(f"- {k}: {v}\n")
        f.write(f"- poison_rate: {acfg['poison_rate']}\n")
        f.write(f"- target_class: {acfg['target_class']}\n")
        f.write(f"- relabel: {acfg.get('relabel', True)}\n\n")
        f.write("### Dataset\n")
        f.write(f"- Total train samples: {len(poisoned_ds)}\n")
        f.write(f"- Poisoned samples: {poison_count} ({poison_count/len(poisoned_ds)*100:.1f}%)\n")
        f.write(f"- Clean samples: {len(poisoned_ds) - poison_count}\n\n")
        f.write("### Training Configuration\n")
        f.write(f"- Device: {device}\n")
        f.write(f"- Epochs: {cfg['training']['epochs']}\n")
        f.write(f"- Optimiser: Adam lr={cfg['training']['lr']} wd={cfg['training']['weight_decay']}\n")
        f.write(f"- Scheduler: CosineAnnealingLR\n\n")
        f.write("## Results\n")
        f.write("| Metric | Value |\n|---|---|\n")
        f.write(f"| Test CA | {test_ca:.4f} |\n")
        f.write(f"| Test ASR | {test_asr:.4f} |\n\n")
        f.write("## Interpretation\n")
        f.write("A high ASR with maintained CA indicates a successful backdoor attack.\n")
        f.write(f"Target ASR threshold: > 0.80 at poison_rate=0.10 (per Gu et al. 2019).\n")


if __name__ == "__main__":
    main()
