# End-to-end pipeline: train clean baseline, train each poisoned model, apply all defenses, report results
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils.seed import set_seed
from src.utils.checkpoint import save_model, load_model
from src.data.loader import GTSRBDataModule
from src.data.preprocessing import build_train_transform, build_eval_transform
from src.data.poisoner import PoisonedDataset
from src.models.lenet import LeNet
from src.attacks.badnets import BadNets
from src.attacks.blended import BlendedInjection
from src.attacks.label_consistent import LabelConsistentAttack
from src.defenses.fine_pruning import FinePruning
from src.defenses.activation_clustering import ActivationClustering
from src.defenses.spectral_signatures import SpectralSignatures
from src.defenses.neural_cleanse import NeuralCleanse
from src.evaluation.metrics import compute_ca, compute_asr
from src.evaluation.reporter import ResultsReporter


DEFENSE_MAP = {
    "fine_pruning": FinePruning,
    "activation_clustering": ActivationClustering,
    "spectral_signatures": SpectralSignatures,
    "neural_cleanse": NeuralCleanse,
}


def train_model(model, loader, val_loader, cfg, device, label=""):
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"],
                                 weight_decay=cfg["training"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["training"]["epochs"])
    criterion = nn.CrossEntropyLoss()
    best_val = 0.0
    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            criterion(model(images), labels).backward()
            optimizer.step()
        scheduler.step()
        val_acc = compute_ca(model, val_loader, device)
        if val_acc > best_val:
            best_val = val_acc
        print(f"  [{label}] Epoch {epoch+1:02d} | val_acc {val_acc:.4f}")
    return model


def main():
    with open("experiments/configs/base.yaml") as f:
        base_cfg = yaml.safe_load(f)

    set_seed(base_cfg["seed"])
    device = base_cfg["device"] if torch.cuda.is_available() else "cpu"

    train_tf = build_train_transform(
        use_grayscale=base_cfg["data"].get("use_grayscale", False),
        use_hist_eq=base_cfg["data"].get("use_hist_eq", True),
    )
    eval_tf = build_eval_transform(
        use_grayscale=base_cfg["data"].get("use_grayscale", False),
        use_hist_eq=base_cfg["data"].get("use_hist_eq", True),
    )

    dm = GTSRBDataModule(
        train_pickle=base_cfg["data"]["train_pickle"],
        test_pickle=base_cfg["data"]["test_pickle"],
        batch_size=base_cfg["data"]["batch_size"],
        val_split=base_cfg["data"]["val_split"],
        num_workers=base_cfg["data"]["num_workers"],
        train_transform=train_tf,
        eval_transform=eval_tf,
        seed=base_cfg["seed"],
    )
    dm.setup()

    reporter = ResultsReporter(base_cfg["results_dir"], "full_pipeline")
    os.makedirs(base_cfg["checkpoint_dir"], exist_ok=True)

    print("=== Step 1: Clean baseline ===")
    clean_model = LeNet(base_cfg["model"]["num_classes"], base_cfg["model"]["in_channels"]).to(device)
    clean_model = train_model(clean_model, dm.train_loader(), dm.val_loader(), base_cfg, device, "clean")
    clean_ca = compute_ca(clean_model, dm.test_loader(), device)
    print(f"Clean CA: {clean_ca:.4f}")
    save_model(clean_model, os.path.join(base_cfg["checkpoint_dir"], "clean_lenet.pth"))
    reporter.log_run({"stage": "clean", "attack": "none", "defense": "none"},
                     {"test_ca": clean_ca, "asr": 0.0})

    target_class = 14
    attacks = {
        "badnets": BadNets(trigger_size=3, trigger_position="bottom_right"),
        "blended": BlendedInjection(alpha=0.15, random_noise=True),
        "label_consistent": LabelConsistentAttack(
            model=clean_model,
            base_trigger=BadNets(trigger_size=3, trigger_position="bottom_right"),
            target_class=target_class,
            device=device,
        ),
    }

    poisoned_models = {}
    for atk_name, attack in attacks.items():
        print(f"\n=== Step 2: Poisoned training ({atk_name}) ===")
        relabel = atk_name != "label_consistent"
        poisoned_ds = PoisonedDataset(
            base_dataset=dm.train_dataset,
            attack=attack,
            poison_rate=0.10,
            target_class=target_class,
            relabel=relabel,
            seed=base_cfg["seed"],
        )
        poisoned_loader = DataLoader(
            poisoned_ds, batch_size=base_cfg["data"]["batch_size"],
            shuffle=True, num_workers=base_cfg["data"]["num_workers"], pin_memory=True,
        )
        model = LeNet(base_cfg["model"]["num_classes"], base_cfg["model"]["in_channels"]).to(device)
        model = train_model(model, poisoned_loader, dm.val_loader(), base_cfg, device, atk_name)
        ca = compute_ca(model, dm.test_loader(), device)
        asr = compute_asr(model, dm.test_dataset.features, dm.test_dataset.labels,
                          attack, target_class, eval_tf, device)
        print(f"  CA: {ca:.4f} | ASR: {asr:.4f}")
        save_model(model, os.path.join(base_cfg["checkpoint_dir"], f"{atk_name}_lenet.pth"))
        reporter.log_run({"stage": "poisoned", "attack": atk_name, "defense": "none"},
                         {"test_ca": ca, "asr": asr})
        poisoned_models[atk_name] = (model, attack)

    print("\n=== Step 3: Defense evaluation ===")
    defense_names = ["fine_pruning", "activation_clustering", "spectral_signatures", "neural_cleanse"]
    for atk_name, (poisoned_model, attack) in poisoned_models.items():
        for def_name in defense_names:
            print(f"  {atk_name} + {def_name}")
            def_cls = DEFENSE_MAP[def_name]
            def_kwargs = {"device": device}
            if def_name == "neural_cleanse":
                def_kwargs["num_classes"] = base_cfg["model"]["num_classes"]
            defense = def_cls(**def_kwargs)
            defended = defense.apply(poisoned_model, dm.val_loader())
            ca_d = compute_ca(defended, dm.test_loader(), device)
            asr_d = compute_asr(defended, dm.test_dataset.features, dm.test_dataset.labels,
                                attack, target_class, eval_tf, device)
            print(f"    CA: {ca_d:.4f} | ASR: {asr_d:.4f}")
            reporter.log_run({"stage": "defended", "attack": atk_name, "defense": def_name},
                             {"test_ca": ca_d, "asr": asr_d})

    print("\n=== Final Results ===")
    reporter.print_table()
    print(f"\nResults saved to {reporter.csv_path}")

    study_dir = "docs/experiments/full_pipeline"
    os.makedirs(study_dir, exist_ok=True)
    with open(os.path.join(study_dir, "study-2026-04-13.md"), "w") as f:
        f.write("# Full Pipeline Study\n\n")
        f.write("## Process\n")
        f.write("End-to-end run: clean baseline -> 3 poisoned models -> 4 defenses each.\n\n")
        f.write("### Configuration\n")
        f.write(f"- Device: {device}\n")
        f.write(f"- Epochs: {base_cfg['training']['epochs']}\n")
        f.write(f"- Poison rate: 0.10\n")
        f.write(f"- Target class: {target_class} (Stop sign)\n")
        f.write(f"- Seed: {base_cfg['seed']}\n\n")
        f.write("## Results Summary\n\n")
        f.write("### Step 1: Clean Baseline\n")
        f.write(f"| Metric | Value |\n|---|---|\n")
        f.write(f"| Test CA | {clean_ca:.4f} |\n\n")
        f.write("### Step 2: Poisoned Models\n")
        f.write("| Attack | Test CA | ASR |\n|---|---|---|\n")
        for atk_name, (pm, atk) in poisoned_models.items():
            ca = compute_ca(pm, dm.test_loader(), device)
            asr = compute_asr(pm, dm.test_dataset.features, dm.test_dataset.labels,
                              atk, target_class, eval_tf, device)
            f.write(f"| {atk_name} | {ca:.4f} | {asr:.4f} |\n")
        f.write("\n### Step 3: Defense Results\n")
        f.write("Results are logged in results/full_pipeline.csv.\n\n")
        f.write("## Notes\n")
        f.write("- CA drop > 0.05 after defense indicates excessive accuracy loss.\n")
        f.write("- ASR reduction < 0.20 indicates defense was ineffective for that attack.\n")
        f.write(f"- Full CSV: {reporter.csv_path}\n")


if __name__ == "__main__":
    main()
