# Sweeps attack and defense parameters for ablation study, writing all results to CSV
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils.seed import set_seed
from src.utils.checkpoint import save_model
from src.data.loader import GTSRBDataModule
from src.data.preprocessing import build_train_transform, build_eval_transform
from src.data.poisoner import PoisonedDataset
from src.models.lenet import LeNet
from src.attacks.badnets import BadNets
from src.attacks.blended import BlendedInjection
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


def train_model(model, loader, val_loader, cfg, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"],
                                 weight_decay=cfg["training"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["training"]["epochs"])
    criterion = nn.CrossEntropyLoss()
    for _ in range(cfg["training"]["epochs"]):
        model.train()
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            criterion(model(images), labels).backward()
            optimizer.step()
        scheduler.step()
    return model


def run_single(base_cfg, attack, target_class, poison_rate, dm, eval_tf, device, defenses):
    poisoned_ds = PoisonedDataset(
        base_dataset=dm.train_dataset,
        attack=attack,
        poison_rate=poison_rate,
        target_class=target_class,
        seed=base_cfg["seed"],
    )
    poisoned_loader = DataLoader(
        poisoned_ds,
        batch_size=base_cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=base_cfg["data"]["num_workers"],
        pin_memory=True,
    )
    model = LeNet(base_cfg["model"]["num_classes"], base_cfg["model"]["in_channels"]).to(device)
    model = train_model(model, poisoned_loader, dm.val_loader(), base_cfg, device)

    ca = compute_ca(model, dm.test_loader(), device)
    asr = compute_asr(model, dm.test_dataset.features, dm.test_dataset.labels,
                      attack, target_class, eval_tf, device)
    results = {"ca": ca, "asr": asr}

    for def_name in defenses:
        def_cls = DEFENSE_MAP[def_name]
        def_kwargs = {"device": device}
        if def_name == "neural_cleanse":
            def_kwargs["num_classes"] = base_cfg["model"]["num_classes"]
        defense = def_cls(**def_kwargs)
        defended = defense.apply(model, dm.val_loader())
        ca_d = compute_ca(defended, dm.test_loader(), device)
        asr_d = compute_asr(defended, dm.test_dataset.features, dm.test_dataset.labels,
                            attack, target_class, eval_tf, device)
        results[f"ca_after_{def_name}"] = ca_d
        results[f"asr_after_{def_name}"] = asr_d

    return results


def main():
    with open("experiments/configs/ablation.yaml") as f:
        abl_cfg = yaml.safe_load(f)
    with open(abl_cfg["base_config"]) as f:
        base_cfg = yaml.safe_load(f)

    set_seed(abl_cfg["seed"])
    device = abl_cfg["device"] if torch.cuda.is_available() else "cpu"

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
        seed=abl_cfg["seed"],
    )
    dm.setup()

    reporter = ResultsReporter(abl_cfg["results_dir"], "ablation")
    defenses = abl_cfg["defenses"]
    sweep = abl_cfg["sweep"]

    for poison_rate in sweep["poison_rate"]["values"]:
        fixed = sweep["poison_rate"]["fixed"]
        attack = BadNets(trigger_size=fixed["trigger_size"], trigger_position="bottom_right")
        cfg_row = {"axis": "poison_rate", "value": poison_rate, "attack": "badnets",
                   "trigger_size": fixed["trigger_size"], "target_class": fixed["target_class"]}
        res = run_single(base_cfg, attack, fixed["target_class"], poison_rate, dm, eval_tf, device, defenses)
        reporter.log_run(cfg_row, res)
        print(f"poison_rate={poison_rate}: CA={res['ca']:.4f} ASR={res['asr']:.4f}")

    for trigger_size in sweep["trigger_size"]["values"]:
        fixed = sweep["trigger_size"]["fixed"]
        attack = BadNets(trigger_size=trigger_size, trigger_position="bottom_right")
        cfg_row = {"axis": "trigger_size", "value": trigger_size, "attack": "badnets",
                   "poison_rate": fixed["poison_rate"], "target_class": fixed["target_class"]}
        res = run_single(base_cfg, attack, fixed["target_class"], fixed["poison_rate"], dm, eval_tf, device, defenses)
        reporter.log_run(cfg_row, res)
        print(f"trigger_size={trigger_size}: CA={res['ca']:.4f} ASR={res['asr']:.4f}")

    for target_class in sweep["target_class"]["values"]:
        fixed = sweep["target_class"]["fixed"]
        attack = BadNets(trigger_size=fixed["trigger_size"], trigger_position="bottom_right")
        cfg_row = {"axis": "target_class", "value": target_class, "attack": "badnets",
                   "trigger_size": fixed["trigger_size"], "poison_rate": fixed["poison_rate"]}
        res = run_single(base_cfg, attack, target_class, fixed["poison_rate"], dm, eval_tf, device, defenses)
        reporter.log_run(cfg_row, res)
        print(f"target_class={target_class}: CA={res['ca']:.4f} ASR={res['asr']:.4f}")

    for alpha in sweep["alpha"]["values"]:
        fixed = sweep["alpha"]["fixed"]
        attack = BlendedInjection(alpha=alpha, random_noise=True)
        cfg_row = {"axis": "alpha", "value": alpha, "attack": "blended",
                   "poison_rate": fixed["poison_rate"], "target_class": fixed["target_class"]}
        res = run_single(base_cfg, attack, fixed["target_class"], fixed["poison_rate"], dm, eval_tf, device, defenses)
        reporter.log_run(cfg_row, res)
        print(f"alpha={alpha}: CA={res['ca']:.4f} ASR={res['asr']:.4f}")

    reporter.print_table()
    print(f"\nAblation results saved to {reporter.csv_path}")


if __name__ == "__main__":
    main()
