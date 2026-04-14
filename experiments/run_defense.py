# Applies a selected defense to a poisoned model and evaluates CA and ASR before and after
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import yaml
import torch

from src.utils.seed import set_seed
from src.utils.checkpoint import load_model, save_model
from src.data.loader import GTSRBDataModule
from src.data.preprocessing import build_eval_transform
from src.models.lenet import LeNet
from src.attacks.badnets import BadNets
from src.attacks.blended import BlendedInjection
from src.defenses.fine_pruning import FinePruning
from src.defenses.activation_clustering import ActivationClustering
from src.defenses.spectral_signatures import SpectralSignatures
from src.defenses.neural_cleanse import NeuralCleanse
from src.evaluation.metrics import compute_ca, compute_asr
from src.evaluation.reporter import ResultsReporter


DEFENSES = {
    "fine_pruning": FinePruning,
    "activation_clustering": ActivationClustering,
    "spectral_signatures": SpectralSignatures,
    "neural_cleanse": NeuralCleanse,
}


def build_attack_from_cfg(acfg):
    if acfg["name"] == "badnets":
        return BadNets(
            trigger_size=acfg.get("trigger_size", 3),
            trigger_position=acfg.get("trigger_position", "bottom_right"),
            trigger_color=tuple(acfg.get("trigger_color", [255, 255, 255])),
        )
    if acfg["name"] == "blended":
        return BlendedInjection(
            trigger_path=acfg.get("trigger_path"),
            alpha=acfg.get("alpha", 0.15),
            random_noise=acfg.get("random_noise", True),
        )
    raise ValueError(f"Unknown attack: {acfg['name']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True, help="Path to poisoned model checkpoint")
    parser.add_argument("--defense", required=True, choices=list(DEFENSES.keys()))
    parser.add_argument("--attack", required=True, choices=["badnets", "blended", "label_consistent"])
    parser.add_argument("--target_class", type=int, default=14)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    device = cfg["device"] if torch.cuda.is_available() else "cpu"

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
        eval_transform=eval_tf,
        seed=cfg["seed"],
    )
    dm.setup()

    model = LeNet(cfg["model"]["num_classes"], cfg["model"]["in_channels"]).to(device)
    load_model(model, args.model, device=device)

    attack_cfg = cfg.get("attack", {"name": args.attack})
    attack = build_attack_from_cfg(attack_cfg)

    ca_before = compute_ca(model, dm.test_loader(), device)
    asr_before = compute_asr(
        model, dm.test_dataset.features, dm.test_dataset.labels,
        attack, args.target_class, eval_tf, device
    )
    print(f"Before defense | CA: {ca_before:.4f} | ASR: {asr_before:.4f}")

    defense_cls = DEFENSES[args.defense]
    defense_kwargs = {"device": device}
    if args.defense == "fine_pruning":
        defense_kwargs["prune_rate"] = 0.3
    elif args.defense == "neural_cleanse":
        defense_kwargs["num_classes"] = cfg["model"]["num_classes"]
    defense = defense_cls(**defense_kwargs)

    defended_model = defense.apply(model, dm.val_loader())

    ca_after = compute_ca(defended_model, dm.test_loader(), device)
    asr_after = compute_asr(
        defended_model, dm.test_dataset.features, dm.test_dataset.labels,
        attack, args.target_class, eval_tf, device
    )
    print(f"After defense  | CA: {ca_after:.4f} | ASR: {asr_after:.4f}")
    print(f"CA drop: {ca_before - ca_after:.4f} | ASR reduction: {asr_before - asr_after:.4f}")

    ckpt_name = f"{attack_cfg['name']}_{args.defense}_defended.pth"
    save_model(defended_model, os.path.join(cfg["checkpoint_dir"], ckpt_name))

    reporter = ResultsReporter(cfg["results_dir"], "defense_results")
    reporter.log_run(
        {"attack": args.attack, "defense": args.defense, "target_class": args.target_class},
        {
            "ca_before": ca_before, "asr_before": asr_before,
            "ca_after": ca_after, "asr_after": asr_after,
            "ca_drop": ca_before - ca_after, "asr_reduction": asr_before - asr_after,
        }
    )
    reporter.print_table()

    study_dir = f"docs/experiments/{args.defense}"
    os.makedirs(study_dir, exist_ok=True)
    study_path = os.path.join(study_dir, "study-2026-04-13.md")
    mode = "a" if os.path.exists(study_path) else "w"
    with open(study_path, mode) as f:
        if mode == "w":
            f.write(f"# Defense Study: {args.defense}\n\n")
            f.write("## Process\n")
            defense_descriptions = {
                "fine_pruning": "Prune neurons dormant on clean data, then fine-tune for 10 epochs.",
                "activation_clustering": "PCA + k-means on penultimate features per class. Flag minority cluster.",
                "spectral_signatures": "SVD outlier scoring on centred per-class features. Flag top epsilon fraction.",
                "neural_cleanse": "Reverse-engineer per-class trigger via optimisation. MAD outlier detection.",
            }
            f.write(defense_descriptions.get(args.defense, "") + "\n\n")
            f.write("## Results\n")
            f.write("| Attack | CA Before | ASR Before | CA After | ASR After | CA Drop | ASR Reduction |\n")
            f.write("|---|---|---|---|---|---|---|\n")
        f.write(
            f"| {args.attack} | {ca_before:.4f} | {asr_before:.4f} | {ca_after:.4f} "
            f"| {asr_after:.4f} | {ca_before - ca_after:.4f} | {asr_before - asr_after:.4f} |\n"
        )


if __name__ == "__main__":
    main()
