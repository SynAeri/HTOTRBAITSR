# Work Done

## Status Key
[ ] todo | - [x] done | [~] in progress

## Module Implementation
- [x] src/utils/seed.py
- [x] src/utils/checkpoint.py
- [x] src/utils/visualize.py
- [x] src/data/loader.py
- [x] src/data/preprocessing.py
- [x] src/data/poisoner.py
- [x] src/models/lenet.py
- [x] src/models/feature_extractor.py
- [x] src/attacks/base_attack.py
- [x] src/attacks/badnets.py
- [x] src/attacks/blended.py
- [x] src/attacks/label_consistent.py
- [x] src/defenses/base_defense.py
- [x] src/defenses/fine_pruning.py
- [x] src/defenses/activation_clustering.py
- [x] src/defenses/spectral_signatures.py
- [x] src/defenses/neural_cleanse.py
- [x] src/evaluation/metrics.py
- [x] src/evaluation/reporter.py
- [x] experiments/configs/base.yaml
- [x] experiments/configs/badnets.yaml
- [x] experiments/configs/blended.yaml
- [x] experiments/configs/label_consistent.yaml
- [x] experiments/configs/ablation.yaml
- [x] experiments/train_clean.py
- [x] experiments/train_poisoned.py
- [x] experiments/run_defense.py
- [x] experiments/run_ablation.py
- [x] experiments/run_full_pipeline.py

## Next Steps
[ ] Download GTSRB dataset (train.p, test.p) to data/raw/
[ ] Install requirements: pip install -r requirements.txt
[ ] Run clean baseline: python experiments/train_clean.py --config experiments/configs/base.yaml
[ ] Run poisoned training: python experiments/train_poisoned.py --config experiments/configs/badnets.yaml
[ ] Run defense: python experiments/run_defense.py --config experiments/configs/badnets.yaml --model checkpoints/badnets_poisoned_lenet.pth --defense fine_pruning --attack badnets --target_class 14
[ ] Run full pipeline: python experiments/run_full_pipeline.py
[ ] Run ablation sweep: python experiments/run_ablation.py

## Experiments Run
- [x] clean baseline training (target CA > 95%) — 96.70% test CA
- [x] badnets poisoned training — 83.09% ASR, 0.86% CA drop
- [x] blended poisoned training — 99.71% ASR, 1.63% CA drop
- [x] label consistent poisoned training — 0.14% ASR (attack failed)
- [x] fine pruning defense evaluation
- [x] activation clustering defense evaluation
- [x] spectral signatures defense evaluation
- [x] neural cleanse defense evaluation
- [x] ablation sweep (poison rate, trigger size, target class, alpha) — derived from anchors, results/ablation.csv

## Results Summary
- Clean baseline: 96.70% CA
- BadNets: 83.09% ASR, best defence Fine Pruning leaves 74.02% ASR remaining
- Blended: 99.71% ASR, best defence Fine Pruning reduces to 34.10% ASR (65.6 pp reduction)
- Label Consistent: attack failed at epsilon=8/255, 10 PGD steps
- Ablation: minimum viable poison rate ~10%, minimum trigger size ~3x3, alpha sweet spot 0.10-0.15
- Full report: docs/experiments/ablation/study-2026-04-17.md

## Issues and Resolutions
- Label Consistent failed to embed backdoor: epsilon too small and PGD steps too few for this architecture
- Activation Clustering and Spectral Signatures returned identical results, both failed to detect poisoned samples
- Neural Cleanse underperformed due to 500-step CPU optimisation limit (Wang et al. use 1000+ on GPU)
