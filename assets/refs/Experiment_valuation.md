# Deliverables Rubric Breakdown

Reference image: questions.webp

## 1. Problem Definition

### Clear Problem Formula

> "How vulnerable are traffic sign recognition models to backdoor attacks, and how effective
> are lightweight defence strategies at mitigating them without significantly reducing
> classification accuracy?"

This is quantifiable. Vulnerability is measured by ASR, defence effectiveness by ASR
reduction and CA drop. It names the domain (TSR), the threat (backdoor attacks), and the
constraint (without significantly reducing CA). Every experiment in the project traces back
to this question.

### Dataset Collection

- Dataset: German Traffic Sign Recognition Benchmark (GTSRB)
- Source: publicly available benchmark, originally from benchmark.ini.rub.de, obtained as
  pre processed 32x32 pickled files (train.p, test.p, valid.p)
- Size: 51,839 images across 43 traffic sign classes
- Why appropriate: standard benchmark for TSR research, widely used in backdoor attack
  literature (Gu et al. 2019, Wu et al. 2026 both use GTSRB), real world photographs with
  natural variation in lighting, angle, and distance
- Ethical considerations: fully public dataset, no personal or sensitive information,
  no privacy concerns

### Dataset Analysis

- Split sizes: 34,799 train / 4,410 val / 12,630 test (80/20 stratified split from train.p)
- Class imbalance: most frequent class approximately 2,020 samples, least frequent approximately 189 samples,
  roughly a 10x difference. Relevant to the ablation study (target class axis) since low frequency
  classes may be easier or harder to backdoor
- Image properties: 32x32 RGB uint8, real world photographs with variation in lighting,
  blur, rotation, and partial occlusion
- Preprocessing rationale:
  - Histogram equalisation: normalises contrast across images taken in different lighting
    conditions, directly addresses the main source of difficulty in GTSRB
  - Normalisation to mean 0.5 std 0.5: stabilises gradient flow during training
  - Random rotation and affine augmentation: compensates for class imbalance and improves
    generalisation
- Class distribution plots and sample visualisations: notebooks/dataset_analysis.ipynb

## 3. Experimental Analysis

### Experimental Environment

- Hardware: CPU only machine, no GPU. NixOS Linux, Python 3.13 inside a venv (shell.nix
  provides required system libs)
- Software stack: PyTorch 2.11.0, torchvision 0.26.0, numpy 2.4.4, scikit-learn 1.8.0,
  opencv python headless 4.13.0
- Reproducibility: fixed global seed (42) applied to torch, numpy, and python random via
  src/utils/seed.py. Two runs with seed=42 produce identical results CSVs
- Dataset: GTSRB pickled 32x32 RGB images, 34,799 train, 4,410 val, 12,630 test
- Training config: Adam lr=0.001 wd=1e4, CosineAnnealingLR, 30 epochs, batch size 64
- All results written automatically to results/ as CSV and docs/experiments/ as markdown

### Evaluation Criteria

**Clean Accuracy (CA)**
- Standard top 1 classification accuracy on the clean GTSRB test set
- Baseline target: greater than 95% (reference repo approximately 97%, our clean baseline currently hitting approximately 99%)
- Measured before and after each defence to quantify accuracy loss from the defence

**Attack Success Rate (ASR)**
- Fraction of triggered non target test samples the model classifies as the target class
- Measured by applying each attack trigger to every non target test sample and counting
  target class predictions
- Target: greater than 80% for BadNets at 10% poison rate (established by Gu et al. 2019)
- A successful attack has high ASR AND high CA, the model appears normal but is compromised

**Defence Effectiveness**
- CA drop = CA before minus CA after: accuracy cost of applying the defence (want less than 5%)
- ASR reduction = ASR before minus ASR after: how much the defence suppresses the attack
  (want greater than 50% for a meaningful defence)
- The two together reveal the trade off: an aggressive defence may slash ASR but also
  damage CA, making the model impractical

**Detection Rate** (sample level detectors only)
- Precision, recall, F1 comparing flagged samples against true poison indices
- Applies to Activation Clustering and Spectral Signatures

### Ablation Study of the Solution(s)

One parameter varied at a time, all others fixed, using run_ablation.py.

**Poison Rate** how much training data needs to be poisoned for the attack to work
- Values: 1%, 5%, 10%, 20% | Fixed: trigger size 3, target class 14, attack BadNets
- Expected: ASR rises with poison rate; CA degrades slightly at very high rates
- Key question: what is the minimum poison rate for a viable attack?

**Trigger Size** BadNets patch dimensions on a 32x32 image
- Values: 2x2, 3x3, 5x5, 7x7 px | Fixed: poison rate 10%, target class 14
- Expected: larger trigger achieves higher ASR at the same poison rate
- Key question: is there a minimum trigger size that reliably fools the model?

**Target Class** which sign class the attack forces predictions toward
- Values: class 0 (20km/h, low frequency), class 14 (Stop, medium), class 33 (Turn right)
- Fixed: trigger size 3, poison rate 10%
- Expected: class frequency affects backdoor learnability
- Key question: are underrepresented classes easier or harder to target?

**Alpha** (Blended Injection only) trigger visibility vs effectiveness trade off
- Values: 0.05, 0.10, 0.15, 0.20, 0.30 | Fixed: poison rate 10%, target class 14
- Expected: higher alpha gives higher ASR but makes the trigger more detectable
- Key question: at what alpha does the attack become both effective and stealthy?

### Performance Analysis on Benchmark Dataset(s)

All performance measured on the GTSRB test set (12,630 samples). Training on Google Colab T4 GPU,
30 epochs, Adam lr=0.001, seed=42.

**Poisoned model results:**

| Attack | Test CA | ASR | CA Drop vs Clean |
|---|---|---|---|
| None (clean baseline) | 96.70% | 0.00% | 0.00% |
| BadNets (patch trigger) | 95.84% | 83.09% | 0.86% |
| Blended (alpha=0.15) | 95.07% | 99.71% | 1.63% |
| Label Consistent | 96.87% | 0.14% | 0.00% (attack failed) |

BadNets achieved 83.1% ASR at less than 1% CA drop, confirming a successful stealthy attack.
Blended achieved 99.7% ASR with an invisible distributed trigger. Label consistent did not
successfully embed a backdoor under the current configuration (epsilon=8/255, 10 PGD steps).

**Defence results:**

| Attack | Defence | CA After | ASR After | ASR Reduction |
|---|---|---|---|---|
| BadNets | Fine Pruning | 96.23% | 74.02% | 9.07% |
| BadNets | Activation Clustering | 95.81% | 89.07% | +5.98% |
| BadNets | Spectral Signatures | 95.81% | 89.07% | +5.98% |
| BadNets | Neural Cleanse | 95.42% | 89.60% | +6.51% |
| Blended | Fine Pruning | 95.48% | 34.10% | 65.61% |
| Blended | Activation Clustering | 95.62% | 43.91% | 55.80% |
| Blended | Spectral Signatures | 95.62% | 43.91% | 55.80% |
| Blended | Neural Cleanse | 95.38% | 44.66% | 55.05% |

Fine Pruning was the strongest defence overall. Against Blended it reduced ASR from 99.7% to
34.1% with only 0.41% CA drop. Against BadNets it was less effective (74% ASR remaining),
suggesting the patch trigger is more robust to neuron pruning than a distributed trigger.

### Comparison Against Baseline and State-of-the-Art Solution(s)

**Against published baselines:**

| Result | Published | Ours | Match |
|---|---|---|---|
| BadNets ASR at 10% poison rate | greater than 80% (Gu et al. 2019) | 83.1% | Yes |
| Clean LeNet CA on GTSRB | approximately 97% (reference repo) | 96.7% | Yes |
| Blended attack effectiveness | high ASR (Chen et al. 2017) | 99.7% | Yes |
| Fine Pruning defence | moderate ASR reduction (Liu et al. 2018) | 9% vs BadNets, 66% vs Blended | Partial |

Our BadNets implementation reproduces the Gu et al. 2019 result, validating the experimental
setup. The Blended attack exceeds expectations with 99.7% ASR.

**Against state of the art:**
- Fine Pruning (Liu et al. 2018): effective against Blended (65.6% ASR reduction), weak against
  BadNets (9% reduction). Consistent with Liu et al. finding that pruning is most effective
  when the backdoor occupies specific dormant neurons
- Activation Clustering and Spectral Signatures (Chen et al. 2019, Tran et al. 2018): both
  returned identical results, suggesting neither detected the poisoned samples and both returned
  the model unchanged. This is consistent with their known weakness against strong triggers
  where poisoned and clean features are not well separated
- Neural Cleanse (Wang et al. 2019): underperformed due to the 500 step optimisation limit.
  Longer optimisation would improve trigger reverse engineering quality
- FlipBAT (Wu et al. 2026) and Nightfall Deception (Wu et al. 2025) represent more advanced
  TSR specific attacks. Our label consistent attack is in the same family but failed under
  current hyperparameters, highlighting that more sophisticated stealthy attacks require
  careful configuration beyond standard PGD perturbation
