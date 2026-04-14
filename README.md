<p align="center">
  <img src="assets/demo/class14_stop_0.png" alt="HTOTRBAITSR" width="120">
</p>

<h1 align="center">HTOTRBAITSR</h1>
<p align="center"><strong>Hidden Triggers on the Road: Backdoor Attacks in Traffic Sign Recognition</strong></p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=fff" alt="Python"/>
  <img src="https://img.shields.io/badge/PyTorch-2.11-EE4C2C?logo=pytorch&logoColor=fff" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/Dataset-GTSRB-4CAF50" alt="GTSRB"/>
  <img src="https://img.shields.io/badge/Demo-Gradio-FF6F00" alt="Gradio"/>
</p>

---

## Overview

PyTorch implementation of backdoor attacks and defences on the German Traffic Sign Recognition Benchmark (GTSRB, 43 classes, 51,839 images). Investigates how stealthy attacks can compromise a traffic sign classifier while remaining undetectable through standard accuracy evaluation, and how lightweight defences mitigate them.

<table>
<tr>
<td width="33%" valign="top">

**Attacks**
- BadNets (pixel patch trigger)
- Blended Injection (noise overlay)
- Label Consistent (PGD perturbation)

</td>
<td width="33%" valign="top">

**Defences**
- Fine Pruning
- Activation Clustering
- Spectral Signatures
- Neural Cleanse

</td>
<td width="34%" valign="top">

**Metrics**
- Clean Accuracy (CA)
- Attack Success Rate (ASR)
- ASR reduction per defence
- CA cost per defence

</td>
</tr>
</table>

---

## Quick Start (Demo)

```bash
git clone <repo-url>
cd HTOTRBAITSR
pip install -r demo_requirements.txt
python demo.py
```
---

## Key Results

| Attack | CA | ASR | Best Defence | ASR After |
|---|---|---|---|---|
| Clean baseline | 96.70% | 0.00% | n/a | n/a |
| BadNets | 95.84% | 83.09% | Fine Pruning | 74.02% |
| Blended | 95.07% | 99.71% | Fine Pruning | 34.10% |
| Label Consistent | 96.87% | 0.14% | n/a (attack failed) | n/a |

Trained on Google Colab T4 GPU. LeNet CNN, seed 42, 30 epochs, Adam lr=0.001.

---

## Project Structure

```
src/           attacks, defences, data, models, evaluation
experiments/   training and pipeline scripts
docs/          study notes and results per experiment
checkpoints/   model weights (three demo weights committed)
assets/demo/   sample images and bundled validation subset
results/       CSV logs (gitignored)
data/raw/      GTSRB pickled dataset (gitignored)
```

---

## Team

<table>
<tr>
<td align="center"><strong>Jordan</strong></td>
</tr>
</table>

<p align="center">
  <sub>Computing Science Studio 2 &mdash; 2026</sub>
</p>
