<h1 align="center">🧠 SCR: Structured Reasoning for Large Language Models</h1>

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-arXiv-b5212f.svg?logo=arxiv)](https://arxiv.org/pdf/2601.07180)
[![HuggingFace](https://img.shields.io/badge/Data&Model-HuggingFace-ffd21e.svg?logo=huggingface)]()
[![License](https://img.shields.io/badge/License-MIT-blue.svg)]()
[![Python](https://img.shields.io/badge/Python-3.10-yellow.svg)]()

</div>

## 📖 Overview

A single long-CoT trajectory **entangles** generation, verification, and revision into one token stream, so neither token-level SFT nor outcome-level RL can assign ability-specific credit. As a result, large reasoning models (LRMs) often keep verifying and revising even after reaching the correct answer.

**SCR (Structured Reasoning)** reorganizes long-CoT reasoning into stages that are **explicit, evaluable, and independently trainable**, realized through a *Generate–Verify–Revise* paradigm:

- **Dynamic Termination Supervision (DTS).** Stops reasoning once self-verification confirms correctness.
- **Selective Loss Masking (SLM).** Excludes incorrect initial answers from the imitation loss while keeping them as context.
- **Progressive Two-Stage RL.** Stage I jointly optimizes generation and verification; Stage II focuses on revision once the verifier is reliable.

Across three backbones and multiple benchmarks, SCR improves reasoning quality and self-verification while **reducing output length by up to 50%**.

<div align="center">
    <img src="./figs/method.png" alt="SCR method overview" width="600"/>
</div>

## 📁 Repository Structure

```
SCR/
├── LLaMA-Factory/        # SFT codebase (adapted)
├── EasyR1/               # RL codebase (adapted)
├── data/                 # Training and evaluation data
├── infer/                # Inference / evaluation scripts
├── figs/                 # Figures and paper PDF
├── run_SCR-SFT.sh        # Entry script for the SFT phase
├── run_SCR-Stage1.sh     # Entry script for RL Stage I (generation + verification)
├── run_SCR-Stage2.sh     # Entry script for RL Stage II (revision)
├── infer.sh              # Entry script for evaluation
└── rl_config.yaml        # RL hyper-parameters
```

## 🚀 Quick Start

### 1. SFT Phase

#### Install Dependencies

```bash
conda create -n SCR-SFT python=3.10.9
conda activate SCR-SFT
cd LLaMA-Factory
pip install -e .
```

#### Run

```bash
bash run_SCR-SFT.sh
```

Key parameters in `run_SCR-SFT.sh`:

| Parameter    | Description                                                                                                  |
| ------------ | ------------------------------------------------------------------------------------------------------------ |
| `model_path` | Path to the base language model. This is the model fine-tuned during training.                               |
| `template`   | Conversation/prompt template for the model.                                                                  |
| `dataset`    | Path to the training dataset. Please update the `EOS` token according to the model type before running SFT.  |

### 2. RL Phase

#### Install Dependencies

```bash
conda create -n SCR-RL python=3.10.9
conda activate SCR-RL
cd EasyR1
pip install -e .
```

#### Run

The RL phase uses a progressive two-stage curriculum. Stage I jointly trains generation and self-verification; Stage II focuses on revision once the verifier becomes reliable.

```bash
# Stage I: jointly optimize initial generation and self-verification
bash run_SCR-Stage1.sh

# Stage II: optimize revision based on the trained verifier
bash run_SCR-Stage2.sh
```

### 3. Evaluation

```bash
bash infer.sh
```

## 📊 Highlights

- ✅ **Explicit reasoning stages.** Each step (generation / verification / revision) is exposed and can be supervised individually.
- ✅ **Trajectory-aware supervision.** DTS + SLM provide targeted SFT signals that prevent imitation of false starts and over-long verification.
- ✅ **Decoupled credit assignment.** Two-stage GRPO assigns ability-specific credit instead of a single outcome-level reward.
- ✅ **Shorter, sharper reasoning.** Up to **50% reduction in output length** without sacrificing accuracy.

## 🎉 Acknowledgements

This repository includes code adapted from the following open-source projects:

- **LLaMA-Factory**: GitHub - hiyouga/LlamaFactory: Unified Efficient Fine-Tuning of 100+ LLMs & VLMs (ACL 2024)
- **EasyR1**: GitHub - hiyouga/EasyR1: EasyR1: An Efficient, Scalable, Multi-Modality RL Training Framework based 

We thank the authors and contributors of these projects for making their code publicly available.

<!-- ## 📌 Citation

If you find this work useful for your research, please consider citing:

```bibtex
@article{scr2026,
  title  = {Structured Reasoning for Large Language Models},
  author = {Anonymous Author(s)},
  year   = {2026}
}
``` -->