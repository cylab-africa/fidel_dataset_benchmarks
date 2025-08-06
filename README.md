# Fidel Baseline: CRNN-based Amharic OCR Training & Evaluation

A complete end-to-end pipeline for training and evaluating a CRNN-based Amharic OCR model.
Features:

* 🔧 **Configurable** via YAML
* 🐍 **Installable** as a module (and editable during development)
* 🎛️ **Weights & Biases** integration for experiment tracking
* ☁️ Works with Conda or plain `venv` + `pip`

---

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Configuration](#configuration)

   * [.env](#env-file)
   * [config YAML](#configyaml)
4. [Project Layout](#project-structure)
5. [Usage](#usage)

   * [As a module](#run-as-a-module)
   * [Via console-script (optional)](#console-script)
6. [Conda-only Quick-Start](#conda-quick-start)
7. [Scripts Reference](#scripts)
8. [TrOCR variation](#trocr-variation)
9. [Notes](#notes)

---

## 🛠️ Prerequisites

* **OS:** Linux or macOS (Unix-like)
* **Python:** 3.8+
* **Git** (for cloning)
* **(Optional)** CUDA-enabled GPU for faster training
* **Weights & Biases** account (for experiment logging)

---

## ⚙️ Installation

1. **Clone the repo**

   ```bash
   git https://github.com/cylab-africa/fidel_dataset_benchmarks.git
   cd fidel_dataset_benchmarks
   ```

2. **Install in editable mode**

   ```bash
   # using pip+venv or inside any Python env:
   python3 -m pip install --upgrade pip
   python3 -m pip install -e .
   ```

   This does two things:

   * Puts `fidel_baseline` (and other packages under `src/`) on your PYTHONPATH
   * (If configured) creates a `fidel-benchmark` console script

---

## 🔧 Configuration

### .env file

In the project root, create a `.env` (and add it to `.gitignore`):

```bash
# .env
WANDB_API_KEY=<your-wandb-key>
```

### config YAML

All hyperparams & paths live in a YAML. E.g. `src/configs/crnn_config.yaml`:



Adjust these values and file paths to your setup.

---

## 📁 Project Structure

```
fidel_dataset_benchmarks/
├── .env
├── pyproject.toml            # installable package config
├── requirements.txt          # for non-editable install
├── src/
│   ├── configs/
│   │   ├── crnn_config.yaml
│   │   └── main.yaml
│   ├── fidel_baseline/       # python package
│   │   ├── __init__.py
│   │   └── benchmark.py
│   ├── my_datasets/
│   ├── models/
│   ├── utils/
│   └── ...
├── scripts/
│   └── runner.sh             # Conda wrapper (optional)
└── README.md
```

---

## 🚀 Usage

### Run as a module

```bash
python3 -m fidel_baseline.benchmark \
  -c src/configs/crnn_config.yaml
```



---

## 🐍 Conda-only Quick-Start

If you prefer Conda:

```bash
conda create -n ocr_env python=3.10 -y
conda activate ocr_env
pip install -r requirements.txt
# then run:
```


---

## 📜 Scripts

* **`fidel_baseline/benchmark.py`**

  * Entry point: parses `--config`, builds datasets, model, trains & evaluates.

---


(Or mirror the module approach if you package it similarly.)

---

## 📝 Notes

* Make sure your `PYTHONPATH` includes `src/` if you skip installation.
* You can monitor live metrics in W\&B under your project name.
* Feel free to add more console scripts or sub-commands as you grow the codebase!

---

Happy training! 🎉
