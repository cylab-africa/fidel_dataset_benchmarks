# Fidel baseline Training and Evaluation of  CRNN

This project provides an end-to-end training pipeline for an Amharic OCR model using a CRNN architecture. It uses Conda for environment management, a `config.yaml` for parameter configuration, and Weights & Biases for experiment tracking.

---

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Project Structure](#project-structure)
3. [Configuration](#configuration)
   - [.env file](#env-file)
   - [config.yaml](#configyaml)
4. [Conda Environment](#conda-environment)
5. [Executing Training](#executing-training)
6. [Scripts](#scripts)
7. [Notes](#notes)

---

## Prerequisites
- **Operating System:** Linux/macOS (Unix-like)
- **Python:** 3.8 or higher
- **Conda:** An existing Conda installation
- **GPU (optional):** CUDA-enabled GPU for faster training

---

## Project Structure
```
CRNN
├── .env
├── config.yaml
├── requirements.txt      # pip dependencies (for venv users)
├── environment.yml       # Conda environment spec (optional)
├── train.py              # Main training script
├── run_training.sh       # Bash script to launch training
├── README.md             # This file
└── src/                  # Source code (datasets, models, utils)
```

---

## Configuration

### .env file
Create a `.env` in the project root (add to `.gitignore`):
```bash
# .env
WANDB_API_KEY=<your wandb API key>
```
The script will load these credentials at runtime to authenticate with Weights & Biases.

### config.yaml
All hyperparameters and file paths live in `config.yaml`. Example:
```yaml
paths:
  train_csv: "../raw_data/.../all_train.csv"
  test_csv:  "../raw_data/.../all_test.csv"
  root_dir:  "../raw_data/Amharic_Data/train"
  test_root: "../raw_data/Amharic_Data/test"

dataset:
  img_height:   100
  img_width:    1000
  batch_size:   64
  dev_size:     0.2
  random_state: 42

testing:
  train_data_type: "typed"
  test_data_type:  "hdd"

model:
  hidden_size: 256
  model_name:  "my_ocr_model"

training:
  learning_rate: 3e-4
  epochs:        30
  save_samples:  true
  sample_count:  10

finetune:
  fine_tune:             false
  fine_tune_model_name:  ""
```
Adjust the paths and parameters to suit your dataset and experimentation needs.

---

## Conda Environment
This project assumes you have a Conda environment with all dependencies installed.

1. **Create environment** (if not already):
   ```bash
   conda create -n ocr_env python=3.9
   ```
2. **Activate environment**:
   ```bash
   conda activate ocr_env
   ```
3. **Install packages**:
   ```bash
   pip install -r requirements.txt
   ```
  
   

---

## Executing Training
Use the `runner.sh` script to launch training:

```bash
./runner.sh <conda_env_name> [config_file]
```

- `<conda_env_name>`: Name of your existing Conda environment (e.g., `ocr_env`).
- `[config_file]`: Path to `config.yaml` (defaults to `config.yaml`).

**Example**:
```bash
chmod +x runner.sh
./runner.sh ocr_env
# or with custom config
./runner.sh ocr_env configs/my_config.yaml
```

This script will:
1. Load environment variables from `.env`.
2. Initialize and activate the specified Conda environment.
3. Run `train.py --config <config_file>` to start training.
4. After training is complete model results on test will be printed to the consolde

---

## Scripts
- **`train.py`**: Main training entry point. Parses `--config`, builds dataset, model, and training loop.
- **`runner.sh`**: Convenience shell wrapper to set up env and kick off `train.py`.

---


# Fidel baseline Training and Evaluation of  TrOCR
change directory to TrOCR and exexcute python scripts there.
