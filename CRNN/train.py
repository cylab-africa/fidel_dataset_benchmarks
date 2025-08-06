
from __future__ import annotations
import pandas as pd
import os
import glob
from dotenv import load_dotenv

import torch
from torch.utils.data import Dataset
from scipy import signal
from scipy.io import wavfile
import cv2
from PIL import Image
import numpy as np

from typing import List, Tuple, Sequence
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from tqdm.auto import tqdm
from collections import Counter
import wandb
import yaml
from dotenv import load_dotenv
from models.crnn import CRNN
import yaml
from my_datasets.crrn_dataset import MyDataset, my_collate_fn, LabelConverter
from utils.evaluation import calculate_metrics , predict
from utils.train_loop import run_epoch

load_dotenv()

with open("../config/crrn_config.yaml") as f:
    cfg = yaml.safe_load(f)


train_csv = pd.read_csv(cfg["paths"]["train_csv"])
test_csv = pd.read_csv(cfg["paths"]["test_csv"])

all_text = ''.join(test_csv['line_text'].tolist() + train_csv['line_text'].tolist())

vocab = set(all_text)


CHARS = ''.join(sorted(vocab))



CHAR2LABEL = {char: i + 1 for i, char in enumerate(CHARS)}
LABEL2CHAR = {label: char for char, label in CHAR2LABEL.items()}



train_df = train_csv[train_csv['type'] == cfg['train_data_type']]

if cfg['test_data_type'] == 'hdd':
    test_18 = test_csv[test_csv['type'] == 'hdd_18']
    test_rand = test_csv[test_csv['type'] == 'hdd_rand']
    test_df = pd.concatenate(test_18, test_rand)
else:
    test_df = test_csv[test_csv['type'] == cfg['test_data_type']]

converter = LabelConverter(MyDataset.CHARS)


train_df, val_df = train_test_split(train_df, test_size=cfg['dataset']['dev_size'], random_state=cfg['dataset']['random_state'], shuffle=True)


img_width = cfg['dataset']['img_width']
img_height = cfg['dataset']['img_height']
data_dir =  cfg['paths']['root_dir']

train_dataset =MyDataset(root_dir=data_dir, mode='train',
                                    df=train_df, img_height=img_height, img_width=img_width)
valid_dataset = MyDataset(root_dir=data_dir, mode='valid',
                                    df=val_df, img_height=img_height, img_width=img_width)
test_dataset = MyDataset(root_dir=cfg['paths']['test_root'], mode='valid',
                                    df=test_df, img_height=img_height, img_width=img_width)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=cfg['dataset']['batch_size'],
    shuffle=True,
    num_workers=4,
    collate_fn=my_collate_fn,
)
valid_loader = torch.utils.data.DataLoader(
    valid_dataset,
    batch_size=cfg['dataset']['batch_size'],
    shuffle=False,
    num_workers=4,
    collate_fn=my_collate_fn,
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=cfg['dataset']['batch_size'],
    shuffle=False,
    num_workers=4,
    collate_fn=my_collate_fn,
)

for images, targets, target_lengths in train_loader:
    
    batch_size = images.size(0)
    
    # Split targets back using target_lengths
    split_targets = torch.split(targets, target_lengths.tolist())

    for i in range(batch_size):
        img = images[i]
        target_tensor = split_targets[i]  # 1D tensor of label indices for this sample
        # plt.imshow(img.squeeze(0).numpy(), cmap='gray')
        print(f"Image {i} shape: {img.shape}")
        print(f"Target indices: {target_tensor.tolist()}")
        
        # Optionally decode target_tensor using your label converter
        decoded_text = converter.decode_indices([target_tensor], raw=False)
        print(f"Decoded text: {decoded_text}")
        
        break  
    break




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CRNN(len(MyDataset.CHARS)+1).to(device)


criterion = nn.CTCLoss(blank=0, zero_infinity=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['training']['learning_rate'])






# set wandb api key in .env file
wandb.login()

if cfg['finetude']['fine_tune']:
    fine_tune_model_name = cfg['finetude']['fine_tune_model_name']
    checkpoint = torch.load(f'{fine_tune_model_name}_logs/best_cer.pt')
    model.load_state_dict(checkpoint['model'])


model_name = cfg['model']['model_name']


wandb.init(project="Amharic OCR", name=model_name)
wandb.watch(model, log="all")

wandb.config.update({
    "learning_rate": cfg['training']['learning_rate'],
    "epochs": cfg['training']['epochs'],
    "batch_size": cfg['dataset']['batch_size'],
    "img_height": img_height,
    "img_width": img_width,
    "train_data_type": cfg['train_data_type'],
    "test_data_type": cfg['test_data_type'],
    "fine_tune": cfg['finetude']['fine_tune'],
})
epochs = cfg['training']['epochs']
LOG_DIR = f"{model_name}_logs"
best_cer = float("inf")
for epoch in range(1, epochs+1):
    print(f"\nEpoch {epoch}/{epochs}")
    train_stats , _ = run_epoch(model, train_loader, criterion, optimizer, device, converter, train=True,
                           epoch_idx=epoch)
    val_stats, best_cer = run_epoch(model, valid_loader, criterion, optimizer, device, converter,
                                        train=False, save_samples=cfg['training']['save_samples'], sample_count=cfg['training']['sample_count'],
                                        log_dir=LOG_DIR, epoch_idx=epoch, best_cer=best_cer)
    print(f"train: {train_stats}\nval  : {val_stats}")
    # Log stats to wandb
    train_stats = {f"train_{k}": v for k, v in train_stats.items()}
    val_stats   = {f"val_{k}": v for k, v in val_stats.items()}
    wandb.log({"epoch": epoch, **train_stats, **val_stats})




test_results = predict(model, test_loader, device, converter)

# Get metrics
wer, cer, f1_word, f1_char = calculate_metrics(test_results)

print(f"WER: {wer}")
print(f"CER: {cer}")
print(f"F1 Score (word-level): {f1_word}")
print(f"F1 Score (character-level): {f1_char}")