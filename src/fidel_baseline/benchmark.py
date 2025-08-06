#!/usr/bin/env python
from __future__ import annotations
import argparse
import os
import json
import pandas as pd
import torch
import yaml
import wandb
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from collections import Counter
from tqdm.auto import tqdm

from models.crnn import CRNN
from my_datasets.crrn_dataset import MyDataset, my_collate_fn, LabelConverter
from utils.train_loop import run_epoch
from utils.evaluation import predict, calculate_metrics
import torch.nn as nn

load_dotenv()

def parse_args():
    parser = argparse.ArgumentParser(prog="fidel-bench", description="Run CRNN OCR benchmark")
    parser.add_argument("-c", "--config", required=True, help="Path to YAML config file")
    parser.add_argument("--train-csv", type=str, default=None, help="Override path to train CSV")
    parser.add_argument("--test-csv", type=str, default=None, help="Override path to test CSV")
    parser.add_argument("--dry-run", action="store_true", help="Load setup without training")
    parser.add_argument("--wandb-key", type=str, default=None, help="W&B API key override")
    return parser.parse_args()


def main(
    config_path: str,
    train_csv_path: str | None = None,
    test_csv_path: str | None = None,
    dry_run: bool = False,
    wandb_key: str | None = None
) -> dict:
    # Load configuration
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    print("Loaded config:", json.dumps(cfg, indent=2))
    # Override CSV paths
    if train_csv_path:
        cfg['paths']['train_csv'] = train_csv_path
    if test_csv_path:
        cfg['paths']['test_csv'] = test_csv_path

    # Optionally login to W&B
    if wandb_key:
        wandb.login(key=wandb_key)

    # Load data
    train_csv = pd.read_csv(cfg['paths']['train_csv'])
    test_csv = pd.read_csv(cfg['paths']['test_csv'])
    all_text = ''.join(train_csv['line_text'].tolist() + test_csv['line_text'].tolist())
    CHARS = ''.join(sorted(set(all_text)))

    # Filter by type
    train_df = train_csv[train_csv['type'] == cfg['train_data_type']]
    if cfg['test_data_type'] == 'hdd':
        t18 = test_csv[test_csv['type'] == 'hdd_18']
        trid = test_csv[test_csv['type'] == 'hdd_rand']
        test_df = pd.concat([t18, trid])
    else:
        test_df = test_csv[test_csv['type'] == cfg['test_data_type']]

    # Split train/val
    train_df, val_df = train_test_split(
        train_df,
        test_size=cfg['dataset']['dev_size'],
        random_state=cfg['dataset']['random_state'],
        shuffle=True
    )

    # Create datasets and loaders
    data_dir = cfg['paths']['root_dir']
    train_ds = MyDataset(data_dir, 'train', train_df, cfg['dataset']['img_height'], cfg['dataset']['img_width'], vocab=CHARS)
    val_ds   = MyDataset(data_dir, 'valid', val_df, cfg['dataset']['img_height'], cfg['dataset']['img_width'], vocab=CHARS)
    test_ds  = MyDataset(cfg['paths']['test_root'], 'valid', test_df, cfg['dataset']['img_height'], cfg['dataset']['img_width'], vocab=CHARS)

    train_loader = DataLoader(train_ds, batch_size=cfg['dataset']['batch_size'], shuffle=True,
                              num_workers=cfg.get('num_workers',4), collate_fn=my_collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=cfg['dataset']['batch_size'], shuffle=False,
                              num_workers=cfg.get('num_workers',4), collate_fn=my_collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=cfg['dataset']['batch_size'], shuffle=False,
                              num_workers=cfg.get('num_workers',4), collate_fn=my_collate_fn)

    # Model setup

    device = torch.device(cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    converter = LabelConverter(CHARS)
    model = CRNN(len(CHARS)+1, cfg['model']['hidden_size']).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['training']['learning_rate'])

    if dry_run:
        print("Dry run complete.")
        return {}

    # W&B init
    wandb.init(project=cfg['wandb']['project'], name=cfg['wandb']['name'], config=cfg['training'])
    wandb.watch(model, log="all")

    # Training epochs
    best_cer = float('inf')
    for epoch in range(1, cfg['training']['epochs']+1):
        train_stats, _ = run_epoch(model, train_loader, criterion, optimizer,
                                  device, converter, train=True,
                                  log_dir=cfg['paths']['log_dir'], epoch_idx=epoch)
        val_stats, best_cer = run_epoch(model, val_loader, criterion, optimizer,
                                        device, converter, train=False,
                                        save_samples=cfg['training']['save_samples'],
                                        sample_count=cfg['training']['sample_count'],
                                        log_dir=cfg['paths']['log_dir'],
                                        epoch_idx=epoch, best_cer=best_cer)
        print(f"Epoch {epoch}/{cfg['training']['epochs']}: train={train_stats}, val={val_stats}")
        wandb.log({**{f'train_{k}':v for k,v in train_stats.items()}, **{f'val_{k}':v for k,v in val_stats.items()}, 'epoch':epoch})

    # Final evaluation
    preds_df = predict(model, test_loader, device, converter)
    metrics = calculate_metrics(preds_df)
    wer, cer, f1_word, f1_char = metrics

    wandb.log({
        'test_wer': wer,
        'test_cer': cer,
        'test_f1_word': f1_word,
        'test_f1_char': f1_char
    })
    print(f"Test metrics: WER={wer}, CER={cer}, F1-Word={f1_word}, F1-Char={f1_char}")
    return metrics

if __name__ == "__main__":
    args = parse_args()
    main(args.config, args.train_csv, args.test_csv, args.dry_run, args.wandb_key)
