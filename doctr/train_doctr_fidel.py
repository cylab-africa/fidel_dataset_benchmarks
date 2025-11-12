#!/usr/bin/env python3
"""
Adapted DocTR Text Recognition Training Script
Based on official DocTR training code with custom dataset support and comprehensive metrics
"""

import datetime
import os
import time
import json
import shutil
from pathlib import Path
import argparse
import logging

import numpy as np
import pandas as pd
import torch
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torchvision.transforms.v2 import Compose, Normalize
from sklearn.model_selection import train_test_split

# Text evaluation libraries
from jiwer import wer, cer
from sklearn.metrics import f1_score
import editdistance

# DocTR imports
from doctr import transforms as T
from doctr.datasets import RecognitionDataset, VOCABS
from doctr.models import recognition
from doctr.utils.metrics import TextMatch

# W&B logging
import wandb

from tqdm.auto import tqdm


def char_to_binary(text1: str, text2: str):
    """Convert character sequences to binary arrays for F1 calculation"""
    all_chars = set(text1 + text2)
    if not all_chars:
        return np.array([]), np.array([])
    
    char_to_idx = {char: idx for idx, char in enumerate(all_chars)}
    
    vec1 = np.zeros(len(all_chars))
    vec2 = np.zeros(len(all_chars))
    
    for char in text1:
        vec1[char_to_idx[char]] = 1
    for char in text2:
        vec2[char_to_idx[char]] = 1
        
    return vec1, vec2


def word_to_binary(text1: str, text2: str):
    """Convert word sequences to binary arrays for F1 calculation"""
    words1 = text1.split()
    words2 = text2.split()
    
    all_words = set(words1 + words2)
    if not all_words:
        return np.array([]), np.array([])
    
    word_to_idx = {word: idx for idx, word in enumerate(all_words)}
    
    vec1 = np.zeros(len(all_words))
    vec2 = np.zeros(len(all_words))
    
    for word in words1:
        vec1[word_to_idx[word]] = 1
    for word in words2:
        vec2[word_to_idx[word]] = 1
        
    return vec1, vec2


def calculate_char_accuracy(pred_text: str, true_text: str) -> float:
    """Calculate character-level accuracy"""
    if len(true_text) == 0:
        return 1.0 if len(pred_text) == 0 else 0.0
    
    correct_chars = sum(1 for p, t in zip(pred_text, true_text) if p == t)
    max_len = max(len(pred_text), len(true_text))
    return correct_chars / max_len if max_len > 0 else 1.0


def calculate_word_accuracy(pred_text: str, true_text: str) -> float:
    """Calculate word-level accuracy"""
    pred_words = pred_text.split()
    true_words = true_text.split()
    
    if len(true_words) == 0:
        return 1.0 if len(pred_words) == 0 else 0.0
    
    correct_words = sum(1 for p, t in zip(pred_words, true_words) if p == t)
    max_len = max(len(pred_words), len(true_words))
    return correct_words / max_len if max_len > 0 else 1.0


def calculate_comprehensive_metrics(pred_texts, true_texts):
    """Calculate all evaluation metrics"""
    if not pred_texts or not true_texts:
        return {metric: 0.0 for metric in ['cer', 'wer', 'f1_char', 'f1_word', 
                                          'char_accuracy', 'word_accuracy', 'edit_distance']}
    
    # Basic metrics
    cer_score = cer(true_texts, pred_texts)
    wer_score = wer(true_texts, pred_texts)
    
    # Character and word accuracies
    char_accuracies = [calculate_char_accuracy(p, t) for p, t in zip(pred_texts, true_texts)]
    word_accuracies = [calculate_word_accuracy(p, t) for p, t in zip(pred_texts, true_texts)]
    
    char_accuracy = np.mean(char_accuracies)
    word_accuracy = np.mean(word_accuracies)
    
    # F1 scores (micro-averaged)
    all_char_true = []
    all_char_pred = []
    all_word_true = []
    all_word_pred = []
    
    for pred, true in zip(pred_texts, true_texts):
        char_pred, char_true = char_to_binary(pred, true)
        word_pred, word_true = word_to_binary(pred, true)
        
        if len(char_pred) > 0:
            all_char_pred.append(char_pred)
            all_char_true.append(char_true)
        
        if len(word_pred) > 0:
            all_word_pred.append(word_pred)
            all_word_true.append(word_true)
    
    # Calculate F1 scores
    f1_char = 0.0
    f1_word = 0.0
    
    if all_char_true and all_char_pred:
        try:
            char_true_flat = np.concatenate(all_char_true)
            char_pred_flat = np.concatenate(all_char_pred)
            f1_char = f1_score(char_true_flat, char_pred_flat, average='micro', zero_division=0)
        except:
            f1_char = 0.0
    
    if all_word_true and all_word_pred:
        try:
            word_true_flat = np.concatenate(all_word_true)
            word_pred_flat = np.concatenate(all_word_pred)
            f1_word = f1_score(word_true_flat, word_pred_flat, average='micro', zero_division=0)
        except:
            f1_word = 0.0
    
    # Edit distances
    edit_distances = [editdistance.eval(t, p) for t, p in zip(true_texts, pred_texts)]
    avg_edit_distance = np.mean(edit_distances)
    
    return {
        'cer': cer_score,
        'wer': wer_score,
        'f1_char': f1_char,
        'f1_word': f1_word,
        'char_accuracy': char_accuracy,
        'word_accuracy': word_accuracy,
        'edit_distance': avg_edit_distance
    }


def create_labels_json_from_csv(csv_path: str, output_dir: Path, image_root: str, labels_filename: str = "labels.json"):
    """Convert CSV format to DocTR's labels.json format"""
    # df = pd.read_csv(csv_path)

    df = csv_path
    
    # Validate required columns
    required_cols = ['image_filename', 'line_text']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in CSV")
    
    # Create labels dictionary using just the filename (not the full path)
    labels = {}
    skipped_files = 0
    total_files = 0
    
    for _, row in df.iterrows():
        total_files += 1
        image_filename = Path(row['image_filename']).name  # Extract just the filename
        text = str(row['line_text']).strip()
        labels[image_filename] = text
        
    
    # Save labels.json
    labels_path = output_dir / labels_filename
    with open(labels_path, 'w', encoding='utf-8') as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)
    
    # print(f"Created {labels_filename}: {len(labels)} valid images, {skipped_files} skipped out of {total_files} total")
    print(f"Created {labels_filename}: {len(labels)} images")
    return labels_path


def setup_experiment_folder(config: dict, config_path: str) -> Path:
    """Setup experiment folder and copy config"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{config['experiment']['name']}_{timestamp}"
    
    experiment_dir = Path(config['experiment']['output_dir']) / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy config file to experiment directory
    config_copy_path = experiment_dir / "config.yaml"
    shutil.copy2(config_path, config_copy_path)
    
    return experiment_dir


def setup_logging(experiment_dir: Path) -> logging.Logger:
    """Setup logging configuration"""
    log_file = experiment_dir / "training.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def fit_one_epoch(model, device, train_loader, batch_transforms, optimizer, scheduler, amp=False, logger=None):
    """Train one epoch"""
    if amp:
        scaler = torch.cuda.amp.GradScaler()

    model.train()
    epoch_train_loss, batch_cnt = 0, 0
    
    pbar = tqdm(train_loader, desc="Training", dynamic_ncols=True)
    for batch_idx, (images, targets) in enumerate(pbar):
        if torch.cuda.is_available():
            images = images.to(device)
        images = batch_transforms(images)

        optimizer.zero_grad()
        

        try:
            # Pre-model debugging
            if batch_idx < 3:
                logger.info(f"Pre-model Batch {batch_idx} debug:")
                logger.info(f"  Images shape: {images.shape}")
                logger.info(f"  Target lengths: {[len(t) for t in targets]}")
                
                # Test inference mode by temporarily switching to eval
                logger.info(f"Testing inference mode for batch {batch_idx}:")
                model.eval()  # Switch to eval mode
                with torch.no_grad():
                    try:
                        inference_output = model(images)  # No targets in eval mode
                        logger.info(f"  Inference output keys: {list(inference_output.keys())}")
                        
                        # Check preds structure
                        if 'preds' in inference_output:
                            preds = inference_output['preds']
                            logger.info(f"  Preds type: {type(preds)}")
                            logger.info(f"  Preds length: {len(preds)}")
                            if len(preds) > 0:
                                first_pred = preds[0]
                                logger.info(f"  First pred type: {type(first_pred)}")
                                logger.info(f"  First pred: {first_pred}")
                        
                        # Also check if there are any other keys with shapes
                        for key, value in inference_output.items():
                            if hasattr(value, 'shape'):
                                logger.info(f"  {key} shape: {value.shape}")
                            elif isinstance(value, (list, tuple)) and len(value) > 0:
                                if hasattr(value[0], 'shape'):
                                    logger.info(f"  {key}[0] shape: {value[0].shape}")
                                    
                    except Exception as eval_e:
                        logger.info(f"  Inference mode error: {eval_e}")
                model.train()  # Switch back to train mode
            
            # Rest of training code...
            if amp:
                with torch.cuda.amp.autocast():
                    model_output = model(images, targets)
                    train_loss = model_output["loss"]
                    
                scaler.scale(train_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                scaler.step(optimizer)
                scaler.update()
            else:
                model_output = model(images, targets)
                train_loss = model_output["loss"]
                
                train_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()

            scheduler.step()
            last_lr = scheduler.get_last_lr()[0]
            pbar.set_description(f"Training loss: {train_loss.item():.6} | LR: {last_lr:.6}")
            epoch_train_loss += train_loss.item()
            batch_cnt += 1
            
        except Exception as e:
            if logger:
                logger.warning(f"Error in batch {batch_idx}: {e}")
            continue

    epoch_train_loss /= batch_cnt if batch_cnt > 0 else 1
    return epoch_train_loss, last_lr


@torch.no_grad()
def evaluate(model, device, val_loader, batch_transforms, val_metric, amp=False, logger=None):
    """Evaluate model and return comprehensive metrics"""
    model.eval()
    val_metric.reset()
    
    val_loss, batch_cnt = 0, 0
    all_pred_texts = []
    all_true_texts = []
    
    pbar = tqdm(val_loader, desc="Validation", dynamic_ncols=True)
    for images, targets in pbar:
        try:
            images = images.to(device)
            images = batch_transforms(images)
            
            if amp:
                with torch.cuda.amp.autocast():
                    out = model(images, targets, return_preds=True)
            else:
                out = model(images, targets, return_preds=True)
            
            # Extract predictions
            if len(out["preds"]):
                words, _ = zip(*out["preds"])
            else:
                words = []
            
            # Update DocTR metric
            val_metric.update(targets, words)
            
            # Store for comprehensive metrics
            all_true_texts.extend(targets)
            all_pred_texts.extend(words if words else [""] * len(targets))

            pbar.set_description(f"Validation loss: {out['loss'].item():.6}")
            
            val_loss += out["loss"].item()
            batch_cnt += 1
            
        except Exception as e:
            if logger:
                logger.warning(f"Error in validation batch: {e}")
            continue

    val_loss /= batch_cnt if batch_cnt > 0 else 1
    
    # DocTR metrics
    doctr_result = val_metric.summary()
    exact_match = doctr_result["raw"]
    partial_match = doctr_result["unicase"]
    
    # Comprehensive metrics
    comprehensive_metrics = calculate_comprehensive_metrics(all_pred_texts, all_true_texts)
    
    return val_loss, exact_match, partial_match, comprehensive_metrics, all_pred_texts, all_true_texts


def main(config_path: str):
    """Main training function"""
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup experiment folder
    experiment_dir = setup_experiment_folder(config, config_path)
    logger = setup_logging(experiment_dir)
    
    logger.info(f"Starting experiment: {experiment_dir.name}")
    logger.info(f"Config: {json.dumps(config, indent=2)}")
    
    # Set random seeds
    torch.manual_seed(config['training']['random_seed'])
    np.random.seed(config['training']['random_seed'])
    
    # Initialize W&B
    wandb.login(key=config['wandb']['api_key'])
    wandb.init(
        project=config['wandb']['project'],
        name=experiment_dir.name,
        config=config,
        dir=str(experiment_dir)
    )
    
    # Setup device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info("Using CUDA")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")
    
    torch.backends.cudnn.benchmark = True
    
    # Load and split training data
    logger.info("Loading and splitting training data...")
    train_df = pd.read_csv(config['data']['train_csv'])

    def clean_text(text):
        # Remove BOM character
        text = text.replace('\ufeff', '')
        # Remove other control characters
        text = ''.join(char for char in text if ord(char) >= 32 or char in ['\t', '\n'])
        return text.strip()

    train_df['line_text'] = train_df['line_text'].apply(clean_text)
    
    # Validate required columns
    required_cols = ['image_filename', 'line_text']
    for col in required_cols:
        if col not in train_df.columns:
            raise ValueError(f"Required column '{col}' not found in train CSV")

    logger.info(f"Train samples: {len(train_df)}")
    # logger.info(f"Val samples: {len(val_df)}")
    


    train_split_df, val_split_df = train_test_split(
        train_df,
        test_size=config['data']['val_split_ratio'],
        random_state=config['training']['random_seed'],
        stratify=train_df['type']
    )
    
    # Create labels.json files for DocTR
    train_labels_path = create_labels_json_from_csv(train_split_df, experiment_dir, config['data']['train_image_root'], "train_labels.json")
    val_labels_path = create_labels_json_from_csv(val_split_df, experiment_dir, config['data']['train_image_root'], "val_labels.json")
    
    # Setup data transforms
    input_size = config['model']['input_size']
    batch_transforms = Normalize(mean=(0.694, 0.695, 0.693), std=(0.299, 0.296, 0.301))
    
    img_transforms = Compose([
        T.Resize((input_size, config['model']['width_multiplier'] * input_size), preserve_aspect_ratio=True),
        T.RandomApply(T.ColorInversion(), 0.1),
    ])
    
    # Create datasets
    train_set = RecognitionDataset(
        img_folder=config['data']['train_image_root'],
        labels_path=str(train_labels_path),
        img_transforms=img_transforms,
    )

    val_set = RecognitionDataset(
        img_folder=config['data']['train_image_root'],
        labels_path=str(val_labels_path),
        img_transforms=T.Resize((input_size, config['model']['width_multiplier'] * input_size), preserve_aspect_ratio=True),
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_set,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=torch.cuda.is_available(),
        collate_fn=train_set.collate_fn,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_set,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=torch.cuda.is_available(),
        collate_fn=val_set.collate_fn,
        drop_last=False,
    )
    
    logger.info(f"Training batches: {len(train_loader)}")
    logger.info(f"Validation batches: {len(val_loader)}")
    
    # Setup vocabulary
    if config['model']['vocab'] == 'combined':
        # Combine English and Ethiopic vocabularies
        english_vocab = VOCABS['english']
        ethiopic_vocab = VOCABS['ethiopic']
        additional_chars = '፡\
፣፤፥፦፠፨'
        vocab = english_vocab + ethiopic_vocab + additional_chars + '።' + ' '
        logger.info("Using combined English + Ethiopic vocabulary")
    else:
        base_vocab = VOCABS[config['model']['vocab']]
        additional_chars = '፡\
፣፤፥፦፠፨'
        # Add space character if not present
        if ' ' not in base_vocab:
            vocab = base_vocab + ' ' + additional_chars + '።'
        else:
            vocab = base_vocab + additional_chars + '።'
        logger.info(f"Using vocab: {config['model']['vocab']}")
    
    logger.info(f"Vocab length: {len(vocab)}")
    logger.info(f"Space character included: {' ' in vocab}")
    
    # Load model
    logger.info(f"Loading model: {config['model']['architecture']}")
    model = recognition.__dict__[config['model']['architecture']](
        pretrained=config['model']['pretrained'],
        input_shape=(3, input_size, config['model']['width_multiplier'] * input_size),
        vocab=vocab,
        max_length=config['model'].get('max_length', 256),   # works for most models except CRNN and VIPTR
    )
    model = model.to(device)

    # Load checkpoint if specified
    if 'checkpoint_path' in config['training'] and config['training']['checkpoint_path']:
        checkpoint_path = config['training']['checkpoint_path']
        if os.path.exists(checkpoint_path):
            logger.info(f"Loading checkpoint from: {checkpoint_path}")
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            logger.info("Checkpoint loaded successfully")
        else:
            logger.warning(f"Checkpoint path specified but file not found: {checkpoint_path}")

    # ADD THIS DEBUGGING CODE:
    logger.info("="*60)
    logger.info("MODEL ARCHITECTURE ANALYSIS")
    logger.info("="*60)

    # Print model structure
    logger.info(f"Full model structure:")
    for name, module in model.named_children():
        logger.info(f"  {name}: {module}")

    # Test with actual dimensions
    model.eval()
    with torch.no_grad():
        test_input = torch.randn(1, 3, input_size, config['model']['width_multiplier'] * input_size)
        if torch.cuda.is_available():
            test_input = test_input.to(device)
        
        logger.info(f"Input shape: {test_input.shape}")
        
        # Test backbone output
        if hasattr(model, 'backbone'):
            backbone_out = model.backbone(test_input)
            logger.info(f"Backbone output shape: {backbone_out.shape}")
            
        # Test full forward pass
        try:
            # Use model.backbone + model.head approach
            if hasattr(model, 'backbone') and hasattr(model, 'head'):
                features = model.backbone(test_input)
                logger.info(f"Features shape: {features.shape}")
                
                # Check if there's reshaping/pooling in head
                if hasattr(model.head, 'forward'):
                    # Try to get intermediate outputs
                    for name, layer in model.head.named_children():
                        logger.info(f"  Head layer {name}: {layer}")
                        
        except Exception as e:
            logger.info(f"Error in forward pass: {e}")

    logger.info("="*60)
    
    # Setup optimizer
    if config['training']['optimizer']['name'] == 'adam':
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=config['training']['optimizer']['lr'],
            betas=(0.95, 0.999),
            eps=1e-6,
            weight_decay=config['training']['optimizer']['weight_decay'],
        )
    elif config['training']['optimizer']['name'] == 'adamw':
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=config['training']['optimizer']['lr'],
            betas=(0.9, 0.999),
            eps=1e-6,
            weight_decay=config['training']['optimizer']['weight_decay'],
        )
    
    # Setup scheduler
    total_steps = config['training']['epochs'] * len(train_loader)
    if config['training']['scheduler']['name'] == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, total_steps, eta_min=config['training']['optimizer']['lr'] / 25e4)
    elif config['training']['scheduler']['name'] == 'onecycle':
        scheduler = OneCycleLR(optimizer, config['training']['optimizer']['lr'], total_steps)
    
    # Setup metrics
    val_metric = TextMatch()
    
    # Training loop
    logger.info("Starting training...")
    # min_loss = np.inf
    min_cer = np.inf 
    best_metrics = None
    
    for epoch in range(config['training']['epochs']):
        # Training
        train_loss, lr = fit_one_epoch(
            model, device, train_loader, batch_transforms, 
            optimizer, scheduler, amp=config['training']['amp'], logger=logger
        )
        
        # Validation
        val_loss, exact_match, partial_match, comprehensive_metrics, pred_texts, true_texts = evaluate(
            model, device, val_loader, batch_transforms, val_metric, 
            amp=config['training']['amp'], logger=logger
        )
        
        # Log metrics
        logger.info(f"Epoch {epoch+1}/{config['training']['epochs']}")
        logger.info(f"  Train Loss: {train_loss:.6f}")
        logger.info(f"  Val Loss: {val_loss:.6f}")
        logger.info(f"  Exact Match: {exact_match:.2%}")
        logger.info(f"  Partial Match: {partial_match:.2%}")
        logger.info(f"  CER: {comprehensive_metrics['cer']:.4f}")
        logger.info(f"  WER: {comprehensive_metrics['wer']:.4f}")
        logger.info(f"  Char Accuracy: {comprehensive_metrics['char_accuracy']:.4f}")
        logger.info(f"  Word Accuracy: {comprehensive_metrics['word_accuracy']:.4f}")
        logger.info(f"  F1 Char: {comprehensive_metrics['f1_char']:.4f}")
        logger.info(f"  F1 Word: {comprehensive_metrics['f1_word']:.4f}")
        logger.info(f"  Edit Distance: {comprehensive_metrics['edit_distance']:.4f}")
        

        # Save best model
        val_cer = comprehensive_metrics['cer']  # Extract CER from metrics
        if val_cer < min_cer:
            min_cer = val_cer
            best_metrics = comprehensive_metrics.copy()
            best_metrics.update({
                'exact_match': exact_match,
                'partial_match': partial_match,
                'val_loss': val_loss,
                'val_cer': val_cer  # Add CER to saved metrics
            })
            torch.save(model.state_dict(), experiment_dir / 'best_model.pt')
            logger.info(f"New best model saved with val_cer: {val_cer:.6f} (val_loss: {val_loss:.6f})")
        
        
        # W&B logging
        wandb.log({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'learning_rate': lr,
            'exact_match': exact_match,
            'partial_match': partial_match,
            'val_cer': comprehensive_metrics['cer'],
            'val_wer': comprehensive_metrics['wer'],
            'val_f1_char': comprehensive_metrics['f1_char'],
            'val_f1_word': comprehensive_metrics['f1_word'],
            'val_char_accuracy': comprehensive_metrics['char_accuracy'],
            'val_word_accuracy': comprehensive_metrics['word_accuracy'],
            'val_edit_distance': comprehensive_metrics['edit_distance'],
        })
    
    # Save final model
    torch.save(model.state_dict(), experiment_dir / 'final_model.pt')
    
    # Final test evaluation
    logger.info("Running final test evaluation...")
    
    # Setup test data
    test_df = pd.read_csv(config['data']['test_csv'])    
    test_labels_path = create_labels_json_from_csv(test_df, experiment_dir, config['data']['test_image_root'], "test_labels.json")


    test_set = RecognitionDataset(
        img_folder=config['data']['test_image_root'],
        labels_path=str(test_labels_path),
        img_transforms=T.Resize((input_size, config['model']['width_multiplier'] * input_size), preserve_aspect_ratio=True),
    )
    
    test_loader = DataLoader(
        test_set,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=torch.cuda.is_available(),
        collate_fn=test_set.collate_fn,
        drop_last=False,
    )
    
    # Load best model for testing
    model.load_state_dict(torch.load(experiment_dir / 'best_model.pt'))
    
    # Test evaluation
    test_metric = TextMatch()
    test_loss, test_exact, test_partial, test_comprehensive, test_pred_texts, test_true_texts = evaluate(
        model, device, test_loader, batch_transforms, test_metric,
        amp=config['training']['amp'], logger=logger
    )
    
    # Log test results
    logger.info("\n" + "="*60)
    logger.info("FINAL TEST RESULTS")
    logger.info("="*60)
    logger.info(f"Test Loss: {test_loss:.6f}")
    logger.info(f"Exact Match: {test_exact:.2%}")
    logger.info(f"Partial Match: {test_partial:.2%}")
    logger.info(f"CER: {test_comprehensive['cer']:.4f}")
    logger.info(f"WER: {test_comprehensive['wer']:.4f}")
    logger.info(f"Character Accuracy: {test_comprehensive['char_accuracy']:.4f}")
    logger.info(f"Word Accuracy: {test_comprehensive['word_accuracy']:.4f}")
    logger.info(f"F1 Character: {test_comprehensive['f1_char']:.4f}")
    logger.info(f"F1 Word: {test_comprehensive['f1_word']:.4f}")
    logger.info(f"Edit Distance: {test_comprehensive['edit_distance']:.4f}")
    logger.info("="*60)
    
    # Save test predictions with metrics
    test_results = []
    for i, (pred_text, true_text) in enumerate(zip(test_pred_texts, test_true_texts)):
        test_results.append({
            'image_filename': test_df.iloc[i]['image_filename'],
            'true_text': true_text,
            'predicted_text': pred_text,
            'label_type': test_df.iloc[i].get('label_type', 'N/A'),  # Use .get() with default
            'character_error_rate': cer([true_text], [pred_text]),
            'word_error_rate': wer([true_text], [pred_text]),
            'character_accuracy': calculate_char_accuracy(pred_text, true_text),
            'word_accuracy': calculate_word_accuracy(pred_text, true_text),
            'edit_distance': editdistance.eval(true_text, pred_text),
            'f1_char': f1_score(*char_to_binary(pred_text, true_text), average='micro', zero_division=0) if char_to_binary(pred_text, true_text)[0].size > 0 else 0.0,
            'f1_word': f1_score(*word_to_binary(pred_text, true_text), average='micro', zero_division=0) if word_to_binary(pred_text, true_text)[0].size > 0 else 0.0,
        })
    
    test_results_df = pd.DataFrame(test_results)
    test_results_df.to_csv(experiment_dir / 'test_predictions.csv', index=False)
    
    # Log final test metrics to W&B
    wandb.log({
        'test_loss': test_loss,
        'test_exact_match': test_exact,
        'test_partial_match': test_partial,
        'test_cer': test_comprehensive['cer'],
        'test_wer': test_comprehensive['wer'],
        'test_f1_char': test_comprehensive['f1_char'],
        'test_f1_word': test_comprehensive['f1_word'],
        'test_char_accuracy': test_comprehensive['char_accuracy'],
        'test_word_accuracy': test_comprehensive['word_accuracy'],
        'test_edit_distance': test_comprehensive['edit_distance'],
    })
    
    # Save experiment summary
    summary = {
        'experiment_name': experiment_dir.name,
        'total_epochs': config['training']['epochs'],
        'train_samples': len(train_split_df),
        'val_samples': len(val_split_df),
        'test_samples': len(test_df),
        'best_val_metrics': best_metrics,
        'final_test_metrics': {
            'test_loss': test_loss,
            'exact_match': test_exact,
            'partial_match': test_partial,
            **test_comprehensive
        },
        'config': config
    }
    
    with open(experiment_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    
    logger.info(f"Training completed! Results saved to {experiment_dir}")
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DocTR Text Recognition Training')
    parser.add_argument('--config', required=True, help='Path to config YAML file')
    
    args = parser.parse_args()
    
    main(args.config)