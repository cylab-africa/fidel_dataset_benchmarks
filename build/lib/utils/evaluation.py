
import editdistance  # For calculating Levenshtein distance (for CER)

import json
from tqdm.auto import tqdm
from collections import Counter
import torch
import pandas as pd
from typing import Tuple
from typing import List


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j]+1,      # deletion
                            cur[-1]+1,     # insertion
                            prev[j-1] + (ca != cb)))  # substitution
        prev = cur
    return prev[-1]


def _update_char_counts(pred: str, gt: str):
    dist = _levenshtein(pred, gt)
    tp = len(gt) - dist  # correct chars = ground truth − edit distance
    return tp, len(pred), len(gt)

def _split_targets(targets: torch.Tensor, lengths: torch.Tensor) -> List[List[int]]:
    out, ptr = [], 0
    for l in lengths:
        out.append(targets[ptr:ptr+l].tolist())
        ptr += l
    return out


def predict(model, loader, device, conv, ) -> Tuple[dict, float]:
    model.eval()
    name =  "test"
    bar = tqdm(loader, desc=f"[{name}]", leave=False)

    results = {'preds': [], 'gts': []}
    for imgs, tgt, tlen in bar:
        imgs, tgt, tlen = imgs.to(device), tgt.to(device), tlen.to(device)
        with torch.set_grad_enabled(False):
            logp = model(imgs)
            in_len = torch.full((imgs.size(0),), logp.size(0), dtype=torch.long, device=device)
           
        gts  = conv.decode_indices(_split_targets(tgt.cpu(), tlen.cpu()), remove_repeats=False)
        gts = [list(map(int, g.split())) for g in gts]
        gts = conv.decode_indices(gts, raw=False)
        
     
        preds= conv.decode(logp.detach().cpu())
        for p, g in zip(preds, gts):
            results['preds'].append(p)
            results['gts'].append(g)
        bar.set_postfix()
            
    result_df = pd.DataFrame(results)
    return result_df


def calculate_metrics(test_df):
    # Initialize variables to store total errors and counts
    total_subs, total_deletions, total_insertions, total_words, total_chars = 0, 0, 0, 0, 0
    total_true_positive_words, total_false_positive_words, total_false_negative_words = 0, 0, 0
    total_true_positive_chars, total_false_positive_chars, total_false_negative_chars = 0, 0, 0
    
    for idx, row in test_df.iterrows():
        # Get ground truth and predicted text
        ground_truth = row['gts'].strip()
        prediction = row['preds'].strip()
        
        # Split ground truth and prediction into words
        gt_words = ground_truth.split()
        pred_words = prediction.split()
        
        # Calculate WER (Word Error Rate)
        # Using Levenshtein distance (edit distance) for word-level WER
        wer_result = editdistance.eval(gt_words, pred_words)
        total_subs += wer_result  # Substitutions in words
        total_deletions += len(gt_words) - len(pred_words)  # Deletions in words
        total_insertions += len(pred_words) - len(gt_words)  # Insertions in words
        total_words += len(gt_words)
        
        # Calculate F1 at word level
        true_positives_words = len(set(gt_words).intersection(pred_words))
        false_positives_words = len(pred_words) - true_positives_words
        false_negatives_words = len(gt_words) - true_positives_words
        
        total_true_positive_words += true_positives_words
        total_false_positive_words += false_positives_words
        total_false_negative_words += false_negatives_words
        
        # Calculate CER (Character Error Rate)
        gt_chars = ''.join(gt_words)
        pred_chars = ''.join(pred_words)
        
        # Using Levenshtein distance (edit distance) for character-level CER
        cer_result = editdistance.eval(gt_chars, pred_chars)
        total_subs += cer_result  # Substitutions in characters
        total_deletions += len(gt_chars) - len(pred_chars)  # Deletions in characters
        total_insertions += len(pred_chars) - len(gt_chars)  # Insertions in characters
        total_chars += len(gt_chars)
        
        # Calculate F1 at character level
        true_positives_chars = len(set(gt_chars).intersection(pred_chars))
        false_positives_chars = len(pred_chars) - true_positives_chars
        false_negatives_chars = len(gt_chars) - true_positives_chars
        
        total_true_positive_chars += true_positives_chars
        total_false_positive_chars += false_positives_chars
        total_false_negative_chars += false_negatives_chars
        
    # Calculate WER
    wer = (total_subs + total_deletions + total_insertions) / total_words if total_words > 0 else 0
    
    # Calculate CER
    cer = (total_subs + total_deletions + total_insertions) / total_chars if total_chars > 0 else 0
    
    # Calculate F1 Score at word level
    precision_word = total_true_positive_words / (total_true_positive_words + total_false_positive_words) if total_true_positive_words + total_false_positive_words > 0 else 0
    recall_word = total_true_positive_words / (total_true_positive_words + total_false_negative_words) if total_true_positive_words + total_false_negative_words > 0 else 0
    f1_word = (2 * precision_word * recall_word) / (precision_word + recall_word) if precision_word + recall_word > 0 else 0
    
    # Calculate F1 Score at character level
    precision_char = total_true_positive_chars / (total_true_positive_chars + total_false_positive_chars) if total_true_positive_chars + total_false_positive_chars > 0 else 0
    recall_char = total_true_positive_chars / (total_true_positive_chars + total_false_negative_chars) if total_true_positive_chars + total_false_negative_chars > 0 else 0
    f1_char = (2 * precision_char * recall_char) / (precision_char + recall_char) if precision_char + recall_char > 0 else 0
    
    return wer, cer, f1_word, f1_char