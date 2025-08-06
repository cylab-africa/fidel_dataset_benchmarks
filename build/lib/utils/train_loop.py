import os
import json
from tqdm.auto import tqdm
from collections import Counter
import torch
import pandas as pd
from typing import Tuple, List
from utils.evaluation import _levenshtein, _split_targets


def run_epoch(model, loader, crit, opt, device, conv, *, train=True,
              save_samples: bool=False, sample_count: int=20,
              log_dir: str="logs", epoch_idx: int=0,
              best_cer: float=float("inf")) -> Tuple[dict, float]:
    os.makedirs(log_dir, exist_ok=True)
    model.train() if train else model.eval()
    name = "train" if train else "val"
    bar = tqdm(loader, desc=f"[{name}]", leave=False)

    tot_loss = tot_ed = tot_wed = 0
    tp_char = pred_tot = gt_tot = 0
    tp_word = 0
    samples = []
    tp_word = fp_word = fn_word = 0  
    for imgs, tgt, tlen in bar:
        imgs, tgt, tlen = imgs.to(device), tgt.to(device), tlen.to(device)
        with torch.set_grad_enabled(train):
            logp = model(imgs)
            in_len = torch.full((imgs.size(0),), logp.size(0), dtype=torch.long, device=device)
            loss = crit(logp, tgt, in_len, tlen)
            if train:
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        # metrics
        gts  = conv.decode_indices(_split_targets(tgt.cpu(), tlen.cpu()), remove_repeats=False)
        gts = [list(map(int, g.split())) for g in gts]
        gts = conv.decode_indices(gts, raw=False)
        
     
        preds= conv.decode(logp.detach().cpu())
        for p, g in zip(preds, gts):
            if save_samples and len(samples) < sample_count:
                samples.append({"gt": g, "pred": p})

            # Levenshtein for CER and WER
            ed = _levenshtein(p, g)
            tot_ed += ed
            tot_wed += _levenshtein(p.split(), g.split())

            # Char-level stats
            tp_char += len(g) - ed
            pred_tot += len(p)
            gt_tot += len(g)

            # Word-level stats
            p_words = p.split()
            g_words = g.split()
            p_counter = Counter(p_words)
            g_counter = Counter(g_words)

            for word in p_counter:
                tp_word += min(p_counter[word], g_counter.get(word, 0))
                fp_word += max(0, p_counter[word] - g_counter.get(word, 0))
            for word in g_counter:
                fn_word += max(0, g_counter[word] - p_counter.get(word, 0))

            tot_loss += loss.item()
            bar.set_postfix(loss=loss.item())

    cer = tot_ed/max(1,gt_tot)
    wer = tot_wed/max(1,gt_tot)
    prec = tp_char/pred_tot if pred_tot else 0; rec = tp_char/gt_tot if gt_tot else 0
    f1c = 2*prec*rec/(prec+rec) if prec+rec else 0
    word_prec = tp_word / (tp_word + fp_word + 1e-8)
    word_rec = tp_word / (tp_word + fn_word + 1e-8)
    f1w = 2 * word_prec * word_rec / (word_prec + word_rec + 1e-8)
    stats = dict(loss=tot_loss/len(loader), CER=cer, WER=wer, F1_char=f1c, F1_word=f1w)

    # ── save sample predictions
    if save_samples:
        samp_path = os.path.join(log_dir, f"samples_epoch{epoch_idx}.json")
        with open(samp_path, "w", encoding="utf8") as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)

    # ── checkpoint best CER
    if not train and cer < best_cer:
        ckpt = {
            "epoch": epoch_idx,
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "CER": cer,
        }
        torch.save(ckpt, os.path.join(log_dir, "best_cer.pt"))
        best_cer = cer

    return stats, best_cer
