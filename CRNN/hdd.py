# %%
from __future__ import annotations
import pandas as pd
import os
import glob

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
test_csv = pd.read_csv('../raw_data/Amharic_Data/test/all_test.csv')
train_csv = pd.read_csv('../raw_data/Amharic_Data/train/all_train.csv')

# %%
all_text = ''.join(test_csv['line_text'].tolist() + train_csv['line_text'].tolist())

# %%
vocab = set(all_text)


# %%
CHARS = ''.join(sorted(vocab))

# %%


class MyDataset(Dataset):
    CHARS = ''.join(sorted(vocab))
    CHAR2LABEL = {char: i + 1 for i, char in enumerate(CHARS)}
    LABEL2CHAR = {label: char for char, label in CHAR2LABEL.items()}

    def __init__(self, root_dir=None, mode=None, df=None, img_height=80, img_width=364):
       

        self.paths = df['image_filename'].tolist()
        self.texts = df['line_text'].tolist()
        self.img_height = img_height
        self.img_width = img_width
        self.root_dir = root_dir

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = os.path.join(self.root_dir, self.paths[index])

        try:
            image = Image.open(path).convert('L')  # grey-scale
        except IOError:
            print(path)
            print('Corrupted image for %d' % index)
            return self[index + 1]

        image = image.resize((self.img_width, self.img_height), resample=Image.BILINEAR)
        image = np.array(image)
        image = image.reshape((1, self.img_height, self.img_width))
        image = (image / 127.5) - 1.0

        image = torch.FloatTensor(image)
        if self.texts:
            text = self.texts[index]
            text = self.texts[index].strip()
            
            target = [self.CHAR2LABEL[c] for c in text]
            target_length = [len(target)]

            target = torch.LongTensor(target)
            target_length = torch.LongTensor(target_length)
            return image, target, target_length
        else:
            return image


def my_collate_fn(batch):
    images, targets, target_lengths = zip(*batch)
    images = torch.stack(images, 0)
    targets = torch.cat(targets, 0)
    target_lengths = torch.cat(target_lengths, 0)
    return images, targets, target_lengths

# %%

class LabelConverter:
    """String ⇄ index-tensor converter for CTC.

    ● blank index = 0
    ● char indices start at 1 → len(charset) + 1 classes in the model.
    """

    def __init__(self, charset: str):
        self.charset   = charset
        self.blank     = 0
        self.char2idx  = {c: i + 1 for i, c in enumerate(charset)}  # 1‑based
        self.idx2char  = {i + 1: c for i, c in enumerate(charset)}

    # --------------------------- Encode ------------------------------------ #
    def encode(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        lengths = torch.tensor([len(t) for t in texts], dtype=torch.long)
        flat    = torch.tensor([self.char2idx[c] for t in texts for c in t], dtype=torch.long)
        return flat, lengths

    # --------------------------- Decode log‑probs --------------------------- #
    @torch.no_grad()
    def decode(self, log_probs: torch.Tensor, raw: bool = False) -> List[str]:
        """Greedy decode from model output (T, B, C)."""
        best = log_probs.argmax(2).permute(1, 0)  # (B,T)
        return self.decode_indices(best, remove_repeats=True, raw=raw)

    # --------------------------- Decode raw indices ------------------------ #
    def decode_indices(
        self,
        sequences: Sequence[Sequence[int]] | torch.Tensor,
        *,
        remove_repeats: bool = False,
        raw: bool = True,
    ) -> List[str]:
        """Convert index sequences → strings.

        Parameters
        ----------
        sequences : (B,T) tensor or list of lists containing *model indices* (0 = blank).
        remove_repeats : drop consecutive duplicate indices (CTC best‑path post‑process).
        raw   : if True, return the indices as space‑separated strings instead of chars.
        """
        if isinstance(sequences, torch.Tensor):
            sequences = sequences.tolist()

        out: List[str] = []
        for seq in sequences:
            chars: List[str] = []
            prev: int | None = None
            for idx in seq:
                if isinstance(idx, torch.Tensor):
                    idx = idx.item()
                if idx != self.blank:
                    if raw:
                        chars.append(str(idx))
                    else:
                        
                        chars.append(self.idx2char.get(idx, ""))
               
            out.append(" ".join(chars) if raw else "".join(chars))
        return out

# %%


# %%

data_type = "hdd"
# Assuming your DataFrame is named `df`
typed_df_1 = train_csv[train_csv['type'] == 'typed']
# data_type2 = "synthetic"
typed_df_2 = train_csv[train_csv['type'] == 'synthetic']

typed_df_3 = train_csv[train_csv['type'] == 'handwritten']

typed_df = pd.concat([typed_df_1, typed_df_2, typed_df_3], ignore_index=True)

# %%
converter = LabelConverter(MyDataset.CHARS)

# %%
print(len(typed_df))
train_df, val_df = train_test_split(typed_df, test_size=0.2, random_state=42, shuffle=True)


# %%
img_width = 1000
img_height = 100
data_dir =  '/home/admin/Gabby/Amharic OCR/HandWritten_Amharic_English_OCR/raw_data/Amharic_Data/train'

# %%
train_dataset =MyDataset(root_dir=data_dir, mode='train',
                                    df=train_df, img_height=img_height, img_width=img_width)
valid_dataset = MyDataset(root_dir=data_dir, mode='valid',
                                    df=val_df, img_height=img_height, img_width=img_width)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4,
    collate_fn=my_collate_fn,
)
valid_loader = torch.utils.data.DataLoader(
    valid_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=4,
    collate_fn=my_collate_fn,
)

# %%
# import matplotlib.pyplot as plt
# image = train_dataset.__getitem__(0)[0]
# image = image.squeeze(0).numpy()


# plt.imshow(image, cmap='gray')


# %%
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
        
        break  # remove or modify to check more
    break

# %%


# %% [markdown]
# ### Modeling

# %%


class CRNN(nn.Module):
    def __init__(self, num_classes: int, hidden_size: int = 256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1, None)),
        )
        self.map_to_seq = nn.Linear(512, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size, bidirectional=True, batch_first=True)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        Accepts (B,1,80,W) and outputs CTC‑ready log‑probs (T, B, C).
        We **average over the height dimension** instead of squeezing so we
        never hit a dim‑mismatch even if `AdaptiveAvgPool2d` leaves H>1 on
        some PyTorch builds.
        """
        feats = self.cnn(x)              # (B, 512, H', W')
        feats = feats.mean(2)            # collapse H' → (B, 512, W')
        feats = feats.permute(0, 2, 1)   # (B, W', 512)
        seq, _ = self.rnn(self.map_to_seq(feats))
        logits = self.classifier(seq)    # (B, W', C)
        log_probs = F.log_softmax(logits, dim=-1)  # (B,W',C)
        return log_probs.permute(1, 0, 2)  

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
model = CRNN(len(MyDataset.CHARS)+1).to(device)

# %%
criterion = nn.CTCLoss(blank=0, zero_infinity=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# %%
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

# %%


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

# %%


# This prompts you to enter your API key the first time
wandb.login(key="7a2297338ed4c184a8cf1c11b29bd7f0f010f9e3")

# %%
model_name = "synthetic_typed_ocr"


checkpoint = torch.load(f'{model_name}_logs/best_cer.pt')

model.load_state_dict(checkpoint['model'])

model_name = "synthetic_typed_hand_ocr"

wandb.init(project="Amharic OCR", name=model_name)
# %%
 # ── training loop ────────────────────────────────
wandb.watch(model, log="all")
epochs = 30
LOG_DIR = f"{model_name}_logs"
best_cer = float("inf")
for epoch in range(1, epochs+1):
    print(f"\nEpoch {epoch}/{epochs}")
    train_stats , _ = run_epoch(model, train_loader, criterion, optimizer, device, converter, train=True,
                           epoch_idx=epoch)
    val_stats, best_cer = run_epoch(model, valid_loader, criterion, optimizer, device, converter,
                                        train=False, save_samples=True, sample_count=10,
                                        log_dir=LOG_DIR, epoch_idx=epoch, best_cer=best_cer)
    print(f"train: {train_stats}\nval  : {val_stats}")
    ## add T to train starts keys
    train_stats = {f"train_{k}": v for k, v in train_stats.items()}
    val_stats   = {f"val_{k}": v for k, v in val_stats.items()}
    wandb.log({"epoch": epoch, **train_stats, **val_stats})

# %%


# %%


# %%



