import os
from typing import List, Tuple, Sequence
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from typing import Sequence
class MyDataset(Dataset):
    

    def __init__(self, root_dir=None, mode=None, df=None, img_height=80, img_width=364, vocab=None):
       

        self.paths = df['image_filename'].tolist()
        self.texts = df['line_text'].tolist()
        self.img_height = img_height
        self.img_width = img_width
        self.root_dir = root_dir

        self.CHARS = ''.join(sorted(vocab))



        self.CHAR2LABEL = {char: i + 1 for i, char in enumerate(self.CHARS)}
        self.LABEL2CHAR = {label: char for char, label in self.CHAR2LABEL.items()}



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

