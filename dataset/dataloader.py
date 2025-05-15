# %%
import pandas as pd
import math
import random
import torch
import yaml
import os
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from torch.nn.utils.rnn import pad_sequence


import cv2
import numpy as np
from PIL import Image, ImageDraw
from torchvision import transforms


# %%

config_path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), '..')), "config/main.yaml")
with open(config_path, "r") as file:
    config = yaml.safe_load(file)
print("config", config)

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")


# %%
class MyCustomOcrDataloader:
    def __init__(self, path_file, preprocessor, tokenizer, img_root="", transform=None):

        self.data =  pd.read_csv(path_file)
        self.data["line_text"]  = self.data["line_text"].str.strip()

        self.data  = self.data[self.data["line_text"] != '']
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.n_data_points = len(self.data)
        self.img_root = img_root
        self.transform = transform
    def __len__(self):
        return self.n_data_points
    
    def __getitem__(self, idx):

        row = self.data.iloc[idx]
        image = Image.open(os.path.join(self.img_root, row["image_filename"])).convert("RGB")
        text = row["line_text"].strip()
        if self.transform:
        #   pixel_values = pixel_values.permute(2, 0, 1)
          image = self.transform(image)
        #   pixel_values = pixel_values.permute(1, 2, 0)
        pixel_values = self.preprocessor(image, return_tensors="pt").pixel_values


        text = self.tokenizer(text, return_tensors="pt", max_length = 512, truncation=True).input_ids
        transcripts_shifted = text[:,:-1]
        golden_transcript = text[:,1:]
        
        return pixel_values, transcripts_shifted.squeeze(0), golden_transcript.squeeze(0)

    def __iter__(self):
        self.idx = 0
        return self  

    def __next__(self):
        if self.idx >= self.n_data_points:
            self.idx = 0
            raise StopIteration
        self.idx += 1
        return self[self.idx] ## calling the 
    
    def collate_fn(self, batch):
        # @NOTE: batch corresponds to output from __getitem__ for a minibatch

        '''
        1.  Extract the features and labels from 'batch'.
        2.  We will additionally need to pad both features and labels,
            look at PyTorch's documentation for pad_sequence.
        3.  This is a good place to perform transforms, if you so wish.
            Performing them on batches will speed the process up a bit.
        4.  Return batch of features, labels, lengths of features, and lengths of labels.

        '''

        batch_images              = [ i[0]  for i in batch]

        # Batch of output characters (shifted and golden).
        batch_transcript        = [i[1] for i in batch]
        batch_golden            = [i[2] for i in batch]

        lengths_image_embed            = [len(i) for i in batch_images]
        lengths_transcript      = [len(i) for i in batch_transcript]


        batch_images_pad          = pad_sequence(batch_images, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        batch_transcript_pad    = pad_sequence(batch_transcript, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        batch_golden_pad        = pad_sequence(batch_golden, batch_first=True, padding_value=self.tokenizer.pad_token_id)

        return batch_images_pad, batch_transcript_pad, batch_golden_pad, torch.tensor(lengths_image_embed), torch.tensor(lengths_transcript)



# %%
class MyOcrDataloader:
    def __init__(self, path_file, preprocessor, tokenizer, img_root=""):

        self.data =  pd.read_csv(path_file)
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.n_data_points = len(self.data)
        # self.n_data_points = 10000
        self.img_root = img_root
    def __len__(self):
        return self.n_data_points
    
    def __getitem__(self, idx):

        row = self.data.iloc[idx]
        image = Image.open(os.path.join(self.img_root, row["image_filename"])).convert("RGB")
        text = row["line_text"]
        pixel_values = self.preprocessor(image, return_tensors="pt").pixel_values
        text = self.tokenizer(text, return_tensors="pt").input_ids

        return pixel_values, text

    def __iter__(self):
        self.idx = 0
        return self  

    def __next__(self):
        if self.idx >= self.n_data_points:
            self.idx = 0
            raise StopIteration
        self.idx += 1
        return self[self.idx] ## calling the 
    
    def collate_fn(self, batch):
        # print(batch[0][1].shape,batch[5][1].shape )
        images  = torch.cat([data[0] for data in batch ], dim=0)
        # max_length = max(len(seq) for seq in batch[1])
        # padded_input_ids = torch.tensor([data[1].tolist() + [self.preprocessor.tokenizer.pad_token_id] * (max_length - len(data)) for data in batch])
        padded_input_ids = torch.nn.utils.rnn.pad_sequence([data[1].squeeze(0) for data in batch],batch_first=True, padding_value=self.tokenizer.pad_token_id)
        return images, padded_input_ids
        



class OCRDataAugmentor:
    def __init__(self):
        self.transformations = [
            self.random_rotation,
            self.gaussian_blur,
            self.dilate,
            self.erode,
            self.downscale,
            # self.underline
        ]
    
    def __call__(self, img: Image.Image) -> Image.Image:
        """Randomly apply one augmentation or return the original image."""
        ops = self.transformations + [lambda x: x]  # include original
        op = random.choice(ops)
        return op(img)

    def random_rotation(self, img: Image.Image) -> Image.Image:
        angle = random.uniform(-10, 10)
        return img.rotate(angle, resample=Image.BILINEAR)

    def gaussian_blur(self, img: Image.Image) -> Image.Image:
        img_cv = np.array(img)
        ksize = random.choice([3, 5])
        blurred = cv2.GaussianBlur(img_cv, (ksize, ksize), 0)
        return Image.fromarray(blurred)

    def dilate(self, img: Image.Image) -> Image.Image:
        img_cv = np.array(img.convert('L'))  # grayscale
        kernel = np.ones((2,2), np.uint8)
        dilated = cv2.dilate(img_cv, kernel, iterations=1)
        return Image.fromarray(dilated).convert('RGB')

    def erode(self, img: Image.Image) -> Image.Image:
        img_cv = np.array(img.convert('L'))  # grayscale
        kernel = np.ones((2,2), np.uint8)
        eroded = cv2.erode(img_cv, kernel, iterations=1)
        return Image.fromarray(eroded).convert('RGB')

    def downscale(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        scale = random.uniform(0.5, 0.8)
        new_w, new_h = int(w * scale), int(h * scale)
        img_down = img.resize((new_w, new_h), resample=Image.BILINEAR)
        img_up = img_down.resize((w, h), resample=Image.BILINEAR)
        return img_up

    def underline(self, img: Image.Image) -> Image.Image:
        img = img.copy()
        draw = ImageDraw.Draw(img)
        w, h = img.size
        y = random.randint(int(h * 0.8), h - 1)
        draw.line((0, y, w, y), fill=(0, 0, 0), width=1)
        return img

# %

######################example usage#########################
############################################################
'''
data = MyOcrDataloader(DATASET_PATH, preprocessor=processor)
train_loader = torch.utils.data.DataLoader(
    dataset     = data,
    batch_size  = config['BATCH_SIZE'],
    shuffle     = True,
    collate_fn= data.collate_fn
    )

for batch in train_loader:
    print(batch[0].shape, processor.tokenizer.batch_decode(batch[1], skip_special_tokens=True))
    break
'''
# %%


# %%



