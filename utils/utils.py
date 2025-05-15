import torch
import Levenshtein
import seaborn as sns
import matplotlib.pyplot as plt
import jiwer
from PIL import Image
import numpy as np
from types import SimpleNamespace

def forward_pass_with_labels(model, batch):
    pixel_values = batch[0].to(model.device)
    labels = batch[1].to(model.device)
    result = model(pixel_values, labels =labels )
    return result.loss, result.logits

def update(model, optimizer, scaler, batch, device="cuda"):
    with torch.autocast(device_type=device, dtype=torch.float16):
        loss, logits = forward_pass_with_labels(model, batch)
    # print(loss)
    scaler.scale(loss).backward()

    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
    return loss.item(), logits


# def train(model, )
def save_model(model, optimizer, scheduler, metric, epoch, path):
    torch.save(
        {"model_state_dict"         : model.state_dict(),
         "optimizer_state_dict"     : optimizer.state_dict(),
         "scheduler_state_dict"     : scheduler.state_dict() if scheduler is not None else {},
         metric[0]                  : metric[1],
         "epoch"                    : epoch},
         path
    )

def load_model(path, model, metric= "valid_acc", optimizer= None, scheduler= None):

    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer != None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler != None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    epoch   = checkpoint["epoch"]
    metric  = checkpoint[metric]

    return [model, optimizer, scheduler, epoch, metric]



''' utility function for Levenshtein Distantce quantification '''
from sklearn.metrics import precision_recall_fscore_support
import jiwer
import Levenshtein

from sklearn.metrics import f1_score
import jiwer
import Levenshtein

def calc_edit_distance(predictions, y, y_len, tokenizer, print_example=False):
    dist = 0.0
    cer = 0.0
    wer = 0.0
    char_f1_total = 0.0
    word_f1_total = 0.0

    batch_size, seq_len = predictions.shape

    for batch_idx in range(batch_size):
        y_sliced    = tokenizer.decode(y[batch_idx, 0 : y_len[batch_idx]], skip_special_tokens=True)
        pred_sliced = tokenizer.decode(predictions[batch_idx], skip_special_tokens=True)

        y_string    = "".join(y_sliced)
        pred_string = "".join(pred_sliced)

        cer += jiwer.cer(y_string, pred_string)
        wer += jiwer.wer(y_string, pred_string)
        dist += Levenshtein.distance(pred_string, y_string)

        # Character F1 per sample
        y_chars = list(y_string)
        pred_chars = list(pred_string)

        common_len = min(len(y_chars), len(pred_chars))
        if common_len > 0:
            char_f1 = f1_score(y_chars[:common_len], pred_chars[:common_len], average='micro', zero_division=0)
            char_f1_total += char_f1

        # Word F1 per sample
        y_words = y_string.split()
        pred_words = pred_string.split()
        common_len = min(len(y_words), len(pred_words))
        if common_len > 0:
            word_f1 = f1_score(y_words[:common_len], pred_words[:common_len], average='micro', zero_division=0)
            word_f1_total += word_f1

        if print_example and batch_idx == 0:
            print("\nGround Truth : ", y_string)
            print("Prediction   : ", pred_string)

    dist /= batch_size
    cer /= batch_size
    wer /= batch_size
    char_f1_avg = char_f1_total / batch_size
    word_f1_avg = word_f1_total / batch_size

    return dist, cer, wer, char_f1_avg, word_f1_avg



def compute_metrics(predictions, actuals, print_example=False):
    dist = 0.0
    cer = 0.0
    wer = 0.0
    char_f1_total = 0.0
    word_f1_total = 0.0

    batch_size =  len(predictions)

    for batch_idx in range(batch_size):
        y_string = actuals[batch_idx]
        pred_string = predictions[batch_idx]

        cer += jiwer.cer(y_string, pred_string)
        wer += jiwer.wer(y_string, pred_string)
        dist += Levenshtein.distance(pred_string, y_string)

        # Character F1 per sample
        y_chars = list(y_string)
        pred_chars = list(pred_string)

        common_len = min(len(y_chars), len(pred_chars))
        if common_len > 0:
            char_f1 = f1_score(y_chars[:common_len], pred_chars[:common_len], average='micro', zero_division=0)
            char_f1_total += char_f1

        # Word F1 per sample
        y_words = y_string.split()
        pred_words = pred_string.split()
        common_len = min(len(y_words), len(pred_words))
        if common_len > 0:
            word_f1 = f1_score(y_words[:common_len], pred_words[:common_len], average='micro', zero_division=0)
            word_f1_total += word_f1

        if print_example and batch_idx == 0:
            print("\nGround Truth : ", y_string)
            print("Prediction   : ", pred_string)

    # dist /= batch_size
    # cer /= batch_size
    # wer /= batch_size
    # char_f1_avg = char_f1_total / batch_size
    # word_f1_avg = word_f1_total / batch_size

    return dist, cer, wer, char_f1_total, word_f1_total

def save_attention_plot(attention_weights, epoch=0):
    ''' function for saving attention weights plot to a file

        @NOTE: default starter code set to save cross attention
    '''

    plt.clf()  # Clear the current figure
    sns.heatmap(attention_weights, cmap="GnBu")  # Create heatmap

    # Save the plot to a file. Specify the directory if needed.
    plt.savefig(f"cross_attention-epoch{epoch}.png")


def resize_and_patch_image(img, target_size=(128, 1536), patch_size=(32, 32),return_tensors ="pt"):
    """
    Resize image while maintaining aspect ratio, pad to target_size, and split into patches of patch_size.
    Returns a list of patches as NumPy arrays.
    """
    target_height, target_width = target_size
    patch_height, patch_width = patch_size


    # Convert to RGB if grayscale or other mode
    img = img.convert("L")
    # img = img.convert("RGB")

    
    # Resize while maintaining aspect ratio
    aspect_ratio = img.width / img.height
    new_height = target_height
    new_width = min(int(aspect_ratio * new_height), target_width)
    img_resized = img.resize((new_width, new_height), Image.BICUBIC)
    # Pad to target size
    # padded_img = Image.new("RGB", (target_width, target_height), (0, 0, 0))
    padded_img = Image.new("L", (target_width, target_height), 0)
    padded_img.paste(img_resized, (0, 0))
    
    # Convert to numpy and extract patches
    img_array = torch.tensor(np.array(padded_img))
    # img_array = img_array.permute((2,0,1)).unsqueeze(0)/255
    img_array = torch.tensor(np.array(padded_img)).unsqueeze(0).unsqueeze(0) / 255.0
    return SimpleNamespace(pixel_values=img_array) # img_array
    patches = []
    for y in range(0, target_height, patch_height):
        for x in range(0, target_width, patch_width):
            patch = img_array[y:y+patch_height, x:x+patch_width]
            if patch.shape[0] == patch_height and patch.shape[1] == patch_width:
                patches.append(patch)
        
        return patches