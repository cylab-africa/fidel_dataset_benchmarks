# %%
import sys
sys.path.append("..")
from dataset.dataloader import MyOcrDataloader, MyCustomOcrDataloader, OCRDataAugmentor
import pandas as pd
import math
import random
import torch
import yaml
import wandb
import os
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import gc
from utils.utils import *
from utils.charactertokenizer import CharacterTokenizer
from jiwer import wer, cer
from models.models import TrOCRMyDecoder
from models.vit import ViT

from tqdm.auto import tqdm
from transformers import AutoTokenizer
from transformers import PreTrainedTokenizerFast

import json

# %%
# encoder = ViT(image_size = (256, 1024), num_classes= None, mlp_dim=1024, patch_size=32, dim=512, depth= 4, heads = 4 )
# img = torch.randn(1, 3, 256, 1024)

# preds = encoder(img) # (1, 1000)
# preds.shape

# %%


# %%
config_path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), '..')), "config/main.yaml")
with open(config_path, "r") as file:
    config = yaml.safe_load(file)
task  = config["TRAIN_TASK"]
os.environ["TOKENIZERS_PARALLELISM"] = "false"

TRAIN_PATH =config[task+"_"+"TRAIN_PATH"]
VAL_PATH =config[task+"_"+"VAL_PATH"]
IMG_ROOT = config[task+"_"+"IMG_ROOT"]

MODEL_ID = config["MODEL_ID"]

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = resize_and_patch_image
# model =VisionEncoderDecoderModel.from_pretrained(MODEL_ID).to(device)
# model.config.decoder_start_token_id = processor.tokenizer.eos_token_id
# model.config.pad_token_id = processor.tokenizer.pad_token_id
# model.config.vocab_size = model.config.decoder.vocab_size

# %%

# tokenizer = AutoTokenizer.from_pretrained("rasyosef/bert-amharic-tokenizer")
# tokenizer.tokenize("የዓለምአቀፉ ነጻ ንግድ መስፋፋት ድህነትን ለማሸነፍ በሚደረገው ትግል አንዱ ጠቃሚ መሣሪያ ሊሆን መቻሉ ብዙ የሚነገርለት ጉዳይ ነው።")
# tokenizer.vocab_size

# %%




# %%
## sanity check for data loader
os.chdir("../dataset")
augmentor = OCRDataAugmentor()
# tokenizer = CharacterTokenizer.from_pretrained('/home/ubuntu/HandWritten_Amharic_English_OCR/Amharic_Char_Tokenizer2')
tokenizer = PreTrainedTokenizerFast.from_pretrained("/home/ubuntu/data/synthetic_data/amharic_tokenizer_hf")

# tokenizer = AutoTokenizer.from_pretrained("rasyosef/bert-amharic-tokenizer")

if tokenizer.bos_token_id== None:
    print("setting tokenizer")
    special_tokens_dict = {
    "bos_token": "<sos>",
    "eos_token": "<eos>"
    }
    tokenizer.add_special_tokens(special_tokens_dict)

train_data = MyCustomOcrDataloader(TRAIN_PATH, preprocessor=processor, tokenizer  = tokenizer, img_root=IMG_ROOT, transform=augmentor)
IMG_ROOT = config[task+"_IMG_ROOT"+"_TEST"]
val_data = MyCustomOcrDataloader(VAL_PATH, preprocessor=processor, tokenizer  = tokenizer, img_root=IMG_ROOT)
train_loader = torch.utils.data.DataLoader(
    dataset     = train_data,
    batch_size  = config['BATCH_SIZE'],
    shuffle     = True,
    collate_fn= train_data.collate_fn
    )



# %%

train_loader    = torch.utils.data.DataLoader(
    dataset     = train_data,
    batch_size  = config["BATCH_SIZE"],
    shuffle     = True,
    num_workers = 4,
    pin_memory  = True,
    collate_fn  = train_data.collate_fn
)

val_loader      = torch.utils.data.DataLoader(
    dataset     = val_data,
    batch_size  = config["BATCH_SIZE"],
    shuffle     = False,
    num_workers = 2,
    pin_memory  = True,
    collate_fn  = train_data.collate_fn,
)

print("No. of Train Images   : ", train_data.__len__())
print("Batch Size           : ", config["BATCH_SIZE"])
print("Train Batches        : ", train_loader.__len__())
print("Val Batches          : ", val_loader.__len__())

# %%
''' Sanity Check '''

print("Checking the Shapes of the Data --\n")

for batch in train_loader:
    x_pad, y_shifted_pad, y_golden_pad, x_len, y_len, = batch

    print(f"x_pad shape:\t\t{x_pad.shape}")
    print(f"x_len shape:\t\t{x_len.shape}\n")

    print(f"y_shifted_pad shape:\t{y_shifted_pad.shape}")
    print(f"y_golden_pad shape:\t{y_golden_pad.shape}")
    print(f"y_len shape:\t\t{y_len.shape}\n")
    plt.imshow(x_pad[0][0].permute((1,2,0)))
    # print(y_shifted_pad)

    break

# %%
''' Please refer to the config file and top sections to fill in the following '''

model = TrOCRMyDecoder(
input_dim                   = None,
dec_num_layers              = config["dec_num_layers"],
dec_num_heads               = config["dec_num_heads"],

d_model                     = config["d_model"],
d_ff                        = config["d_ff"],

target_vocab_size           = len(tokenizer),
eos_token                   = tokenizer.eos_token_id,
sos_token                   = tokenizer.bos_token_id,
pad_token                   = tokenizer.pad_token_id,

enc_dropout                 = config["enc_dropout"],
dec_dropout                 = config["enc_dropout"],
pre_train=False,
# decrease to a small number if you are just trying to implement the network
max_seq_length              = 512 , # Max sequence length for transcripts. Check data verification.
).to(device)


checkpoint = torch.load("/home/ubuntu/HandWritten_Amharic_English_OCR/dataset/checkpoints-llm-basic-cnn-transformer/TYPEDcheckpoint-best-loss-model.pth")
model.load_state_dict(checkpoint['model_state_dict'])
def num_parameters(mode):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params / 1E6

para = num_parameters(model)
print("#"*10)
print(f"Model Parameters:\n {para}")
print("#"*10)

# %%
### freeze encoder
# full_model.encoder.embedding.load_state_dict(model.encoder.state_dict())
for param in model.encoder.parameters():
    param.requires_grad = True # TODO make it non-trainable


# for param in model.decoder.parameters():
#     param.requires_grad = True # TODO make it non-trainable
# %%
def train_model(model, train_loader, optimizer):

    model.train()
    batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, leave=False, position=0, desc="Train")

    total_loss          = 0
    running_loss        = 0.0
    running_perplexity  = 0.0

    for i, (inputs, targets_shifted, targets_golden, inputs_lengths, targets_lengths) in enumerate(train_loader):

        optimizer.zero_grad()

        inputs          = inputs.to(device)
        targets_shifted = targets_shifted.to(device)
        targets_golden  = targets_golden.to(device)

        with torch.cuda.amp.autocast():
            # passing the minibatch through the model
            # raw_predictions, attention_weights = model(inputs, inputs_lengths, targets_shifted, targets_lengths)
            raw_predictions, attention_weights = model(inputs, inputs_lengths, targets_shifted, targets_lengths)


            padding_mask = torch.logical_not(torch.eq(targets_shifted, tokenizer.pad_token_id))

            # cast the mask to float32
            padding_mask = padding_mask.float()
            loss = loss_func(raw_predictions.transpose(1,2), targets_golden)*padding_mask
            loss = loss.sum() / padding_mask.sum()

        scaler.scale(loss).backward()   # This is a replacement for loss.backward()
        scaler.step(optimizer)          # This is a replacement for optimizer.step()
        scaler.update()                 # This is something added just for FP16

        running_loss        += float(loss.item())
        perplexity          = torch.exp(loss)
        running_perplexity  += perplexity.item()

        # online training monitoring
        batch_bar.set_postfix(
            loss = "{:.04f}".format(float(running_loss / (i + 1))),
            perplexity = "{:.04f}".format(float(running_perplexity / (i + 1)))
        )

        batch_bar.update()

        del inputs, targets_shifted, targets_golden, inputs_lengths, targets_lengths
        torch.cuda.empty_cache()

    running_loss        = float(running_loss / len(train_loader))
    running_perplexity  = float(running_perplexity / len(train_loader))

    batch_bar.close()

    return running_loss, running_perplexity, attention_weights

# %%

def validate_fast(model, dataloader):
    model.eval()

    # progress bar
    batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, leave=False, position=0, desc="Val", ncols=5)

    running_distance = 0.0
    running_cer = 0.0
    running_wer = 0.0
    running_char_f1 = 0.0
    running_word_f1 = 0.0


    for i, (inputs, targets_shifted, targets_golden, inputs_lengths, targets_lengths) in enumerate(dataloader):

        inputs  = inputs.to(device)
        targets_golden = targets_golden.to(device)

        with torch.inference_mode():
            greedy_predictions = model.recognize(inputs, inputs_lengths)

        # calculating Levenshtein Distance
        # @NOTE: modify the print_example to print more or less validation examples
        dist, cer, wer, charf1, word_f1 = calc_edit_distance(greedy_predictions, targets_golden, targets_lengths, tokenizer, print_example=True)
        running_distance += dist
        running_cer += cer
        running_wer += wer
        running_char_f1 += charf1
        running_word_f1 += word_f1


        # online validation distance monitoring
        batch_bar.set_postfix(
            running_distance = "{:.04f}".format(float(running_distance / (i + 1))),
            running_cer = "{:.04f}".format(float(running_cer / (i + 1))),
            running_wer = "{:.04f}".format(float(running_wer / (i + 1))),
            running_char_f1 = "{:.04f}".format(float(running_char_f1 / (i + 1))),
            running_word_f1 = "{:.04f}".format(float(running_word_f1 / (i + 1)))

        )

        batch_bar.update()

        del inputs, targets_shifted, targets_golden, inputs_lengths, targets_lengths
        torch.cuda.empty_cache()

        if i==4: break      # validating only upon first five batches

    batch_bar.close()
    running_distance /= 5
    running_cer /= 5
    running_wer /= 5
    running_char_f1 /= 5
    running_word_f1 /= 5

    return running_distance, running_cer, running_wer, running_char_f1, running_word_f1

def validate_full(model, dataloader):
    model.eval()

    # progress bar
    batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, leave=False, position=0, desc="Val", ncols=5)

    running_distance = 0.0
    running_cer = 0.0
    running_wer = 0.0
    running_char_f1 = 0.0
    running_word_f1 =0.0

    for i, (inputs, targets_shifted, targets_golden, inputs_lengths, targets_lengths) in enumerate(dataloader):

        inputs  = inputs.to(device)
        targets_golden = targets_golden.to(device)

        with torch.inference_mode():
            greedy_predictions = model.recognize(inputs, inputs_lengths)

        # calculating Levenshtein Distance
        # @NOTE: modify the print_example to print more or less validation examples
        dist, cer, wer, charf1, word_f1 = calc_edit_distance(greedy_predictions, targets_golden, targets_lengths, tokenizer, print_example=False)
        running_distance += dist
        running_cer += cer
        running_wer += wer
        running_char_f1 += charf1
        running_word_f1 += word_f1


        # online validation distance monitoring
        batch_bar.set_postfix(
            running_distance = "{:.04f}".format(float(running_distance / (i + 1))),
            running_cer = "{:.04f}".format(float(running_cer / (i + 1))),
            running_wer = "{:.04f}".format(float(running_wer / (i + 1))),
            running_char_f1 = "{:.04f}".format(float(running_char_f1 / (i + 1))),
            running_word_f1 = "{:.04f}".format(float(running_word_f1 / (i + 1)))

        )
        batch_bar.update()

        del inputs, targets_shifted, targets_golden, inputs_lengths, targets_lengths
        torch.cuda.empty_cache()


    batch_bar.close()
    running_distance /= len(dataloader)
    running_cer /= len(dataloader)
    running_wer /= len(dataloader)
    running_char_f1 /= len(dataloader)
    running_word_f1 /= len(dataloader)


    return running_distance, running_cer, running_wer, running_char_f1, running_word_f1

# %%
''' defining optimizer '''
# vocab_size = len(tokenizer)
# weights = torch.ones(vocab_size).to("cuda")  # default weight = 1 for all
# whitespace_token_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(' ')[0])
# # Decrease weight for whitespace token
# weights[whitespace_token_id] = 0.1  # e.g., reduce impact by 90%
loss_func   = torch.nn.CrossEntropyLoss(ignore_index = tokenizer.pad_token_id)
scaler      = torch.cuda.amp.GradScaler()
if config["optimizer"] == "SGD":
  # feel free to change any of the initializations you like to fit your needs
  optimizer = torch.optim.SGD(model.parameters(),
                              lr=config["learning_rate"],
                              momentum=config["momentum"],
                              weight_decay=1E-4,
                              nesterov=config["nesterov"])

elif config["optimizer"] == "Adam":
  # feel free to change any of the initializations you like to fit your needs
  optimizer = torch.optim.Adam(model.parameters(),
                               lr=float(config["learning_rate"]),
                               weight_decay=1e-4)

elif config["optimizer"] == "AdamW":
  # feel free to change any of the initializations you like to fit your needs
  optimizer = torch.optim.AdamW(model.parameters(),
                                lr=float(config["learning_rate"]),
                                weight_decay=0.01)

''' defining scheduler '''

if config["scheduler"] == "ReduceLR":
  #Feel Free to change any of the initializations you like to fit your needs
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                factor=config["factor"], patience=config["patience"], min_lr=1E-8, verbose=True)

elif config["scheduler"] == "CosineAnnealing":
  #Feel Free to change any of the initializations you like to fit your needs
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                T_max = 35, eta_min=1E-8)

# %%
device

# %%
# using WandB? resume training?

USE_WANDB = config["USE_WANDB"]
RESUME_LOGGING = False

# creating your WandB run
run_name = "{}_Transformer_ENC-{}/{}_DEC-{}/{}_{}_{}_{}_{}".format(
    config["Name"],
    config["enc_num_layers"],       # only used in Part II with the Transformer Encoder
    config["enc_num_heads"],        # only used in Part II with the Transformer Encoder
    config["dec_num_layers"],
    config["dec_num_heads"],
    config["d_model"],
    config["d_ff"],
    config["optimizer"],
    config["scheduler"])
task = "handonly"

if USE_WANDB:

    wandb.login(key="3c7b273814544590b64c54d9a5242bde38616e02", relogin=True) # TODO enter your key here

    if RESUME_LOGGING:
        run_id = ""
        run = wandb.init(
            id     = run_id,        ### Insert specific run id here if you want to resume a previous run
            resume = True,          ### You need this to resume previous runs, but comment out reinit=True when using this
            project = task+"ocr-cnn-lstm",  ### Project should be created in your wandb account
        )

    else:
        run = wandb.init(
            name    = run_name,     ### Wandb creates random run names if you skip this field, we recommend you give useful names
            reinit  = True,         ### Allows reinitalizing runs when you re-run this cell
            project = task+"ocr-cnn-lstm",  ### Project should be created in your wandb account
            config  = config        ### Wandb Config for your run
        )

        ### Save your model architecture as a string with str(model)
        model_arch  = str(model)

        ### Save it in a txt file
        arch_file   = open("model_arch.txt", "w")
        file_write  = arch_file.write(model_arch)
        arch_file.close()

        ### Log it in your wandb run with wandb.save()
        # wandb.save("model_arch.txt")

# %%
e                   = 0
best_loss           = 1000
best_distance  = 0
checkpoint_root = os.path.join(os.getcwd(), "checkpoints-llm-basic-cnn-transformer")
os.makedirs(checkpoint_root, exist_ok=True)
if USE_WANDB:
    wandb.watch(model, log="all")
task =  config["TRAIN_TASK"]
checkpoint_best_loss_model_filename     = task +'checkpoint-best-loss-model.pth' 
checkpoint_best_distance_model_filename     = task +'checkpoint-best-distance-model.pth'

checkpoint_last_epoch_filename          = 'checkpoint-epoch-'
best_loss_model_path                    = os.path.join(checkpoint_root, checkpoint_best_loss_model_filename)
best_distance_model_path                    = os.path.join(checkpoint_root, checkpoint_best_distance_model_filename)


if RESUME_LOGGING:
    # change if you want to load best test model accordingly
    checkpoint = torch.load(wandb.restore(checkpoint_best_loss_model_filename, run_path=""+run_id).name)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    e = checkpoint['epoch']

    print("Resuming from epoch {}".format(e+1))
    print("Epochs left: ", config['epochs']-e)
    print("Optimizer: \n", optimizer)

torch.cuda.empty_cache()
gc.collect()

# epochs = config["epochs"]
epochs = 100

for epoch in range(e, epochs):

    print("\nEpoch {}/{}".format(epoch+1, epochs))

    curr_lr = float(optimizer.param_groups[0]["lr"])

    train_loss, train_perplexity, attention_weights = train_model(model, train_loader, optimizer)

    print("\nEpoch {}/{}: \nTrain Loss {:.04f}\t Train Perplexity {:.04f}\t Learning Rate {:.04f}".format(
        epoch + 1, config["epochs"], train_loss, train_perplexity, curr_lr))

    if (epoch >-1 )and (epoch % 10 == 0):    # validate every 2 epochs to speed up training
        levenshtein_distance, cer, wer, charf1, wordf1 = validate_fast(model, val_loader)
        if best_distance <= levenshtein_distance:
            best_distance = levenshtein_distance
            save_model(model, optimizer, scheduler, ['val_distance', levenshtein_distance], epoch, best_distance_model_path)
            # wandb.save(best_loss_model_path)
            print("Saved distance training model")
        print("Levenshtein Distance {:.04f}".format(levenshtein_distance))
        if USE_WANDB:
            wandb.log({"train_loss"     : train_loss,
                    "train_perplexity"  : train_perplexity,
                    "learning_rate"     : curr_lr,
                    "val_distance"      : levenshtein_distance,
                    "charf1": charf1,
                    "wordf1": wordf1,})

    else:
        if USE_WANDB:

            wandb.log({"train_loss"     : train_loss,
                    "train_perplexity"  : train_perplexity,
                    "learning_rate"     : curr_lr})

    # # plotting the encoder-nearest and decoder-nearest attention weights
    # attention_keys = list(attention_weights.keys())

    # attention_weights_decoder_self       = attention_weights[attention_keys[0]][0].cpu().detach().numpy()
    # attention_weights_decoder_cross      = attention_weights[attention_keys[-1]][0].cpu().detach().numpy()

    # # saving the cross-attention weights
    # save_attention_plot(attention_weights_decoder_cross, epoch+100)

    # plot_attention_weights((attention_weights[attention_keys[0]][0]).cpu().detach().numpy())
    # plot_attention_weights(attention_weights[attention_keys[-1]][0].cpu().detach().numpy())

    # if config["scheduler"] == "ReduceLR":
    #     scheduler.step(levenshtein_distance)
    # else:
    #     scheduler.step()

    # ### Highly Recommended: Save checkpoint in drive and/or wandb if accuracy is better than your current best
    # epoch_model_path = os.path.join(checkpoint_root, (checkpoint_last_epoch_filename + str(epoch) + '.pth'))
    # save_model(model, optimizer, scheduler, ['train_loss', train_loss], epoch, epoch_model_path)
    ## wandb.save(epoch_model_path) ## Can't save on wandb for all epochs, may blow up storage


    if best_loss >= train_loss:
        best_loss = train_loss
        save_model(model, optimizer, scheduler, ['train_loss', train_loss], epoch, best_loss_model_path)
        # wandb.save(best_loss_model_path)
        print("Saved best training model")

### Finish your wandb run
# run.finish()

# %%
#### sweeper eval
# checkpoint = torch.load(best_distance_model_path)
# model.load_state_dict(checkpoint['model_state_dict'])
result = {}
for task in ["HANDWRITTEN", "SYNTHETIC", "TYPED"]:
    print(f"{config["TRAIN_TASK"]} is been evaluated on {task}")
    VAL_PATH =config[task+"_VAL_PATH"]
    IMG_ROOT = config[task+"_IMG_ROOT"+"_TEST"]
    
    
    

    val_data = MyCustomOcrDataloader(VAL_PATH, preprocessor=processor, tokenizer  = tokenizer, img_root=IMG_ROOT)
    val_loader      = torch.utils.data.DataLoader(
    dataset     = val_data,
    batch_size  = config["BATCH_SIZE"],
    shuffle     = False,
    num_workers = 2,
    pin_memory  = True,
    collate_fn  = train_data.collate_fn,
    )
    levenshtein_distance, cer, wer, charf1, wordf1 = validate_full(model, val_loader)
    result[task]  = {}
    for metric, score in zip (["lev", "cer", "wer", "charf1", "wordf1"], [levenshtein_distance, cer, wer, charf1, wordf1]):
        result[task][metric] = score

    with open(f"llm_result{config["TRAIN_TASK"]}_.json", 'w') as file:
        json.dump(result, file, indent=4)

# %%



