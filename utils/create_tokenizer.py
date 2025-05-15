from typing import Dict, List, Optional, Sequence, Union
import json
import os
from pathlib import Path
import torch

from charactertokenizer import CharacterTokenizer

raw_data = {i: char for i, char in enumerate("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?-:")}

# Create vocabulary list (excluding special tokens)
VOCAB = ["<pad>", "<sos>", "<eos>"] + list(raw_data.values())
VOCAB_MAP = {VOCAB[i]: i for i in range(0, len(VOCAB))}
PAD_TOKEN = VOCAB_MAP["<pad>"]
SOS_TOKEN = VOCAB_MAP["<sos>"]
EOS_TOKEN = VOCAB_MAP["<eos>"]

print(f"Length of Vocabulary: {len(VOCAB)}")
print(f"VOCAB: {VOCAB}")
print(f"PAD_TOKEN: {PAD_TOKEN}")
print(f"SOS_TOKEN: {SOS_TOKEN}")
print(f"EOS_TOKEN: {EOS_TOKEN}")

# Extract the character set (excluding special tokens)
characters = list(raw_data.values())
print(f"Number of characters: {len(characters)}")

# Create the tokenizer
tokenizer = CharacterTokenizer(
    characters=characters,
    model_max_length=512,  # Set appropriate max length for your model
)

# Map special tokens to match your requirements
# Note: The CharacterTokenizer uses different special token names than your original code
# We'll need to map them correctly
def map_special_tokens():
    # Map [PAD] to <pad>
    tokenizer._vocab_str_to_int["<pad>"] = tokenizer._vocab_str_to_int["[PAD]"]
    tokenizer._vocab_int_to_str[tokenizer._vocab_str_to_int["<pad>"]] = "<pad>"
    
    # Map [BOS] to <sos>
    tokenizer._vocab_str_to_int["<sos>"] = tokenizer._vocab_str_to_int["[BOS]"]
    tokenizer._vocab_int_to_str[tokenizer._vocab_str_to_int["<sos>"]] = "<sos>"
    
    # Map [SEP] to <eos>
    tokenizer._vocab_str_to_int["<eos>"] = tokenizer._vocab_str_to_int["[SEP]"]
    tokenizer._vocab_int_to_str[tokenizer._vocab_str_to_int["<eos>"]] = "<eos>"
    
    # Update the tokens
    tokenizer.bos_token = "<sos>"
    tokenizer.eos_token = "<eos>"
    tokenizer.pad_token = "<pad>"

# Apply special token mapping (optional)
# Uncomment if you want to use your original token names
# map_special_tokens()

# Create directory to save the tokenizer
os.makedirs("character_tokenizer", exist_ok=True)

# Save the tokenizer
tokenizer.save_pretrained("character_tokenizer")
print(f"Tokenizer saved to 'character_tokenizer' directory")

# Test the tokenizer
def test_tokenizer(tokenizer):
    # Test text
    test_text = "Hello world!"
    
    # Tokenize
    tokens = tokenizer.tokenize(test_text)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    
    print(f"Original text: {test_text}")
    print(f"Tokens: {tokens}")
    print(f"Token IDs: {token_ids}")
    
    # Encode with special tokens
    encoded = tokenizer(text=test_text, add_special_tokens=True)
    print(f"Encoded with special tokens: {encoded['input_ids']}")
    
    # Decode back to text
    decoded = tokenizer.decode(encoded['input_ids'])
    print(f"Decoded text: {decoded}")
    
    # Batch encoding for model input
    batch_texts = ["Hello world!", "Testing"]
    batch_encoded = tokenizer(
        batch_texts,
        padding=True,
        truncation=True,
        return_tensors="pt"  # Return PyTorch tensors
    )
    
    print("\nBatch encoding:")
    for key, value in batch_encoded.items():
        print(f"{key}: {value}")

# Test the tokenizer
test_tokenizer(tokenizer)

# Example usage in models
print("\nExample usage in models:")
print("""
# Load the tokenizer
from character_tokenizer import CharacterTokenizer

# Load the saved tokenizer
tokenizer = CharacterTokenizer.from_pretrained('character_tokenizer')

# Prepare inputs for the model
text = "Hello world!"
inputs = tokenizer(text, return_tensors="pt")

# Pass to your model
# outputs = model(**inputs)

# Decode model outputs
# predicted_ids = outputs.logits.argmax(-1)
# decoded_text = tokenizer.decode(predicted_ids[0])
""")

# How to load the tokenizer for inference
print("\nTo load the tokenizer later:")
print("""
from character_tokenizer import CharacterTokenizer

tokenizer = CharacterTokenizer.from_pretrained('character_tokenizer')
""")