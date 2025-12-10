import numpy as np
import torch
from importlib.metadata import version
import tiktoken
import sys
import os

sys.path.append(os.path.abspath("../src"))
from utils import create_data_from_txt, create_dataloader_v1, train_model_simple, text_to_token_ids, token_ids_to_text, generate_text_simple
from GPTModel import GPTModel
from visualization import plot_train_val_loss


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("Device : ", device)

    train_text, val_text = create_data_from_txt("./data/the-verdict.txt", split_ratio=0.9)

    train_dataloader = create_dataloader_v1(
        train_text,
        batch_size=2,
        max_length=256,
        stride=256,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    val_dataloader = create_dataloader_v1(
        val_text,
        batch_size=2,
        max_length=256,
        stride=256,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    total_train_tokens = sum(len(sample[0]) for sample in train_dataloader.dataset)
    total_val_tokens = sum(len(sample[0]) for sample in val_dataloader.dataset)

    print(f"--- DataLoader created. ---")
    print(f"Number of batches in train_dataloader: {len(train_dataloader)}")
    print(f"Total number of tokens in train_dataloader: {total_train_tokens}")
    print(f"Number of batches in val_dataloader: {len(val_dataloader)}")
    print(f"Total number of tokens in val_dataloader: {total_val_tokens}")

    tokenizer = tiktoken.get_encoding("gpt2")
    print(f"--- Tokenizer loaded. ---")

    GPT_CONFIG_124M = {
        "vocab_size": 50257,    # Vocabulary size (+1 for special token)
        "context_length": 256, # Context length (reduced to avoid computation load)
        "emb_dim": 768,         # Embedding dimension
        "n_heads": 12,          # Number of attention heads
        "n_layers": 12,         # Number of layers
        "drop_rate": 0.1,       # Dropout rate (common for GPT and later iterations)
        "qkv_bias": False       # Query-Key-Value bias
    }

    model = GPTModel(GPT_CONFIG_124M)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
    num_epochs = 10
    eval_freq = 5
    eval_iter = 5
    start_context = "Every effort moves you"

    print(f"--- Start of training. ---")

    model, train_losses, val_losses, tokens_seen = train_model_simple(model, 
                                                                      train_dataloader, 
                                                                      val_dataloader, 
                                                                      optimizer, 
                                                                      device, 
                                                                      num_epochs, 
                                                                      eval_freq, 
                                                                      eval_iter, 
                                                                      start_context, 
                                                                      tokenizer
                                                                    )
    
    steps_per_epoch = len(train_dataloader) // eval_freq
    epochs_for_points = np.arange(len(train_losses)) / steps_per_epoch
    plot_train_val_loss(epochs_for_points, train_losses, val_losses)
    print(f'Loss plot saved to  "./results/loss_plot.png"')
    print(f"--- End of training. ---")

    input = "My name is "
    print("Input: ", input)

    batch = text_to_token_ids(input)
    batch = batch.to(device)

    generated_tokens = generate_text_simple(model, batch, 10, 256)

    generated_text = token_ids_to_text(generated_tokens)
    print("Generated: ", generated_text)
