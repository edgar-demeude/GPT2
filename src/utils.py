import torch
import tiktoken
from torch.utils.data import DataLoader
import sys
import os
import torch.nn.functional as F

sys.path.append(os.path.abspath("../src"))
from GPTDatasetV1 import GPTDatasetV1


# ------ DATA PROCESSING ------

def create_data_from_txt(file, split_ratio=0.9):
    with open(file, 'r', encoding='utf-8') as f:
        text = f.read()

    total_length = len(text)

    split_idx = int(len(text) * split_ratio)
    train_text = text[:split_idx]
    val_text = text[split_idx:]

    train_length = len(train_text)
    val_text = len(val_text)

    print(f"--- Dataset loaded. ---")
    print(f"Total characters: {total_length}")
    print(f"Training characters: {train_length}")
    print(f"Validation characters: {val_text}")

    return train_text, val_text


def create_dataloader_v1(txt, batch_size=4, max_length=1024, stride=1024, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dataloader


# ------ TOKENIZATION ------

def text_to_token_ids(text, tokenizer=None, add_batch_dim=True):
    """
    Encode un texte en IDs de tokens et ajoute une dimension batch si nécessaire.

    Args:
        text (str): Texte à encoder.
        tokenizer: Tokenizer à utiliser (par défaut, GPT-2).
        add_batch_dim (bool): Si True, ajoute une dimension batch (forme (1, seq_len)).

    Returns:
        torch.Tensor: Tenseur des IDs de tokens.
    """
    if tokenizer is None:
        tokenizer = tiktoken.get_encoding("gpt2")
    tokens = tokenizer.encode(text)
    tensor = torch.tensor(tokens, dtype=torch.long)
    return tensor.unsqueeze(0) if add_batch_dim else tensor


def token_ids_to_text(token_ids, tokenizer=None):
    """
    Décode des IDs de tokens en texte.

    Args:
        token_ids (torch.Tensor): Tenseur des IDs de tokens (forme (seq_len,) ou (batch_size, seq_len)).
        tokenizer: Tokenizer à utiliser (par défaut, GPT-2).

    Returns:
        str: Texte décodé.
    """
    if tokenizer is None:
        tokenizer = tiktoken.get_encoding("gpt2")
    # Gestion des batches : si batch_size > 1, on ne prend que la première séquence
    if token_ids.dim() == 2:
        token_ids = token_ids[0]
    return tokenizer.decode(token_ids.tolist())


# ------ LOSS FUNCTIONS ------

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)

    logits = model(input_batch) # [B, T, V]

    flattened_logits = logits.view(-1, logits.size(-1))  # [6, 50257]
    flattened_targets = target_batch.view(-1).long()  # [6]

    loss = F.cross_entropy(flattened_logits, flattened_targets)

    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    model.eval()
    total_loss = 0.0
    count_batches = 0

    with torch.no_grad():
        for batch_idx, (input_batch, target_batch) in enumerate(data_loader):
            if num_batches is not None and batch_idx >= num_batches:
                break

            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
            count_batches += 1

    if count_batches == 0:
        return 0.0

    avg_loss = total_loss / count_batches
    return avg_loss


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


# ------ GENERATION ------

def generate_text_simple(model, batch, n_tokens, max_context):
    model.eval()
    batch_size, seq_len = batch.shape
    generated_tokens = torch.zeros((batch_size, n_tokens), dtype=torch.long, device=batch.device)

    current_context = batch.clone()

    for i in range(n_tokens):
        with torch.no_grad():
            logits = model(current_context) # (batch_size, seq_len, vocab_size)

        last_logits = logits[:, -1, :] # (batch_size, vocab_size)

        probs = F.softmax(last_logits, dim=-1) # (batch_size, vocab_size)

        next_token = torch.argmax(probs, dim=-1) # (batch_size,)

        generated_tokens[:, i] = next_token

        next_token = next_token.unsqueeze(-1)  # adds a dimension for concat (batch_size, 1)

        current_context = torch.cat([current_context, next_token], dim=1)
        if current_context.shape[1] > max_context:
            current_context = current_context[:, -max_context:]

    return generated_tokens


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_embedding_layer.weight.shape[0]

    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model,
            batch=encoded,
            n_tokens=50,
            max_context=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(start_context, decoded_text.replace("\n", " "))
    model.train()


# ------ TRAINING ------

def train_model_simple(model, train_loader, val_loader,
                       optimizer, device, num_epochs,
                       eval_freq, eval_iter,
                       start_context, tokenizer):

    model.to(device)

    train_losses = []
    val_losses = []
    tokens_seen = []

    tokens_per_batch = train_loader.batch_size * train_loader.dataset[0][0].shape[0]

    total_tokens = 0

    for epoch in range(num_epochs):
        model.train()
        for step, (input_batch, target_batch) in enumerate(train_loader):
            # 1) Forward + loss
            loss = calc_loss_batch(input_batch, target_batch, model, device)

            # 2) Backward
            optimizer.zero_grad()
            loss.backward()

            # 3) Optimizer step
            optimizer.step()

            total_tokens += tokens_per_batch

            # 4) Évaluation périodique
            if (step + 1) % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                print(
                    f"Epoch {epoch+1}/{num_epochs}, "
                    f"Step {step+1}, "
                    f"Train loss: {train_loss:.4f}, "
                    f"Val loss: {val_loss:.4f}"
                )

                # 5) Générer un exemple de texte
                generate_and_print_sample(
                    model, tokenizer, device, start_context
                )

                train_losses.append(train_loss)
                val_losses.append(val_loss)
                tokens_seen.append(total_tokens)


    return model, train_losses, val_losses, tokens_seen
