import tiktoken
import torch

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