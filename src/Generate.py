import torch
import torch.nn.functional as F


def top_k_logits(logits, k):
    # Ensure k is not larger than the vocabulary size
    k = min(k, logits.size(-1))
    if k <= 0:
        return logits
    # Find the indices of the top k logits
    top_k_values, top_k_indices = torch.topk(logits, k, dim=-1)
    # Create a mask for the top k logits
    mask = torch.full_like(logits, float('-inf'))
    # Use scatter_ to set the mask values for the top-k indices
    mask.scatter_(-1, top_k_indices, 0)
    # Apply the mask
    return logits + mask


def generate(model, idx, max_new_tokens, context_size, temperature=1.0, top_k=None, eos_id=None):
    """
    Generate new tokens autoregressively using the model.
    
    Args:
        model: The GPT model to use for generation
        idx: Initial token indices, shape (batch_size, seq_len)
        max_new_tokens: Maximum number of tokens to generate
        context_size: Maximum context length (older tokens are truncated)
        temperature: Sampling temperature (0.0 = greedy, higher = more random)
        top_k: If set, only sample from top k most likely tokens
        eos_id: End-of-sequence token id (stops generation if encountered)
    
    Returns:
        generated_tokens: Tensor of shape (batch_size, max_new_tokens)
    """
    model.eval()
    batch_size, seq_len = idx.shape
    generated_tokens = torch.zeros((batch_size, max_new_tokens), dtype=torch.long, device=idx.device)
    current_context = idx.clone()
    
    for i in range(max_new_tokens):
        with torch.no_grad():
            # Get logits from model for current context
            logits = model(current_context)  # (batch_size, seq_len, vocab_size)
        
        # Only use logits from last position
        last_logits = logits[:, -1, :]  # (batch_size, vocab_size)
        
        # Apply top-k filtering if specified
        if top_k is not None:
            last_logits = top_k_logits(last_logits, top_k)
        
        # Sample next token based on temperature
        if temperature == 0.0:
            # Greedy decoding: always pick most likely token
            next_token = torch.argmax(last_logits, dim=-1)  # (batch_size,)
        else:
            # Stochastic sampling: sample according to probability distribution
            probs = F.softmax(last_logits / temperature, dim=-1)  # (batch_size, vocab_size)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)  # (batch_size,)
        
        # Store generated token
        generated_tokens[:, i] = next_token
        
        # Check for end-of-sequence token
        if eos_id is not None and (next_token == eos_id).any():
            break
        
        # Append new token to context
        next_token = next_token.unsqueeze(-1)  # (batch_size, 1)
        current_context = torch.cat([current_context, next_token], dim=1)
        
        # Truncate context if it exceeds maximum length
        if current_context.shape[1] > context_size:
            current_context = current_context[:, -context_size:]
    
    return generated_tokens