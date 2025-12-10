from torch.utils.data import Dataset
import torch

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        txt = str(txt)
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Create chunks
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1:i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

        # Handle the last chunk if it's smaller than max_length
        if len(token_ids) > max_length:
            last_start = max(len(token_ids) - max_length - 1, 0)
            input_chunk = token_ids[last_start:last_start + max_length]
            target_chunk = token_ids[last_start + 1:last_start + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
        elif len(token_ids) > 1:
            input_chunk = token_ids[:-1]
            target_chunk = token_ids[1:]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index]
    
