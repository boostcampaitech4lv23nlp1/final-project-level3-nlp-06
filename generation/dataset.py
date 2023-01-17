from torch.utils.data import Dataset
import pandas as pd
import torch

### Dataset ###
class ParallelDatasetForMBart(Dataset):
    def __init__(self, config, tokenizer, eval=False):
        self.config = config
        if eval:
            dataset = pd.read_csv(config["eval_data_path"])
        else:
            dataset = pd.read_csv(config["train_data_path"])
        self.tokenizer=tokenizer
        self.source = tokenizer(
            list(dataset["source"]),
            max_length=config["max_length"],
            padding="max_length",
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors="pt"
        ) 
        self.labels = tokenizer(
            list(dataset["target"]),
            max_length=config["max_length"],
            padding="max_length",
            return_attention_mask=True,
            return_token_type_ids=True,
        )

    def __len__(self):
        return len(self.source["input_ids"])

    def __getitem__(self, idx):
        return {
            "input_ids":self.source["input_ids"][idx],
            "attention_mask":self.source["attention_mask"][idx],
            "labels": torch.tensor([self.tokenizer.bos_token_id]+self.labels["input_ids"][idx]),
        }

def collate_fn(batch):
    input=[]
    attention=[]
    labels=[]
    for b in batch:
        input.append(b["input_ids"].unsqueeze(0))
        attention.append(b["attention_mask"].unsqueeze(0))
        labels.append(b["labels"].unsqueeze(0))
    return {
        "input_ids":torch.cat(input,0),
        "attention_mask":torch.cat(attention,0),
        "labels": torch.cat(labels,0)
    } # Shape: (batch_size, max length)
