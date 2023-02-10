from torch.utils.data import Dataset
import pandas as pd
import torch

class PairedDataset(Dataset):
    def __init__(self, config, tokenizer, eval=False):
        self.config = config
        self.tokenizer = tokenizer
        
        if eval:
            self.data = pd.read.csv(config["val_data_path"])
        else:
            self.data = pd.read_csv(config["train_data_path"])
        self.texts = list(self.data["text"])
        self.targets = list(self.data["target"])
        
        self.input = []
        self.output = []
        self.preprocess()
    
    def preprocess(self):
        
        ## Make prompts
        store = []
        if self.config["input_type"] == 0:
            for i in range(len(self.texts)):
                now_data = "순화할 문장 : " + self.texts[i] + " 순화된 이후 문장 : " 
                store.append(now_data)
        ## This for one shot
        else:
            pass
        
        ## Tokenizing
        self.input = self.tokenizer(
            store,
            max_length=self.config["max_length"],
            padding="max_length",
            return_attention_mask=True,
            return_token_type_ids=True,
            truncation=True,
            return_tensors="pt",
        )
        
        self.output = self.tokenizer(
            self.targets,
            max_length=self.config["max_length"],
            padding="max_length",
            return_attention_mask=True,
            return_token_type_ids=True,
            truncation=True,
            return_tensors="pt",
        )
        
    def __getitem__(self, idx):
        return {
            "input_ids": self.input["input_ids"][idx],
            "attention_mask": self.input["attention_mask"][idx],
            "label_ids": self.output["input_ids"][idx],
        }
    
    def __len__(self):
        return len(self.data)

def collate_fn(batch):
    input_ids = []
    attention_mask = []
    label_ids = []
    
    for b in batch:
        input_ids.append(b["input_ids"].unsqueeze(0))
        attention_mask.append(b["attention_mask"].unsqueeze(0))
        label_ids.append(b["label_ids"].unsqueeze(0))
        
    return {
        "input_ids":torch.cat(input_ids, 0),
        "attention_mask":torch.cat(attention_mask, 0),
        "label_ids": torch.cat(label_ids, 0)
    }
    
