from .tokenization_kocharelectra import KoCharElectraTokenizer
from torch.utils.data import Dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

import pandas as pd
import torch
import random
from tqdm import tqdm


class MLMDataset(Dataset):
    def __init__(self, df_path, tokenizer_name):
        super(MLMDataset, self).__init__()
        self.df = pd.read_csv(df_path)
        
        if tokenizer_name == "monologg/kocharelectra-base-discriminator":
            self.tokenizer = KoCharElectraTokenizer.from_pretrained(tokenizer_name)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        self.text = list(self.df["comment"])
        self.tokenized_sentences = self.tokenizer(
            self.text,
            max_length=128,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        self.label = []
        
        print("masking...")
        for i, tokenized_sentence in enumerate(tqdm(self.tokenized_sentences['input_ids'])):
            sentence, label = self.masking(tokenized_sentence.int(), self.tokenized_sentences['attention_mask'][i])
            self.tokenized_sentences['input_ids'][i] = sentence
            self.label.append(label)
        
        
    def masking(self, sentence, attention_mask):
        label = [0 for _ in range(len(sentence))]
        length = 0
        for mask in attention_mask:
            if mask == 0:
                break
            length += 1
        for i in range(1, length-1):
            masking = random.random()
            if masking < 0.15:
                label[i] = 1
                masking /= 0.15
                if masking < 0.8:
                    sentence[i] = self.tokenizer.mask_token_id
                elif masking < 0.9:
                    sentence[i] = random.randrange(self.tokenizer.vocab_size)
                else:
                    sentence[i] = sentence[i]
        sentence = torch.IntTensor(sentence)
        label = torch.IntTensor(label)

        return sentence, label
        
    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, idx):
        return {
            "input_ids": self.tokenized_sentences['input_ids'][idx],
            "token_type_ids": self.tokenized_sentences['token_type_ids'][idx],
            "attention_mask": self.tokenized_sentences['attention_mask'][idx],
            "label": self.label[idx]
        }
    