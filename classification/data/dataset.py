from torch.utils.data import Dataset
import torch
from transformers import AutoTokenizer
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer


class BASE_Dataset(Dataset):
    def __init__(self, csv_path: str, tokenizer_name: str):
        super(BASE_Dataset, self).__init__()
        self.df = pd.read_csv(csv_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.text = None
        self.label = None
        self.tokenized_sentences = None
        
    def preprocess_dataframe(self):
        raise NotImplementedError
    
    def encoding_sentences(self):
        tokenized_sentences = self.tokenizer(
            self.text,
            max_length=128,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        return tokenized_sentences
    
    def __getitem__(self, idx):
        return {
            "input_ids": self.tokenized_sentences['input_ids'][idx],
            "token_type_ids": self.tokenized_sentences['token_type_ids'][idx],
            "attention_mask": self.tokenized_sentences['attention_mask'][idx],
            "label": self.label[idx]
        }
    
    def __len__(self):
        return len(self.text)
    

class Apeach_Dataset(BASE_Dataset):
    def __init__(self, csv_path, tokenizer_name):
        super(Apeach_Dataset, self).__init__(csv_path, tokenizer_name)
        self.text, self.label = self.preprocess_dataframe(self.df)
        self.tokenized_sentences = self.encoding_sentences()
        
    def preprocess_dataframe(self, df):
        text = list(df['text'])
        label = list(df['class'])
        
        return text, label
        
    
class kmhas_Dataset(BASE_Dataset):
    def __init__(self, csv_path, tokenizer_name):
        super(kmhas_Dataset, self).__init__(csv_path, tokenizer_name)
        self.text, self.label = self.preprocess_dataframe(self.df)
        self.tokenized_sentences = self.encoding_sentences()
        
    def preprocess_dataframe(self, df):
        text = list(df["text"])
        labels = [0 if label == "[8]" else 1 for label in df["label"]]
        return text, labels
    
    
class KOLD_Dataset(Dataset):
    def __init__(self, csv_path, tokenizer_name):
        super(KOLD_Dataset, self).__init__(csv_path, tokenizer_name)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.text, self.label = self.preprocess_dataframe(self.df)
        self.tokenized_sentences = self.encoding_sentences()
        
    def preprocess_dataframe(self, df):
        text = list(self.df['comment'])
        label = [1. if off == True else 0. for off in self.df['OFF']]
        return text, label
        