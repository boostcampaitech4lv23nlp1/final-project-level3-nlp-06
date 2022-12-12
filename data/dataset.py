from torch.utils.data import Dataset
import torch
from transformers import AutoTokenizer
import pandas as pd


class BASE_Dataset(Dataset):
    def __init__(self, csv_path: str, tokenizer_name: str):
        super(BASE_Dataset, self).__init__()
        self.df = pd.read_csv(csv_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.text = None
        self.label = None
        self.label2num = None
        
        
    def preprocess_dataframe(self):
        raise NotImplementedError
        
        
    def set_label2num(self, label):
        labels = list(set(label))
        label2num = {label: i for i, label in enumerate(labels)}
        print("label mapped to :", label2num)
        
        return label2num
    
    
    def __getitem__(self, idx):
        encoded_text = self.tokenizer.encode_plus(
            text=self.text[idx],
            return_tensors='pt',
            max_length=256,
            truncation=True,
            padding="max_length",
        )
        encoded_text = {k: v.squeeze() for k, v in encoded_text.items()}
        label = self.label2num[self.label[idx]]
        encoded_text["label"] = torch.tensor(label)
        
        return encoded_text
    
    
    def __len__(self):
        return len(self.text)
    

class Apeach_Dataset(BASE_Dataset):
    def __init__(self, csv_path, tokenizer_name):
        super(Apeach_Dataset, self).__init__(csv_path, tokenizer_name)
        self.text, self.label = self.preprocess_dataframe(self.df)
        self.label2num = self.set_label2num(self.label)
        self.num_labels = len(self.label2num)
        
        
    def preprocess_dataframe(self, df):
        text = list(df['text'])
        label = list(df['class'])
        
        return text, label
        
    