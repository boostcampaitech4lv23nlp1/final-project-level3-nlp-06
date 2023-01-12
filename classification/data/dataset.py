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
        
    
class kmhas_Dataset(BASE_Dataset):
    def __init__(self, csv_path, tokenizer_name):
        super(kmhas_Dataset, self).__init__(csv_path, tokenizer_name)
        self.enc = MultiLabelBinarizer()
        self.text, self.label = self.preprocess_dataframe(self.df)
        self.num_labels = 9
        
    def preprocess_dataframe(self, df):
        text = list(df["text"])
        label = [eval(label) for label in df['label']]
        label = self.enc.fit_transform(label)
        
        return text, label
        
    def __getitem__(self, idx):
        encoded_text = self.tokenizer.encode_plus(
            text=self.text[idx],
            return_tensors='pt',
            max_length=256,
            truncation=True,
            padding="max_length",
        )
        encoded_text = {k: v.squeeze() for k, v in encoded_text.items()}
        encoded_text["label"] = self.label[idx]
        
        return encoded_text
    
    
class KOLD_Dataset(Dataset):
    def __init__(self, csv_path, tokenizer_name="monologg/koelectra-base-v3-discriminator"):
        super(KOLD_Dataset, self).__init__(csv_path, tokenizer_name)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.sentences = list(self.df['comment'])
        self.label = [1. if off == True else 0. for off in self.df['OFF']]
        self.tokenized_sentences = self.tokenizer(
            self.sentences,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        )
        
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        return {
            "input_ids": self.tokenized_sentences['input_ids'][idx],
            "token_type_ids": self.tokenized_sentences['token_type_ids'][idx],
            "attention_mask": self.tokenized_sentences['attention_mask'][idx],
            "label": self.labels[idx]
        }
        