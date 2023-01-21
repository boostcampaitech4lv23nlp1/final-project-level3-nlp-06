import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from multiprocessing import Pool


## TODO: config로부터 max_length 받기
## TODO: config로부터 label_type 받기
class Span_Dataset(Dataset):
    def __init__(self, csv_path: str, tokenizer_name: str):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = 128
        self.inputs = self.tokenizer(
            list(self.df['comment']), 
            max_length=self.max_length, 
            padding="max_length", 
            return_offsets_mapping=True, 
            return_tensors="pt"
        )
        self.label_type="token_classification"
        pool = Pool(processes=4)
        self.labels = pool.map(self.set_labels, range(len(self.df['comment'])))
        self.class_labels = [1 if off==True else 0 for off in self.df['OFF']]
        
    def __len__(self):
        return self.inputs['input_ids'].shape[0]
    
    def __getitem__(self, index):
        return {
            "input_ids": self.inputs['input_ids'][index],
            "token_type_ids": self.inputs['token_type_ids'][index],
            "attention_mask": self.inputs['attention_mask'][index],
            "labels": self.labels[index],
            "class_label": self.class_labels[index]
        }
    
    def set_labels(self, target_idx):
        start = eval(self.df['start_index'][target_idx])
        end = eval(self.df['end_index'][target_idx])

        span_ranges = []

        for s, e in zip(start, end):
            start_token_idx = -1
            end_token_idx = -1
            for i, mapping in enumerate(self.inputs['offset_mapping'][target_idx]):
                # charactor 위치가 token 길이 범위에 속한다면,
                if s in range(mapping[0].item(), mapping[1].item()):
                    start_token_idx = i
                if e in range(mapping[0].item(), mapping[1].item()+1):
                    end_token_idx = i
            span_ranges.append((start_token_idx, end_token_idx))
            
        if self.label_type == "token_classification":
            # For Token classification:
            label = torch.tensor([-100 for _ in range(self.max_length)])
            label[1:sum(self.inputs['attention_mask'][target_idx]).item()-1] = 0
            for s, e in span_ranges:
                label[s:e+1] = 1
        elif self.label_type == "index_classification":
            # For span index classification:
            start_label = [span[0] for span in span_ranges]
            end_label = [span[1] for span in span_ranges]

        return label.tolist()
    