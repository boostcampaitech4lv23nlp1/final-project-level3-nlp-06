import torch
import random
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from data import PairedDataset, collate_fn
from transformers import (
    AutoTokenizer,
    MBartForConditionalGeneration, 
    MBart50TokenizerFast,
    is_torch_available,
)

device = "cuda" if torch.cuda.is_available() else "cpu"

data = pd.read_csv("reward_model_data_v1.0.0.csv")
text = list(data["text"])

model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50")
model.load_state_dict(torch.load("/opt/ml/save/model/bart_7epoch.pt"))
model.to(device)

tokenizer = AutoTokenizer.from_pretrained(
    "facebook/mbart-large-50", 
    src_lang="ko_KR", 
    tgt_lang="ko_KR"
)
model.eval()
store1 = []
for i in range(len(text)):
    print("epoch 7", i)
    now_data = "순화할 문장 : " + text[i] + ", 순화된 이후 문장 : " 
    inputs = tokenizer(now_data, return_tensors="pt")
    inputs.to(device)
    translated_tokens = model.generate(
        **inputs, 
        decoder_start_token_id=tokenizer.lang_code_to_id["ko_KR"], 
        early_stopping=True, 
        max_length=len(text[i])*2,
        no_repeat_ngram_size=2,
    )
    out = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    store1.append(out)

del model 

model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50")
model.load_state_dict(torch.load("/opt/ml/save/model/bart_10epoch.pt"))
model.to(device)

model.eval()

store2 = []
store3 = []
for i in range(len(text)):
    print("epoch 10", i)
    now_data = "순화할 문장 : " + text[i] + ", 순화된 이후 문장 : " 
    inputs = tokenizer(now_data, return_tensors="pt")
    inputs.to(device)
    translated_tokens = model.generate(
        **inputs, 
        decoder_start_token_id=tokenizer.lang_code_to_id["ko_KR"], 
        early_stopping=True, 
        max_length=len(text[i])*2,
        no_repeat_ngram_size=2,
        #num_return_sequences=2,
    )
    out = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    store2.append(out)
    
    translated_tokens = model.generate(
        **inputs, 
        decoder_start_token_id=tokenizer.lang_code_to_id["ko_KR"], 
        early_stopping=True, 
        max_length=len(text[i])*2,
        no_repeat_ngram_size=2,
        #num_return_sequences=2,
        top_p=0.85,
        do_sample=True,
        top_k=10,
    )
    out = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    store3.append(out)

data["label1"] = store1
data["label2"] = store2
data["label3"] = store3

data.to_csv("reward_sample.csv", index=False)