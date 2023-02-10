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

data = pd.read_csv("paired_data_v0.0.1.csv")

model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50")
model.load_state_dict(torch.load("/opt/ml/save/model/model-40.pt"))
model.to(device)

tokenizer = AutoTokenizer.from_pretrained(
    "facebook/mbart-large-50", 
    src_lang="ko_KR", 
    tgt_lang="ko_KR"
)
model.eval()

'''''
text = list(data["text"])[30:50]
for i in range(len(text)):
    now_data = "순화할 문장 : " + text[i] + ", 순화된 이후 문장 : " 
    inputs = tokenizer(now_data, return_tensors="pt")
    inputs.to(device)
    translated_tokens = model.generate(**inputs, decoder_start_token_id=tokenizer.lang_code_to_id["ko_KR"], early_stopping=True, max_length=len(text[i])*2)
    out = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    #print("-------------")
    #print(text[i])
    #print(out)
'''''

temp = "당신은 쓰레기고 재수없고 역겹다. 얼굴도 좆같이 생겨서 왜 저러냐"
now_data = "순화할 문장 : " + temp + ", 순화된 이후 문장 : "
inputs = tokenizer(now_data, return_tensors="pt")
inputs.to(device)
translated_tokens = model.generate(**inputs, decoder_start_token_id=tokenizer.lang_code_to_id["ko_KR"], early_stopping=True, max_length=len(temp) * 2)
out = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
print(out)

