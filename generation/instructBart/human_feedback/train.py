import yaml
import torch
import wandb
import random
import string
import numpy as np
import pandas as pd
import torch.nn as nn

from tqdm import tqdm
from collections import deque
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    MBartForConditionalGeneration, 
    MBart50TokenizerFast,
    is_torch_available,
)


class Config():
    ## setting
    epoch = 1
    
    ## hyperparameter
    lr = 6e-5
    
    ## path
    train_data_path = "lr_model_data_v1.0.0.csv"
    save_path = "/opt/ml/save/model/model-4"
    
    ## wandb
    project = "instructBART"
    entity = "koohack"
    name = "temp"
    
    
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        

def encode(tokenizer, text_sentence, add_special_tokens=True):
    text_sentence = text_sentence.replace('<mask>', tokenizer.mask_token)
    # if <mask> is the last token, append a "." so that models dont predict punctuation.
    if tokenizer.mask_token == text_sentence.split()[-1]:
        text_sentence += ' .'

    input_ids = torch.tensor([tokenizer.encode(text_sentence, add_special_tokens=add_special_tokens)])
    mask_idx = torch.where(input_ids == tokenizer.mask_token_id)[1].tolist()[0]
    return input_ids, mask_idx

def decode(tokenizer, pred_idx, top_clean):
    ignore_tokens = string.punctuation + '[PAD]'
    tokens = []
    for w in pred_idx:
        token = ''.join(tokenizer.decode(w).split())
        if token not in ignore_tokens:
            tokens.append(token.replace('##', ''))
    return '\n'.join(tokens[:top_clean])


def main():
    ## Setting
    torch.cuda.empty_cache()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = Config()
    
    wandb.init(
        project=config.project,
        entity=config.entity,
        name=config.name,
    )
    
    ## Compare model
    compare_model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50")
    compare_model.load_state_dict(torch.load("/opt/ml/save/model/bart_7epoch.pt"))
    compare_model.to(device)
    
    ## update model
    update_model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50")
    update_model.load_state_dict(torch.load("/opt/ml/save/model/bart_10epoch.pt"))
    update_model.to(device)
    
    ## reward model
    reward_model = AutoModelForSequenceClassification.from_pretrained("happy06/kcelectra-base-v2022-reward-regression")
    reward_model.to(device)
    reward_model.eval()
    reward_tokenizer = AutoTokenizer.from_pretrained("beomi/KcELECTRA-base")
    
    ## tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "facebook/mbart-large-50", 
        src_lang="ko_KR", 
        tgt_lang="ko_KR"
    )
    
    ## Get Data
    dataset = pd.read_csv(config.train_data_path)

    ## Train
    train(compare_model, update_model, tokenizer, dataset, reward_model, reward_tokenizer, config)

    return

def train(compare_model, update_model, tokenizer, dataset, reward_model, reward_tokenizer, config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    texts = dataset["text"]
    targets = dataset["target"]
    
    optimizer = torch.optim.Adam(update_model.parameters(), lr=config.lr)
    
    episode = 0
    
    loss_fn = torch.nn.KLDivLoss(reduction='batchmean')
    
    for e in range(config.epoch):
        for i in tqdm(range(len(texts))):
           
            episode += 1
            
            now_text = "순화할 문장 : " + texts[i] + " 순화된 이후 문장 : "
            
            tokenized = tokenizer(
                now_text,
                truncation=True,
                return_tensors="pt",
            )
            tokenized.to(device)
            input_ids = tokenized.input_ids
            
            ## logics
            with torch.no_grad():
                compare_logics = compare_model(input_ids)
            update_logics = update_model(input_ids)
            
            update_logics = update_logics["logits"][..., :-1, :]
            compare_logics = compare_logics["logits"][..., :-1, :]
            
            ## get generated text
            translated_tokens = update_model.generate(**tokenized, decoder_start_token_id=tokenizer.lang_code_to_id["ko_KR"], early_stopping=True, max_length=len(texts[i])*2)
            out = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
            
            ## reward
            reward_text = texts[i] + "[SEP]" + out
            reward_token = reward_tokenizer(
                texts[i],
                out,
                truncation=True,
                return_tensors="pt",
            )
            reward_token.to(device)
            
            reward = reward_model(**reward_token)["logits"][0].detach()
            reward = (reward - 1.5) / 1.5
            
            klloss = loss_fn(update_logics, compare_logics)
            
            loss = reward - 0.00001 * klloss
            
            ## give old model new policy, new model update the policy
            compare_model.load_state_dict(update_model.state_dict())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            wandb.log({
                "train_loss": loss.detach()
            })
            
        torch.save(update_model.state_dict(), config.save_path+str(e)+".pt")
    
if __name__ == "__main__":
    set_seed(6)
    main()