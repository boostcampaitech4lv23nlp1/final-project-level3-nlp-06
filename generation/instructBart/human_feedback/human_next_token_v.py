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
    epoch = 5
    
    ## hyperparameter
    lr = 5e-5
    
    ## path
    train_data_path = "lr_model_data_v1.0.0.csv"
    


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
    
    ## Compare model
    compare_model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50")
    compare_model.load_state_dict(torch.load("/opt/ml/save/model/bart_7epoch.pt"))
    compare_model.to(device)
    
    ## update model
    update_model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50")
    update_model.load_state_dict(torch.load("/opt/ml/save/model/bart_10epoch.pt"))
    update_model.to(device)
    
    ## reward model
    #reward_model = AutoModelForSequenceClassification.from_pretrained("happy06/kcelectra-base-v2022-reward-regression")
    reward_model = [0, 1, 2, 3, 4]
    
    ## tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "facebook/mbart-large-50", 
        src_lang="ko_KR", 
        tgt_lang="ko_KR"
    )
    
    ## Get Data
    dataset = pd.read_csv(config.train_data_path)
    
    '''''
    text = "순화할 문장 : 시발 동명이형 얼굴이 좆같아요. 순화된 이후 문장 : 당신의 <mask>"
    
    input_ids, mask_idx = encode(tokenizer, text, add_special_tokens=True)
    input_ids = input_ids.to(device)
    
    predict = update_model(input_ids)
    
    print(predict[0][0, mask_idx, :])
    print(predict[0][0, mask_idx, :].topk(5))
    pred = decode(tokenizer, predict[0][0, mask_idx, :].topk(5).indices.tolist(), 5)
    '''''
    train(compare_model, update_model, tokenizer, dataset, config)

    return

 
def train(compare_model, update_model, tokenizer, dataset, config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    texts = dataset["text"]
    targets = dataset["target"]
    
    optimizer = torch.optim.Adam(update_model.parameters(), lr=config.lr)
    
    episode = 0
    
    loss_fn = torch.nn.KLDivLoss(reduction='batchmean')
    
    for e in range(config.epoch):
        for i in range(len(texts)):
           
            episode += 1
            
            now_text = "순화할 문장 : " + texts[i] + " 순화된 이후 문장 : "
            
            memory_old = []
            memory_new = []
            memory_action = []
            length_count = 0
            loss = 0
            
            ## one episode start
            while True:
                #now_text += " <mask>"
                token = tokenizer(
                    now_text,
                    return_attention_mask=True,
                    return_token_type_ids=True,
                    return_tensors="pt",
                )
                #input_ids, mask_idx = encode(tokenizer, now_text, add_special_tokens=True)
                #input_ids = input_ids.to(device)
                print(token)
                break
                ## logics
                '''''
                with torch.no_grad():
                    compare_logics = compare_model(input_ids)
                update_logics = update_model(input_ids)
                print(update_logics)
                print(update_logics.keys())
                break
                '''''
                '''''
                ## token logics
                old_logic = compare_logics[0][0, mask_idx, :]
                new_logic = update_logics[0][0, mask_idx, :]
                
                actions = update_logics[0][0, mask_idx, :].topk(5).indices.tolist()
                values = update_logics[0][0, mask_idx, :].topk(5).values.tolist()
                
                ## select the action
                action = 2
                for i, w in enumerate(actions):
                    if w == 3:
                        continue
                    elif w == 5 and length_count < 4:
                        continue
                    elif w == 2 and values[i] < 50:
                        continue
                    else:
                        action = w
                        break
                
                ## store infomations
                memory_old.append(old_logic)
                memory_new.append(new_logic)
                memory_action.append(action)
                
                ## decode current action & replace mask token
                ignore_tokens = string.punctuation + '[PAD]'
                next_word = ''.join(tokenizer.decode(action).split())
                if next_word not in ignore_tokens:
                    next_word = next_word.replace("##", "")
                
                if next_word == "</s>":
                    break
                now_text = now_text.replace("<mask>", next_word)
                   
                length_count += 1
                print(now_text)
                '''''
            ## get reward
            text1 = texts[i]
            text2 = "te"
            
            ## TODO: reward setting
            #reward = reward_model(inputs)
            reward = 1
            
            ## give old model new policy, new model update the policy
            compare_model.load_state_dict(update_model.state_dict())
            
            klloss = 0
            for i in range(len(memory_new)):
                klloss += loss_fn(memory_new[i], memory_old[i])
            klloss = klloss / len(memory_new)
            
            loss =  reward - 0.001 * klloss 
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            break
        break
                
        
    return
    
if __name__ == "__main__":
    set_seed(6)
    main()