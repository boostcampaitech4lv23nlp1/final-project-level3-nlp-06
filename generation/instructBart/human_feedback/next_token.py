import yaml
import torch
import wandb
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

def get_config():
    with open("arg.yaml", "r") as f:
        args = yaml.load(f, Loader=yaml.Loader)
    return args

if __name__ == "__main__":
    ## Setting
    set_seed(6)
    torch.cuda.empty_cache()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = get_config()
    
    '''''
    wandb.init(
        project=config["project"],
        entity=config["entity"],
        name=config["name"],
    )
    '''''
    
    ## Model
    model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50")
    model.to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(
        "facebook/mbart-large-50", 
        src_lang="ko_KR", 
        tgt_lang="ko_KR"
    )
    
    ## Preprocessing
    train_dataset = PairedDataset(config, tokenizer)
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=config["batch_size"]
    )
    
    ## Preprocessing
    train_data = pd.read_csv(config["train_data_path"])
    
    ## Train setting
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    step_count = 0
    
    model.train()
    total_loss = []
    
    texts = list(train_data["text"])
    targets = list(train_data["target"])
    
    for e in range(1):#config["epoch"]):
        for i in tqdm(range(len(train_data))):
            now_text = "순화할 문장 : " + texts[i] + " 순화된 이후 문장 : "
            target = targets[i]
            
            target_token = tokenizer(
                target,
                return_attention_mask=True,
                return_token_type_ids=True,
                return_tensors="pt",
            )

            print(target_token["input_ids"])
            print(tokenizer.decode(250014))
            print(tokenizer.decode(8000))
            print(tokenizer.decode(20463))
            break
        break
    
    '''''
    ## Train
    model.train()
    total_loss = []
    for e in range(config["epoch"]):
        
        for i, item in enumerate(tqdm(train_dataloader)):
            input_ids = item["input_ids"].to(device)
            attention_mask = item["attention_mask"].to(device)
            label_ids = item["label_ids"].to(device)
            
            output = model(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                decoder_input_ids=label_ids,
            )
            
            logits = output["logits"]
            
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = label_ids[..., 1:].contiguous()
            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss.append(loss.detach())
            
            ## Train loss
            step_count += 1
            if not step_count % 20:
                wandb.log({
                    "train_loss": sum(total_loss) / len(total_loss)
                })
                total_loss = []

        ## TODO: Here we need evaluation
        ##
        ## 
        
    torch.save(model.state_dict(), config["save_path"])
    print("---------End--------")
    '''''
    
## facebook/mbart-large-50

## 어떤 모델이 RLHF의 initial 모델로 사용할 수 있는지에 대한 논의는 아직 완벽하게 이뤄지지 않았다.
## 가령 예를 들어, 그냥 initial model이 될 수도 있고, 
## pre-trained된 model 혹은 fine-tuning된 모델이 될 수도 있다.

## 1. Prompts + Text Dataset + Human Augmented text + train initial model

## 사람의 score를 직접 반영하기에는 uncalibrated & noisy하기 때문이다.

## ranking 방법이 여러가지 존재한다. 두 모델이 있을 때, 두 모델에서 생성한 text를 비교 분석하는 방식이 있을 수 있다.
## loss 값이 scalar reward가 되서 모델에 적용된다??
## 

## 여기서 어떻게 설정할지 :
## 1. agent = LLM
## 2. env = chat with us
## 3. policy = strategy
## 4. action = next token prediction
## 5. reward = how good the answer was?

## 어떻게 두 text에 대한 reward를 적절히 줘야 하는지가 문제
## Stable-Baselines3 (an RL library) to solve the LunarLander-v2 environment.
## use gradient to update the policy
## n generated text with different model
## reward model을 어떤걸로 줘야 하냐.... text - text 유사도 느낌이랑 비슷하게 가야하나...?


## ------------------------------------------------------------------
## reward 모델은 대략 2000~3000개의 데이터가 필요할 것으로 예상된다.
## human scoring 필수


## 1. Fine-tuning with simple method
## 1-1. Fine-tuning with only prompt
## 1-2. Fine-tuning with few-shot

## 2. Reward checking with model generated with different seed or different model
## 2-1. train reward model (regression)

## 3. PPO training
## 3-1. how to train




