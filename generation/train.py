import argparse
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    MBartForConditionalGeneration,
    is_torch_available
)
sys.path.append("..")
from tqdm import tqdm
import wandb
import yaml
from reward_util import cal_bl_loss, cal_sc_loss
from dataset import ParallelDatasetForMBart, collate_fn
from classification.model.cnn_model import CNNModel
import random

def set_seed(seed: int = 6):
    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if use multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

### Main ###
def main(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load BART
    model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50")
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-50", src_lang="ko_KR", tgt_lang="ko_KR")

    # Load style classifier
    sc_tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    classifier = CNNModel(config, sc_tokenizer.vocab_size)
    classifier.to(device)
    classifier.load_state_dict(torch.load(config["classifier_path"]))
    classifier.eval()

    wandb.init(
        config=config,
        project=config["wandb_project"],
        entity=config["wandb_entity"],
        group=config["wandb_group"],
        name=config["wandb_name"],
    )

    train_dataset = ParallelDatasetForMBart(config, tokenizer)
    eval_dataset = ParallelDatasetForMBart(config, tokenizer, eval=True)

    train_loader = DataLoader(
        train_dataset, 
        collate_fn=collate_fn, 
        shuffle=True,
        batch_size=config["batch_size"],
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        collate_fn=collate_fn,
        batch_size=config["batch_size"],
    )

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    epoch = config["epoch"]

    for e in range(epoch):
        total_loss = []
        total_loss_ce = []
        total_loss_sc = []
        total_loss_co = []

        model.train()
        for i, batch in enumerate(tqdm(train_loader)):
            src = batch["input_ids"].to(device)
            tgt = batch["labels"].to(device)
            logits = model(
                input_ids=src,
                attention_mask=batch["attention_mask"].to(device),
                decoder_input_ids=tgt,
            )[0]
            
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = tgt[..., 1:].contiguous()
            loss_ce = loss_fn(shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1))
            total_loss_ce.append(loss_ce.item())

            # Get additional losses
            loss_sc = torch.tensor(0)
            idx = tgt.ne(tokenizer.pad_token_id).sum(-1) # Find the end of a sentence
            loss_sc = cal_sc_loss(device, logits, idx, classifier, tokenizer, sc_tokenizer, 0)
            total_loss_sc.append(loss_sc.item())
            loss_co = torch.tensor(0)
            idx = tgt.ne(tokenizer.pad_token_id).sum(-1)
            loss_co = cal_bl_loss(device, logits, tgt, idx, tokenizer)
            total_loss_co.append(loss_co.item())
            loss = loss_ce + loss_sc + loss_co
            total_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i%100 == 0:
                wandb.log({
                    "loss": sum(total_loss)/len(total_loss),
                    "ce_loss":sum(total_loss_ce)/len(total_loss_ce),
                    "sc_loss":sum(total_loss_sc)/len(total_loss_sc),
                    "bl_loss":sum(total_loss_co)/len(total_loss_co)
                })
                total_loss_ce=[]
                total_loss_sc=[]
                total_loss_co=[]
        
        model.eval()
        val_loss=[]
        for i,batch in enumerate(tqdm(eval_loader)):
            src = batch["input_ids"].to(device)
            tgt = batch["labels"].to(device)
            with torch.no_grad():
                logits = model(src, attention_mask=batch["attention_mask"].to(device), decoder_input_ids=tgt)[0]
            
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = tgt[..., 1:].contiguous()
            loss_ce = loss_fn(shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1))
            val_loss.append(loss_ce.detach().item())
            
        wandb.log({"eval_ce_loss":sum(val_loss)/len(val_loss)})
        val_loss=[]

    torch.save(model.state_dict(), config["bart_path"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument("--conf", type=str, default="config.yaml", help="Config file path (.yaml)")
    args = parser.parse_args()
    with open(args.conf, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)
    set_seed(6)
    main(config)