from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import torch.nn as nn
import torch
from data import MLMDataset
from model import ELECTRA
import argparse
import wandb
import yaml


def main(config):
    train_dataset = MLMDataset(config["train_path"], config["discriminator"])
    valid_dataset = MLMDataset(config["valid_path"], config["discriminator"])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32)
    model = ELECTRA(config["generator"], config["discriminator"], train_dataset.tokenizer.mask_token_id)
    
    criterion = nn.BCELoss()
    
    optimizer = AdamW(model.parameters(), lr=5e-5)
    
    model.to('cuda')
    
    wandb.init(
        entity=config["wandb_entity"],
        name=config["wandb_name"],
    )
    
    for epoch in range(5):
        model.train()
        for data in tqdm(train_loader):
            data = {k: v.to('cuda') for k, v in data.items()}
            label = data.pop('label')
            logits = model(data)
            loss = criterion(logits, label.float())
            wandb.log({"train loss": loss})
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        valid_loss = 0
        model.eval()
        for data in valid_loader:
            data = {k: v.to('cuda') for k, v in data.items()}
            label = data.pop('label')
            with torch.no_grad():
                logits = model(data)
            loss = criterion(logits, label.float())
            valid_loss += loss
        valid_loss /= len(valid_loader)
        wandb.log({"valid loss": valid_loss})
        save_path = config["save_path"] + "_" + str(epoch) + ".pt"
        torch.save(model.state_dict(), save_path)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hate Speech Classification')
    parser.add_argument("--conf", type=str, default="config.yaml", help="config file path(.yaml)")
    args = parser.parse_args()
    with open(args.conf, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)
    main(config)
    