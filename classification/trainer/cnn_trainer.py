from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn as nn
import torch
from sklearn.metrics import f1_score

from tqdm import tqdm
import wandb
import os

## TODO: CNN Trainer도 huggingface trainer로 합치기.
class CNNTrainer:
    def __init__(self, config, model, train_dataset, valid_dataset) -> None:
        self.config = config
        self.model = model
        self.valid_dataset = valid_dataset
        self.train_loader = DataLoader(train_dataset, config["batch_size"], shuffle=True)
        self.valid_loader = DataLoader(valid_dataset, config["batch_size"])
        
        self.criterion = nn.BCELoss()
        self.optimizer = AdamW(self.model.parameters(), lr=config["lr"])
        self.best_model = None
        self.best_f1 = 0
    
    def train(self):
        wandb.init(
            project=self.config["wandb_project"], 
            entity=self.config["wandb_entity"], 
            name=self.config["wandb_name"],
            group=self.config["wandb_group"],
            config=self.config
        )
        
        for epoch in tqdm(range(self.config["epochs"])):
            self.model.train()
            for data in self.train_loader:
                inputs = data['input_ids'].to('cuda')
                label = data['label'].to('cuda').float()
                outputs = self.model(inputs).squeeze()
                
                loss = self.criterion(outputs, label)
                wandb.log({"train loss": loss, "epochs": epoch})
                loss.backward()
                self.optimizer.step()
                self.model.zero_grad()
                
            valid_loss = 0
            preds = []
            self.model.eval()
            for data in self.valid_loader:
                inputs = data['input_ids'].to('cuda')
                label = data['label'].to('cuda').float()
                with torch.no_grad():
                    outputs = self.model(inputs).squeeze()
                loss = self.criterion(outputs, label)
                valid_loss += loss
                
                preds += [1 if output.item() > 0.5 else 0 for output in outputs.cpu()]
                
            f1 = f1_score(preds, self.valid_dataset.labels)
            wandb.log({"valid loss": valid_loss/len(self.valid_loader), "f1 score": f1})
            
            if f1 > self.best_f1:
                self.best_f1 = f1
                self.best_model = self.model.state_dict()
            
        best_model_path = os.path.join(self.config["checkpoint_dir"], "pytorch_model.bin")
        torch.save(self.best_model.state_dict(), best_model_path)
        