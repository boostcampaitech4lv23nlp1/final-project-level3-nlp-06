from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn as nn
import torch
from sklearn.metrics import f1_score

from tqdm import tqdm
import wandb


class CNNTrainer:
    def __init__(self, config, model, train_dataset, valid_dataset) -> None:
        self.config = config
        self.model = model
        self.train_loader = DataLoader(train_dataset, config["batch_size"], shuffle=True)
        self.valid_loader = DataLoader(valid_dataset, config["batch_size"])
        
        self.criterion = nn.BCELoss()
        self.optimizer = AdamW(self.model.parameters(), lr=config["lr"])
    
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
                batch_size = data['input_ids'].shape[0]
                inputs = data['input_ids']
                label = data['label']
                inputs = inputs.to('cuda')
                label = label.to('cuda').float()
                outputs = self.model(inputs).reshape(batch_size, 1)
                
                loss = self.criterion(outputs, label)
                wandb.log({"train loss": loss, "epochs": epoch})
                loss.backward()
                self.optimizer.step()
                self.model.zero_grad()
                
            valid_loss = 0
            preds = []
            self.model.eval()
            for data in self.valid_loader:
                inputs = data['input_ids']
                label = data['label']
                inputs = inputs.to('cuda')
                label = label.to('cuda').float()
                with torch.no_grad():
                    outputs = self.model(inputs).reshape(batch_size, 1)
                preds += [1 if output.item() > 0.5 else 0 for output in outputs.cpu()]
                loss = self.criterion(outputs, label)
                valid_loss += loss
            f1 = f1_score(preds, self.valid_dataset.labels)
            wandb.log({"valid loss": valid_loss/len(self.valid_loader), "f1 score": f1})
        