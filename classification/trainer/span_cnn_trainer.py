from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn as nn
import torch
from sklearn.metrics import f1_score

from tqdm import tqdm
import wandb
import os
from transformers import DataCollatorForTokenClassification

## TODO: CNN Trainer도 huggingface trainer로 합치기.
class SpanCNNTrainer:
    def __init__(self, config, model, train_dataset, valid_dataset) -> None:
        self.config = config
        self.model = model
        self.valid_dataset = valid_dataset
        data_collator = DataCollatorForTokenClassification(tokenizer=train_dataset.tokenizer)
        
        self.train_loader = DataLoader(train_dataset, config["batch_size"], shuffle=True, collate_fn=data_collator)
        self.valid_loader = DataLoader(valid_dataset, config["batch_size"], collate_fn=data_collator)
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.optimizer = AdamW(self.model.parameters(), lr=config["lr"])
        self.best_model = None
        self.best_f1 = 0
        
    def get_accuracy(self, preds, labels):
        corrects = []
        for label, pred in zip(labels, preds):
            correct = 0
            for l, p in zip(label, pred):
                l = 1 if l > 0.5 else 0
                if l == p:
                    correct += 1
            correct /= 128
            corrects.append(correct)
        return sum(corrects) / len(preds)
    
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
                label = data['labels'].to('cuda').float()
                outputs = self.model(inputs)
                
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
                label = data['labels'].to('cuda').float()
                with torch.no_grad():
                    outputs = self.model(inputs)
                loss = self.criterion(outputs, label)
                valid_loss += loss
                preds.append(outputs.cpu())
                
            preds = torch.cat(preds, dim=0)
            acc = self.get_accuracy(preds, self.valid_dataset.labels)
            wandb.log({"valid loss": valid_loss/len(self.valid_loader), "Accuracy": acc})
            
            if acc > self.best_f1:
                self.best_f1 = acc
                self.best_model = self.model.state_dict().copy()
        
        print("best accuracy score :", self.best_f1)
        best_model_path = os.path.join(self.config["checkpoint_dir"], "pytorch_model.bin")
        torch.save(self.best_model, best_model_path)
        