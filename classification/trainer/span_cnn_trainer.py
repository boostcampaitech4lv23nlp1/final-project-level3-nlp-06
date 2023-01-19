from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn as nn
import torch
from sklearn.metrics import accuracy_score

from tqdm import tqdm
import numpy as np
import wandb
import os
from transformers import DataCollatorForTokenClassification

## TODO: CNN Trainer도 huggingface trainer로 합치기.
class SpanCNNTrainer:
    def __init__(self, config, model, train_dataset, valid_dataset) -> None:
        self.config = config
        self.model = model
        self.valid_dataset = valid_dataset
        
        self.train_loader = DataLoader(train_dataset, config["batch_size"], shuffle=True, collate_fn=span_collator)
        self.valid_loader = DataLoader(valid_dataset, config["batch_size"], collate_fn=span_collator)
        
        self.criterion1 = nn.CrossEntropyLoss(ignore_index=-100)
        self.criterion2 = nn.BCELoss()
        self.optimizer = AdamW(self.model.parameters(), lr=config["lr"])
        self.best_model = None
        self.best_loss = 100
    
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
                data = {k: v.to('cuda') for k, v in data.items()}
                token_label = data.pop('token_labels')
                class_label = data.pop('class_labels').float()
                token_output, class_output = self.model(data['input_ids'], data['attention_mask'])
                
                token_loss = self.criterion1(token_output, token_label)
                class_loss = self.criterion2(class_output, class_label)
                loss = token_loss + class_loss
                wandb.log({
                    "train loss": loss, 
                    "train token loss": token_loss,
                    "train class loss": class_loss,
                    "epochs": epoch
                })
                loss.backward()
                self.optimizer.step()
                self.model.zero_grad()
                
            valid_loss = 0
            valid_token_loss = 0
            valid_class_loss = 0
            
            token_outputs = []
            class_outputs = []
            
            self.model.eval()
            for data in self.valid_loader:
                data = {k: v.to('cuda') for k, v in data.items()}
                token_label = data.pop('token_labels')
                class_label = data.pop('class_labels').float()
                with torch.no_grad():
                    token_output, class_output = self.model(data['input_ids'], data['attention_mask'])
                token_loss = self.criterion1(token_output, token_label)
                class_loss = self.criterion2(class_output, class_label)
                
                valid_token_loss += token_loss
                valid_class_loss += class_loss
                valid_loss += loss
                
                token_outputs.append(token_output.cpu())
                class_outputs += class_output.tolist()
                
            token_outputs = torch.cat(token_outputs, dim=0)
            pred_labels = [1 if output > 0.5 else 0 for output in class_outputs]
            class_acc = accuracy_score(pred_labels, self.valid_dataset.class_labels)
            token_acc = self.compute_metrics(token_outputs, self.valid_dataset.labels)
            valid_loss /= len(self.valid_loader)
            wandb.log({
                "valid loss": valid_loss, 
                "valid token loss": valid_token_loss/len(self.valid_loader),
                "valid class loss": valid_class_loss/len(self.valid_loader),
                "class accuracy": class_acc,
                "token accuracy": token_acc["Accuracy"],
                "hate token accuracy": token_acc["hate token accuracy"],
                "none hate token accuracy": token_acc["none hate token accuracy"]
            })
            
            if valid_loss < self.best_loss:
                self.best_loss = valid_loss
                self.best_model = self.model.state_dict().copy()
        
        print("best accuracy score :", self.best_f1)
        best_model_path = os.path.join(self.config["checkpoint_dir"], "pytorch_model.bin")
        torch.save(self.best_model, best_model_path)
        
    def compute_metrics(self, predictions, labels):
        predictions = np.argmax(predictions, axis=2)
        
        true_predictions = [
            [p.item() for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [l for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        n_cnt = 0
        p_cnt = 0

        n_cor = 0
        p_cor = 0

        for pred, label in zip(true_predictions, true_labels):
            for p, l in zip(pred, label):
                if l == 0:
                    p_cnt += 1
                    if p == l:
                        p_cor += 1
                else:
                    n_cnt += 1
                    if p == l:
                        n_cor += 1
        return {
            "Accuracy": (n_cor+p_cor)/(n_cnt+p_cnt),
            "hate token accuracy": n_cor/n_cnt,
            "none hate token accuracy": p_cor/p_cnt
        }
        
def span_collator(batch):
    input_ids = []
    attention_masks = []
    token_labels = []
    class_labels = []
    for data in batch:
        input_ids.append(data['input_ids'])
        attention_masks.append(data['attention_mask'])
        token_labels.append(torch.tensor(data['labels']))
        class_labels.append(int(data['class_label']))
    
    input_ids = torch.stack(input_ids)
    attention_masks = torch.stack(attention_masks)
    token_labels = torch.stack(token_labels)
    class_labels = torch.tensor(class_labels)
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_masks,
        'token_labels': token_labels,
        'class_labels': class_labels
    }
    