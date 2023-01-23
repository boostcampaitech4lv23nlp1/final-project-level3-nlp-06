from transformers import DataCollatorForTokenClassification, get_linear_schedule_with_warmup
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
        data_collator = DataCollatorForTokenClassification(tokenizer=train_dataset.tokenizer)
        self.train_loader = DataLoader(train_dataset, config["batch_size"], shuffle=True, collate_fn=data_collator)
        self.valid_loader = DataLoader(valid_dataset, config["batch_size"], collate_fn=data_collator)
        
        self.seq_criterion = nn.BCELoss()
        self.token_criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
        self.optimizer = AdamW(self.model.parameters(), lr=config["lr"])
        
        t_total = self.config["epochs"]*len(self.train_loader)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(self.config["warmup_ratio"]*t_total),
            num_training_steps=t_total
        )
        
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
                seq_label = data.pop('class_label')
                token_label = data.pop('labels')
                
                token_output, seq_output = self.model(**data)
                
                token_loss = self.token_criterion(token_output.transpose(1, 2), token_label)
                seq_loss = self.seq_criterion(seq_output.squeeze(), seq_label.float())
                total_loss = token_loss + seq_loss
                
                wandb.log({"token loss": token_loss, "sequence loss": seq_loss, "epochs": epoch})
                total_loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                self.model.zero_grad()
                
            valid_loss = 0
            seq_preds = []
            token_preds = []
            
            self.model.eval()
            for data in self.valid_loader:
                data = {k: v.to('cuda') for k, v in data.items()}
                seq_label = data.pop('class_label').float()
                token_label = data.pop('labels')
                with torch.no_grad():
                    token_output, seq_output = self.model(**data)
                token_loss = self.token_criterion(token_output.transpose(1, 2), token_label)
                seq_loss = self.seq_criterion(seq_output.squeeze(), seq_label.float())
                valid_loss += token_loss + seq_loss
                
                seq_preds += [1 if output.item() > 0.5 else 0 for output in seq_output.cpu()]
                token_preds.append(torch.argmax(token_output, dim=-1).cpu())
            
            token_preds = torch.cat(token_preds, dim=0)
            f1 = f1_score(seq_preds, self.valid_dataset.class_labels)
            metric = compute_metrics(token_preds, self.valid_dataset.labels)
            metric["f1 score"] = f1
            metric["valid loss"] = valid_loss/len(self.valid_loader)
            wandb.log(metric)
            
            if metric["valid loss"] < self.best_loss:
                self.best_loss = metric["valid loss"]
                self.best_model = self.model.state_dict().copy()

        best_model_path = os.path.join(self.config["checkpoint_dir"], "pytorch_model.bin")
        torch.save(self.best_model, best_model_path)
        
def compute_metrics(predictions, labels):
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
        "Token Accuracy": (n_cor+p_cor)/(n_cnt+p_cnt),
        "hate token accuracy": n_cor/n_cnt,
        "none hate token accuracy": p_cor/p_cnt
    }
