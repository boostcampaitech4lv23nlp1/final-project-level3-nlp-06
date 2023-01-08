import torch
from torch.optim import AdamW
import torch.nn.functional as F
import os
import wandb
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup


class Trainer:
    def __init__(self, config, model, train_loader, valid_loader=None):
        self.config = config
        self.model = model
        self.device = config["device"]
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        
        # Optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": self.config["weight_decay"]},
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.config["learning_rate"],
            eps=1e-8
        )
        
    def forwawrd_step(self, negative_batch, positive_batch):
        negative_batch = {k: v.to(self.device) for k, v in negative_batch.items()}
        positive_batch = {k: v.to(self.device) for k, v in positive_batch.items()}
        negative_outputs = self.model(**negative_batch)[1]
        positive_outputs = self.model(**positive_batch)[1]
        score = torch.matmul(negative_outputs, positive_outputs.T)
        score = F.log_softmax(score, dim=-1)
        
        return negative_outputs, positive_outputs, score
    
    def train(self):
        wandb.init(
            project=self.config["wandb_project"], 
            name=self.config["wandb_name"], 
            notes=self.config["wandb_note"], 
            entity=self.config["wandb_entity"], 
            group=self.config["wandb_group"],
            config=self.config
        )
        epochs = self.config["epochs"]
        
        t_total = len(self.train_loader) * epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(self.config["warmup_ratio"]*t_total),
            num_training_steps=t_total
        )
        
        self.model.to(self.device)
        global_step = 0
        for epoch in range(epochs):
            print(f"epoch : {epoch}/{epochs-1}")
            train_loss = 0
            self.model.train()
            for data in tqdm(self.train_loader):
                batch_size = data[0]['input_ids'].shape[0]
                _, _, score = self.forwawrd_step(data[0], data[1])
                labels = torch.arange(0, batch_size).long().to(self.device)
                
                loss = F.nll_loss(score, labels)
                train_loss += loss
                
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                self.model.zero_grad()
                
                if (global_step != 0) and (global_step%self.config["eval_step"] == 0):
                    train_loss /= global_step
                    wandb.log({"train loss": train_loss, "epochs": epoch})
                    self.validation(global_step)
                global_step += 1
    
    def validation(self, step):
        self.model.eval()
        valid_loss = 0
        valid_step = 0
        print(f"validate in {step} step...")
        for data in tqdm(self.valid_loader):
            batch_size = data[0]['input_ids'].shape[0]
            with torch.no_grad():
                _, _, score = self.forwawrd_step(data[0], data[1])
                labels = torch.arange(0, batch_size).long().to(self.device)
                loss = F.nll_loss(score, labels)
                valid_loss += loss
            valid_step += 1
        valid_loss /= valid_step
        save_name = os.path.join(self.config["model_save_path"], f"model-{step}-{round(valid_loss.item(), 4)}.bin")
        torch.save(self.model.state_dict, save_name)
        print("valid loss :", valid_loss)
        wandb.log({"validation loss": valid_loss})
        
    def evaluate(self):
        ## TODO: 모델 평가 어떻게 할거야?
        pass
    