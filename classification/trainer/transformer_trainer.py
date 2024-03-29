from transformers import TrainingArguments, Trainer, DataCollatorForTokenClassification
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
import wandb


class HuggingfaceTrainer:
    def __init__(self, config, model, train_dataset, valid_dataset):
        self.config = config
        self.labels = valid_dataset.labels

        training_args = TrainingArguments(
            output_dir=config["checkpoint_dir"],
            save_total_limit=2,
            save_steps=config["save_step"],
            num_train_epochs=config["epochs"],
            learning_rate=config["lr"],
            per_device_train_batch_size=config["batch_size"],
            per_device_eval_batch_size=config["batch_size"],
            warmup_ratio=0.5,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=config["eval_step"],
            evaluation_strategy="steps",
            eval_steps=config["eval_step"],
            load_best_model_at_end=True,
            report_to='wandb',
        )
        
        if self.config["model"]=="SpanTransformer":
            data_collator = DataCollatorForTokenClassification(tokenizer=train_dataset.tokenizer)
            self.trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=valid_dataset,
                data_collator=data_collator,
                compute_metrics=self.compute_metrics
            )
        else:
            self.trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=valid_dataset,
                compute_metrics=self.calc_f1_score
            )
            
    def calc_f1_score(self, p):
        predictions, labels = p
        preds = [1 if pred > 0.5 else 0 for pred in predictions]
        f1 = f1_score(preds, labels, average="micro")
        precision = precision_score(preds, labels)
        recall = recall_score(preds, labels)
        return {
            "f1 score": f1,
            "precision": precision,
            "recall": recall
        }
        
    def compute_metrics(self, p):
        predictions, labels = p
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
        
    def train(self):
        wandb.init(
            project=self.config["wandb_project"], 
            entity=self.config["wandb_entity"], 
            name=self.config["wandb_name"],
            group=self.config["wandb_group"],
            config=self.config
        )
        self.trainer.train()
        