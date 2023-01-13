from transformers import TrainingArguments, Trainer
from sklearn.metrics import f1_score
import wandb


class HuggingfaceTrainer:
    def __init__(self, config, model, train_dataset, valid_dataset):
        self.config = config
        self.labels = valid_dataset.label

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

        self.trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            ## TODO: calculate f1 score
            # compute_metrics=self.calc_f1_score
        )
        
    def calc_f1_score(self, preds):
        return f1_score(self.labels, preds, average="micro") * 100.0
        
    def train(self):
        wandb.init(
            project=self.config["wandb_project"], 
            entity=self.config["wandb_entity"], 
            name=self.config["wandb_name"],
            group=self.config["wandb_group"],
            config=self.config
        )
        self.trainer.train()
        