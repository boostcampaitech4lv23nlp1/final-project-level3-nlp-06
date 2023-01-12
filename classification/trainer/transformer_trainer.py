from transformers import TrainingArguments, Trainer
from utils import Compute_metrics
import wandb


class HuggingfaceTrainer:
    def __init__(self, config, model, train_dataset, valid_dataset):
        self.config = config
        CM = Compute_metrics(multi_label=config["multi_label"], num_labels=config["num_labels"])
        compute_metrics = CM.compute_metrics

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
            logging_steps=100,
            evaluation_strategy='steps',
            eval_steps=config["eval_step"],
            load_best_model_at_end=True,
            report_to='wandb',
        )

        self.trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            compute_metrics=compute_metrics
        )
        
    def train():
        wandb.init(
            project=self.config["wandb_project"], 
            entity=self.config["wandb_entity"], 
            name=self.config["wandb_name"],
            group=self.config["wandb_group"],
            config=self.config
        )
        self.trainer.train()
        