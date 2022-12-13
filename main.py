import torch
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, roc_auc_score
import wandb
import yaml
import argparse

from data import Apeach_Dataset, kmhas_Dataset
from utils import Compute_metrics


Dataset = {"APEACH": Apeach_Dataset, "kmhas": kmhas_Dataset}

def main(config):
    if config["multi_label"]:
        model = AutoModelForSequenceClassification.from_pretrained(
            config["model_name"], 
            num_labels=config["num_labels"], 
            problem_type="multi_label_classification"
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            config["model_name"], 
            num_labels=config["num_labels"]
        )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    train_dataset = Dataset[config["dataset"]](config["train_dir"], config["model_name"])
    valid_dataset = Dataset[config["dataset"]](config["valid_dir"], config["model_name"])
    
    CM = Compute_metrics(multi_label=config["multi_label"], num_labels=config["num_labels"])
    compute_metrics = CM.compute_metrics

    wandb.init(project=config["wandb_project"], entity=config["wandb_entity"], name=config["wandb_name"])

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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hate Speech Classification')
    parser.add_argument("--conf", type=str, default="config.yaml", help="config file path(.yaml)")
    args = parser.parse_args()
    with open(args.conf, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)
    main(config)
    