import torch
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score
import wandb
import json

from data import Apeach_Dataset
from utils import calc_f1_score, Auprc


def main(config):
    model = AutoModelForSequenceClassification.from_pretrained(config.model_name, num_labels=config.num_labels)

    # set device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    # Dataset
    train_dataset = Apeach_Dataset(config.train_dir, config.model_name)
    valid_dataset = Apeach_Dataset(config.valid_dir, config.model_name)


    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        probs = pred.predictions
        
        f1 = calc_f1_score(preds, labels)
        auprc_obj = Auprc(train_dataset.num_labels)
        auprc = auprc_obj.calc(probs, labels)
        acc = accuracy_score(labels, preds)
        
        return {
            "micro f1 score": f1,
            "auprc": auprc,
            "accuracy": acc
        }
        

    wandb.init(project=config.wandb_project, entity=config.wandb_entity, name=config.wandb_name)

    training_args = TrainingArguments(
        output_dir="./results",
        save_total_limit=2,
        save_steps=config.save_step,
        num_train_epochs=config.epochs,
        learning_rate=config.lr,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        warmup_ratio=0.5,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=100,
        evaluation_strategy='steps',
        eval_steps=config.eval_step,
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
    with open("config.json", "r") as f:
        config = json.load(f)
    main(config)
