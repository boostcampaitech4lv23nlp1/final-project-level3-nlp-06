import torch
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, roc_auc_score
import yaml
import argparse
import numpy as np
import pickle

from data import Apeach_Dataset, kmhas_Dataset
from utils import calc_f1_score, Auprc


def main(config):
    model = AutoModelForSequenceClassification.from_pretrained(config["checkpoint_dir"], num_labels=config["num_labels"], problem_type="multi_label_classification")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    valid_dataset = kmhas_Dataset(config["valid_dir"], config["model_name"])

    def compute_metrics(pred):
        sigmoid = torch.nn.Sigmoid()
        
        labels = pred.label_ids
        probs = sigmoid(torch.tensor(pred.predictions))
        preds = torch.zeros(probs.shape)
        preds[torch.where(probs >= 0.5)] = 1
        
        f1 = calc_f1_score(preds, labels)
        auprc = roc_auc_score(labels, preds, average="macro")
        acc = accuracy_score(labels, preds)
        
        return {
            "micro f1 score": f1,
            "auprc": auprc*100,
            "accuracy": acc
        }
        
    trainer = Trainer(
        model=model,
        compute_metrics=compute_metrics
    )

    result = trainer.predict(valid_dataset)
    
    preds = [np.argmax(pred) for pred in result.predictions]
    
    with open(config["label_map_dir"], "rb") as f:
        label_map = pickle.load(f)
    preds = [label_map[pred] for pred in preds]
        
    df = valid_dataset.df
    df["pred"] = preds
    df.to_csv(config["result_dir"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hate Speech Classification')
    parser.add_argument("--conf", type=str, default="inference_config.yaml", help="config file path(.yaml)")
    args = parser.parse_args()
    with open(args.conf, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)
    main(config)
    