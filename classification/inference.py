import torch
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import yaml
import argparse
import numpy as np

from data import Apeach_Dataset, kmhas_Dataset
from utils import Compute_metrics, get_prediction


Dataset = {"APEACH": Apeach_Dataset, "kmhas": kmhas_Dataset}

def main(config):
    if config["multi_label"]:
        model = AutoModelForSequenceClassification.from_pretrained(
            config["checkpoint_dir"], 
            num_labels=config["num_labels"], 
            problem_type="multi_label_classification"
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            config["checkpoint_dir"], 
            num_labels=config["num_labels"]
        )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    valid_dataset = Dataset[config["dataset"]](config["valid_dir"], config["model_name"])

    CM = Compute_metrics(multi_label=config["multi_label"], num_labels=config["num_labels"])
    compute_metrics = CM.compute_metrics
        
    trainer = Trainer(
        model=model,
        compute_metrics=compute_metrics
    )

    result = trainer.predict(valid_dataset)
    print(result.metrics)
    
    preds = get_prediction(result, config["label_map_dir"], config["multi_label"])
    
    df = valid_dataset.df
    df["pred"] = preds
    probs = torch.sigmoid(torch.tensor(result.predictions))
    df["probs"] = [str(np.round(p, 2).tolist()) for p in probs.tolist()]
    df.to_csv(config["result_dir"])
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hate Speech Classification')
    parser.add_argument("--conf", type=str, default="inference_config.yaml", help="config file path(.yaml)")
    args = parser.parse_args()
    with open(args.conf, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)
    main(config)
    