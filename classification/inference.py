import torch
from torch.utils.data import DataLoader
import yaml
import argparse
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score

from data import Apeach_Dataset, kmhas_Dataset, KOLD_Dataset
from model import CNNModel, transformer


Dataset = {"APEACH": Apeach_Dataset, "k-mhas": kmhas_Dataset, "KOLD": KOLD_Dataset}
models = {"CNN": CNNModel, "Transformer": transformer}

def main(config):
    valid_dataset = Dataset[config["dataset"]](config["valid_dir"], config["model_name"])
    valid_loader = DataLoader(valid_dataset, batch_size=config["batch_size"])
    model = models[config["model"]](config, valid_dataset.tokenizer.vocab_size)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    
    preds = []
    model.eval()
    for data in tqdm(valid_loader):
        batch_size = data['input_ids'].shape[0]
        data = {k: v.to('cuda') for k, v in data.items()}
        label = data.pop('label')
        with torch.no_grad():
            if config["model"] == "CNN":
                outputs = model(data).reshape(batch_size, 1)
            else:
                outputs = model(**data).logits
        preds += [output.item() for output in outputs.cpu()]
    f1 = f1_score(preds, valid_dataset.labels)
    print("f1 score :", f1)
    df = {
        "sentence": valid_dataset.sentences,
        "label": valid_dataset.labels,
        "preds": preds
    }
    pd.DataFrame.from_dict(df).to_csv(config["result_dir"], index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hate Speech Classification')
    parser.add_argument("--conf", type=str, default="inference_config.yaml", help="config file path(.yaml)")
    args = parser.parse_args()
    with open(args.conf, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)
    main(config)
    