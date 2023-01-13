import yaml
import torch
import argparse

from data import Apeach_Dataset, kmhas_Dataset, KOLD_Dataset, Beep_Dataset, Unsmile_Dataset
from model import CNNModel, transformer
from trainer import HuggingfaceTrainer, CNNTrainer


Dataset = {"APEACH": Apeach_Dataset, "BEEP!": Beep_Dataset, "Unsmile": Unsmile_Dataset, "k-mhas": kmhas_Dataset, "KOLD": KOLD_Dataset}
models = {"CNN": CNNModel, "Transformer": transformer}
trainers = {"CNN": CNNTrainer, "Transformer": HuggingfaceTrainer}


def main(config):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    train_dataset = Dataset[config["dataset"]](config["train_dir"], config["model_name"])
    valid_dataset = Dataset[config["dataset"]](config["valid_dir"], config["model_name"])
    model = models[config["model"]](config, train_dataset.tokenizer.vocab_size)
    model.to(device)
    
    trainer = trainers[config["model"]](config, model, train_dataset, valid_dataset)

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hate Speech Classification')
    parser.add_argument("--conf", type=str, default="config.yaml", help="config file path(.yaml)")
    args = parser.parse_args()
    with open(args.conf, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)
    main(config)
    