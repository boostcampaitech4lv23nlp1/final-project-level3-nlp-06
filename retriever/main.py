import yaml
import argparse
from trainer import Trainer
from dataset import KOLD_dataset
from transformers import AutoModel


def main(config):
    ## TODO: save model config at save_model_path for huggingface upload.
    model_config = AutoModel.from_pretrained(config["model_name"])
    model = AutoModel.from_pretrained(config["model_name"])
    train_loader = KOLD_dataset(config, config["train_path"])
    valid_loader = KOLD_dataset(config, config["valid_path"])
    trainer = Trainer(config, model, train_loader, valid_loader)
    trainer.train()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='config for retriever.')
    parser.add_argument("--conf", type=str, default="config.yaml", help="config file path(.yaml)")
    args = parser.parse_args()
    with open(args.conf, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)
    main(config)
    