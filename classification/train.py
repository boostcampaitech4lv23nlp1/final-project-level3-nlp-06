import yaml
import torch
import random
import argparse

from data import *
from model import *
from trainer import *


Dataset = {"APEACH": Apeach_Dataset, "BEEP!": Beep_Dataset, "Unsmile": Unsmile_Dataset, 
           "k-mhas": kmhas_Dataset, "KOLD": KOLD_Dataset, "KOLD_SPAN": Span_Dataset, "KOLD_Sequence_SPAN": Sequence_Span_Dataset}
models = {"CNN": CNNModel, "VerifiableCNN": VerifiableCNN, "Transformer": transformer, 
          "SpanTransformer": span_transformer, "TokenSequenceTransformer": Token_Sequence_transformer}
trainers = {"CNN": CNNTrainer, "VerifiableCNN": CNNTrainer, "Transformer": HuggingfaceTrainer, 
            "SpanTransformer": HuggingfaceTrainer, "TokenSequenceTransformer": TokenSequenceTrainer}


def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(random_seed)
    
    
def main(config):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    set_seed(81)

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
    
