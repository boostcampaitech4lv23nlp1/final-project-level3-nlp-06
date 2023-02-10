import torch
import wandb
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (
    Trainer,
    AutoModel,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorWithPadding,
    AutoModelForSequenceClassification,
)
from sklearn.metrics import mean_squared_error


class RewardDataset(Dataset):
    def __init__(self, config, tokenizer, eval=False):
        self.config = config
        self.tokenizer = tokenizer

        if eval:
            self.data = pd.read_csv("/opt/ml/InstructBART/reward_result_v1.0.0_valid.csv")
        else:
            self.data = pd.read_csv("/opt/ml/InstructBART/reward_result_v1.0.0_train.csv")

        self.text = self.data["text"].tolist()
        self.label1 = self.data["label1"].tolist()
        self.label2 = self.data["label2"].tolist()
        self.label3 = self.data["label3"].tolist()
        self.target = self.data["target"].tolist()
        self.reward1 = self.data["reward1"].tolist()
        self.reward2 = self.data["reward2"].tolist()
        self.reward3 = self.data["reward3"].tolist()
        self.reward4 = self.data["reward4"].tolist()
        self.preprocess()

    def preprocess(self):
        self.tokenized_inputs = {"input_ids": [], "attention_mask": [], "token_type_ids": [], "label": []}
        for i in range(len(self.text)):
            orig = self.text[i]
            label1 = self.label1[i]
            label2 = self.label2[i]
            label3 = self.label3[i]
            target = self.target[i]

            tokenized_label1 = self.tokenizer(
                orig,
                label1,
                max_length=384,
                padding="max_length",
            )
            tokenized_label2 = self.tokenizer(
                orig,
                label2,
                max_length=384,
                padding="max_length",
            )
            tokenized_label3 = self.tokenizer(
                orig,
                label3,
                max_length=384,
                padding="max_length",
            )
            tokenized_target = self.tokenizer(
                orig,
                target,
                max_length=384,
                padding="max_length",
            )

            self.tokenized_inputs["input_ids"].append(tokenized_label1["input_ids"])
            self.tokenized_inputs["attention_mask"].append(tokenized_label1["attention_mask"])
            self.tokenized_inputs["token_type_ids"].append(tokenized_label1["token_type_ids"])
            self.tokenized_inputs["label"].append(self.reward1[i] - 1)

            self.tokenized_inputs["input_ids"].append(tokenized_label2["input_ids"])
            self.tokenized_inputs["attention_mask"].append(tokenized_label2["attention_mask"])
            self.tokenized_inputs["token_type_ids"].append(tokenized_label2["token_type_ids"])
            self.tokenized_inputs["label"].append(self.reward2[i] - 1)

            self.tokenized_inputs["input_ids"].append(tokenized_label3["input_ids"])
            self.tokenized_inputs["attention_mask"].append(tokenized_label3["attention_mask"])
            self.tokenized_inputs["token_type_ids"].append(tokenized_label3["token_type_ids"])
            self.tokenized_inputs["label"].append(self.reward3[i] - 1)

            self.tokenized_inputs["input_ids"].append(tokenized_target["input_ids"])
            self.tokenized_inputs["attention_mask"].append(tokenized_target["attention_mask"])
            self.tokenized_inputs["token_type_ids"].append(tokenized_target["token_type_ids"])
            self.tokenized_inputs["label"].append(self.reward4[i] - 1)

    def __len__(self):
        return len(self.tokenized_inputs["input_ids"])

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.tokenized_inputs["input_ids"][idx]),
            "attention_mask": torch.tensor(self.tokenized_inputs["attention_mask"][idx]),
            "token_type_ids": torch.tensor(self.tokenized_inputs["token_type_ids"][idx]),
            "label": torch.tensor(self.tokenized_inputs["label"][idx]).type(torch.FloatTensor),
        }


def main():
    wandb.init(
        entity="ymnseol",
        project="reward",
    )

    model = AutoModelForSequenceClassification.from_pretrained("beomi/KcELECTRA-base-v2022", num_labels=1)
    model.to("cuda")
    tokenizer = AutoTokenizer.from_pretrained("beomi/KcELECTRA-base-v2022")

    train_dataset = RewardDataset(None, tokenizer, eval=False)
    eval_dataset = RewardDataset(None, tokenizer, eval=True)

    training_args = TrainingArguments(
        output_dir="/opt/ml/InstructBART/result",
        evaluation_strategy="steps",
        eval_steps=10,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        learning_rate=5e-5,
        num_train_epochs=1,
        logging_steps=5,
        save_steps=5,
        save_total_limit=2,
        report_to=["wandb"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()
    trainer.save_model()
    torch.save(model.state_dict(), "/opt/ml/InstructBART/result/model.pt")

    test_dataset = RewardDataset(None, tokenizer, True)
    result = trainer.predict(test_dataset)
    pd.DataFrame({"pred": [r for r in result.predictions], "label": result.label_ids}).to_csv("/opt/ml/InstructBART/model_output_valid.csv")

if __name__ == "__main__":
    main()
