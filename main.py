import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, roc_auc_score
import wandb

from data import Apeach_Dataset, kmhas_Dataset
from utils import calc_f1_score, Auprc


## TODO: Config 만들어서 아래 옵션들 설정하도록 하기. 귀찮으니까 일단 아래거 수정해서 사용.
num_labels=9
model_name = "monologg/koelectra-base-v3-discriminator"
wandb_project="apeach"
wandb_entity="intrandom5"
wandb_name="huggingface_kmhas"
train_dir = "csv_files/kmhas-train.csv"
valid_dir = "csv_files/kmhas-valid.csv"
batch_size = 32
epochs = 4
lr = 5e-5
save_step = 300
eval_step = 300


tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, problem_type="multi-label-classification")

# set device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# Dataset
train_dataset = kmhas_Dataset(train_dir, model_name)
valid_dataset = kmhas_Dataset(valid_dir, model_name)


# def compute_metrics(pred):
#     labels = pred.label_ids
#     preds = pred.predictions.argmax(-1)
#     probs = pred.predictions
    
#     f1 = calc_f1_score(preds, labels)
#     auprc_obj = Auprc(train_dataset.num_labels)
#     auprc = auprc_obj.calc(probs, labels)
#     acc = accuracy_score(labels, preds)
    
#     return {
#         "micro f1 score": f1,
#         "auprc": auprc,
#         "accuracy": acc
#     }

def compute_metrics(pred):
    sigmoid = torch.nn.Sigmoid()
    
    labels = pred.label_ids
    probs = sigmoid(pred.predictions)
    preds = torch.zeros(probs.shape)
    preds[torch.where(probs >= 0.5)] = 1
    
    f1 = calc_f1_score(preds, labels)
    auprc = roc_auc_score(labels, preds, average="macro")
    acc = accuracy_score(labels, preds)
    
    return {
        "micro f1 score": f1,
        "auprc": auprc,
        "accuracy": acc
    }
    

wandb.init(project=wandb_project, entity=wandb_entity, name=wandb_name)


training_args = TrainingArguments(
    output_dir="./results/kmhas",
    save_total_limit=2,
    save_steps=save_step,
    num_train_epochs=epochs,
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    warmup_ratio=0.5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    evaluation_strategy='steps',
    eval_steps=eval_step,
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
