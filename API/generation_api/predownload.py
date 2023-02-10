import torch
from transformers import (
    T5ForConditionalGeneration,
)

model = T5ForConditionalGeneration.from_pretrained("happy06/KcT5-purificate")