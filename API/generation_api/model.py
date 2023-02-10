import torch
from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
)
from pydantic import BaseModel


class Hatespeech(BaseModel):
    sentence: str


class GenerationModel:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model = T5ForConditionalGeneration.from_pretrained("happy06/KcT5-purificate")
        model.eval()
        model.to(self.device)
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained("beomi/KcT5-dev")

    def purification(self, word):
        inputs = self.tokenizer([word], max_length=300,truncation=True)
        summary_ids = self.model.generate(
            torch.tensor(inputs["input_ids"]).to(self.device),
            max_length=torch.tensor(inputs["input_ids"]).size(-1) * 2,
            early_stopping=True,
            repetition_penalty=2.0,
            temperature=0.8,
            forced_bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        result = self.tokenizer.decode(summary_ids.squeeze().detach().tolist(), skip_special_tokens=True)
        return result
